import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import wandb
import numpy as np
import os
import csv
import json
from datetime import datetime
from torchvision.transforms import v2

# ====================================================
# DenseNet-121 Implementation
# ====================================================
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()
        self.drop_rate = drop_rate
        self.dense_layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        new_features = self.dense_layer(x)
        if self.drop_rate > 0:
            new_features = torch.nn.functional.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layer = DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size, drop_rate)
            layers.append(layer)
        self.dense_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.dense_block(x)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.transition_layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition_layer(x)

class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), bn_size=4, 
                 drop_rate=0, num_classes=100):
        super(DenseNet, self).__init__()
        
        # First convolution
        num_init_features = 2 * growth_rate
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = TransitionLayer(num_features, num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = torch.relu(features)
        out = torch.nn.functional.adaptive_avg_pool2d(out, 1)
        feature_vec = out.view(out.size(0), -1)
        logits = self.classifier(feature_vec)
        return feature_vec, logits

def densenet121(num_classes=100, drop_rate=0.0):
    return DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), bn_size=4, 
                    drop_rate=drop_rate, num_classes=num_classes)

# ====================================================
# Config - Optimized for DenseNet-121
# ====================================================
BATCH_SIZE = 64  # Smaller batch size due to memory constraints
EPOCHS = 400
LR = 0.1  # Standard LR for DenseNet
WEIGHT_DECAY = 1e-4  # DenseNet works well with this
MOMENTUM = 0.9
DROP_RATE = 0.2  # DenseNet benefits from dropout
LABEL_SMOOTHING = 0.1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_DIR = "./checkpoints_iid_densenet121"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
SEED = 42

# Load pre-computed expert indices from splits/
EXPERT_INDICES_PATH = "splits/expert_train_indices.npy"
MIXUP_ALPHA = 1.0
CUTMIX_ALPHA = 1.0
AUGMENTATION_MODE = 'cutmix'  # Can be 'mixup', 'cutmix', or None

# ====================================================
# Data Loaders - Same as augmented version
# ====================================================
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    transforms.RandomErasing(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

# Global test loader
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def get_shared_unique_indices(base_indices, shared_ratio=0.4, unique_ratio=0.15, seed=42):
    """Build 4 expert index sets from a base set of indices."""
    total_len = len(base_indices)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(base_indices) # Permute the provided base indices

    shared_size = int(round(shared_ratio * total_len))
    unique_size = int(round(unique_ratio * total_len))
    max_unique = max(0, (total_len - shared_size) // 4)
    if unique_size > max_unique:
        unique_size = max_unique

    shared_indices = perm[:shared_size]
    offset = shared_size
    expert_indices = []
    for i in range(4):
        start = offset + i * unique_size
        end = start + unique_size
        unique_i = perm[start:end]
        combined = np.concatenate([shared_indices, unique_i])
        expert_indices.append(combined)
    return expert_indices, shared_indices, [perm[shared_size + i*unique_size: shared_size + (i+1)*unique_size] for i in range(4)]

def load_expert_indices():
    """Load pre-computed expert indices and create IID distribution like train_iid_augmented.py"""
    if not os.path.exists(EXPERT_INDICES_PATH):
        raise FileNotFoundError(f"Expert indices file not found at: {EXPERT_INDICES_PATH}")
    
    master_expert_indices = np.load(EXPERT_INDICES_PATH)
    print(f"Loaded expert indices from {EXPERT_INDICES_PATH}")
    print(f"Master expert indices shape: {master_expert_indices.shape}")
    
    # Create IID splits from the master pool using same logic as train_iid_augmented.py
    expert_indices_list, shared_indices, unique_indices_list = get_shared_unique_indices(
        master_expert_indices, shared_ratio=0.4, unique_ratio=0.15, seed=SEED
    )
    
    print(f"Created IID splits:")
    print(f"  Shared samples: {len(shared_indices)} (40% of {len(master_expert_indices)})")
    for i, expert_indices in enumerate(expert_indices_list):
        print(f"  Expert {i}: {len(expert_indices)} total samples ({len(shared_indices)} shared + {len(unique_indices_list[i])} unique)")
    
    return expert_indices_list

# ====================================================
# Training Function
# ====================================================
def train_model(train_subset, model_id):
    trainloader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = densenet121(drop_rate=DROP_RATE).to(DEVICE)
    # Optimized optimizer settings for DenseNet-121
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    # Multi-step LR schedule works well for DenseNet
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 350], gamma=0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    wandb.init(project="PuGA", name=f"densenet121_expert_{model_id}", config={
        "model_id": model_id,
        "arch": "DenseNet-121",
        "dataset": "CIFAR-100",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "dropout": DROP_RATE,
        "augmentation": AUGMENTATION_MODE
    })

    best_acc = 0.0
    best_train_acc = 0.0
    best_epoch = -1
    epochs_no_improve = 0
    patience = 30  # As requested
    min_epochs = 50  # Minimum epochs before early stopping

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        pbar = tqdm(trainloader, desc=f"[DenseNet121 Expert {model_id}] Epoch {epoch+1}/{EPOCHS}")
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            if AUGMENTATION_MODE == 'mixup':
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, MIXUP_ALPHA, DEVICE)
                optimizer.zero_grad()
                _, outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            elif AUGMENTATION_MODE == 'cutmix':
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, CUTMIX_ALPHA, DEVICE)
                optimizer.zero_grad()
                _, outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                optimizer.zero_grad()
                _, outputs = model(inputs)
                loss = criterion(outputs, targets)

            loss.backward()
            # Gradient clipping for DenseNet stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            pbar.set_postfix(loss=loss.item(), acc=100.*correct/total)

        train_acc = 100.*correct/total
        train_loss = total_loss / total
        scheduler.step()

        if train_acc > best_train_acc:
            best_train_acc = train_acc

        # --- Evaluation ---
        model.eval()
        test_loss, test_correct, test_total = 0, 0, 0

        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                features, logits = model(inputs)
                loss = criterion(logits, targets)

                test_loss += loss.item() * targets.size(0)
                _, predicted = logits.max(1)
                test_correct += predicted.eq(targets).sum().item()
                test_total += targets.size(0)

        test_acc = 100.*test_correct/test_total
        test_loss /= test_total

        print(f"✅ Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        # WandB logging
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": scheduler.get_last_lr()[0]
        })

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"best_densenet121_expert_{model_id}.pth"))
        else:
            epochs_no_improve += 1

        # Early stopping
        if epoch + 1 >= min_epochs and epochs_no_improve >= patience:
            print(f"⛔ Early stopping at epoch {epoch+1}: no improvement for {patience} epochs.")
            break

    # --- Final Summary ---
    print(f"🏁 DenseNet121 Expert {model_id} finished.")
    print(f"📈 Best Test Accuracy: {best_acc:.2f}% at Epoch {best_epoch}")
    print(f"🧠 Best Train Accuracy: {best_train_acc:.2f}%")

    wandb.log({
        "best_test_acc": best_acc,
        "best_train_acc": best_train_acc,
        "best_epoch": best_epoch
    })

    wandb.finish()

    # CSV report
    report_path = os.path.join(CHECKPOINT_DIR, "report_densenet121.csv")
    fieldnames = [
        "timestamp",
        "expert_id",
        "train_samples",
        "best_test_acc",
        "best_train_acc",
        "best_epoch",
    ]
    row = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "expert_id": int(model_id),
        "train_samples": int(len(train_subset)),
        "best_test_acc": float(best_acc),
        "best_train_acc": float(best_train_acc),
        "best_epoch": int(best_epoch if best_epoch is not None else -1),
    }
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    write_header = not os.path.exists(report_path)
    with open(report_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

# ====================================================
# Run Expert 3 Training (4th expert, 0-indexed)
# ====================================================
if __name__ == "__main__":
    full_trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)

    # Load pre-computed expert indices
    expert_indices_list = load_expert_indices()

    # Train only expert 3 (4th expert, 0-indexed) - DenseNet121
    expert_id = 3
    expert_full_indices = expert_indices_list[expert_id]
    subset_full = Subset(full_trainset, expert_full_indices)
    
    print(f"Training DenseNet-121 Expert {expert_id} with {len(subset_full)} samples")
    train_model(subset_full, expert_id)
