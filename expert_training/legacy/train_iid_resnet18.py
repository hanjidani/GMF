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
# ResNet-18 Implementation
# ====================================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.nn.functional.adaptive_avg_pool2d(out, 1)
        features = out.view(out.size(0), -1)
        logits = self.linear(features)
        return features, logits

def resnet18(num_classes=100):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)

# ====================================================
# Config - Optimized for ResNet-18
# ====================================================
BATCH_SIZE = 128
EPOCHS = 400
LR = 0.2  # Higher LR for lighter model
WEIGHT_DECAY = 1e-4  # Reduced weight decay for lighter model
MOMENTUM = 0.9
DROP_RATE = 0.0
LABEL_SMOOTHING = 0.1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_DIR = "./checkpoints_iid_resnet18"
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

    model = resnet18().to(DEVICE)
    # Optimized optimizer settings for ResNet-18
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    # Warmup + Cosine schedule for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    wandb.init(project="PuGA", name=f"resnet18_expert_{model_id}", config={
        "model_id": model_id,
        "arch": "ResNet-18",
        "dataset": "CIFAR-100",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
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
        pbar = tqdm(trainloader, desc=f"[ResNet18 Expert {model_id}] Epoch {epoch+1}/{EPOCHS}")
        
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
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"best_resnet18_expert_{model_id}.pth"))
        else:
            epochs_no_improve += 1

        # Early stopping
        if epoch + 1 >= min_epochs and epochs_no_improve >= patience:
            print(f"⛔ Early stopping at epoch {epoch+1}: no improvement for {patience} epochs.")
            break

    # --- Final Summary ---
    print(f"🏁 ResNet18 Expert {model_id} finished.")
    print(f"📈 Best Test Accuracy: {best_acc:.2f}% at Epoch {best_epoch}")
    print(f"🧠 Best Train Accuracy: {best_train_acc:.2f}%")

    wandb.log({
        "best_test_acc": best_acc,
        "best_train_acc": best_train_acc,
        "best_epoch": best_epoch
    })

    wandb.finish()

    # CSV report
    report_path = os.path.join(CHECKPOINT_DIR, "report_resnet18.csv")
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
# Run Expert 2 Training (3rd expert, 0-indexed)
# ====================================================
if __name__ == "__main__":
    full_trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)

    # Load pre-computed expert indices
    expert_indices_list = load_expert_indices()

    # Train only expert 2 (3rd expert, 0-indexed) - ResNet18
    expert_id = 2
    expert_full_indices = expert_indices_list[expert_id]
    subset_full = Subset(full_trainset, expert_full_indices)
    
    print(f"Training ResNet-18 Expert {expert_id} with {len(subset_full)} samples")
    train_model(subset_full, expert_id)
