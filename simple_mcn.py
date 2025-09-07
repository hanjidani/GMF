import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
import os
import numpy as np
import csv
from datetime import datetime

# ====================================================
# WideResNet-28-10 Implementation
# ====================================================
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equal_in_out = (in_planes == out_planes)
        self.conv_shortcut = None
        if not self.equal_in_out:
            self.conv_shortcut = nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)

    def forward(self, x):
        shortcut = x if self.equal_in_out else self.conv_shortcut(x)
        out = self.relu1(self.bn1(x))
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        if self.drop_rate > 0:
            out = nn.functional.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return out + shortcut

class NetworkBlock(nn.Module):
    def __init__(self, num_layers, in_planes, out_planes, block, stride, drop_rate):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(block(in_planes if i == 0 else out_planes,
                                out_planes,
                                stride if i == 0 else 1,
                                drop_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor=10, num_classes=100, drop_rate=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        k = widen_factor
        channels = [16, 16*k, 32*k, 64*k]

        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.block1 = NetworkBlock(n, channels[0], channels[1], BasicBlock, stride=1, drop_rate=drop_rate)
        self.block2 = NetworkBlock(n, channels[1], channels[2], BasicBlock, stride=2, drop_rate=drop_rate)
        self.block3 = NetworkBlock(n, channels[2], channels[3], BasicBlock, stride=2, drop_rate=drop_rate)
        self.bn1 = nn.BatchNorm2d(channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        features = out
        logits = self.fc(features)
        return features, logits

# ====================================================
# MCN Baseline Architecture
# ====================================================
class SimpleMultiplicativeFusion(nn.Module):
    def __init__(self, num_experts, feature_dim):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in range(num_experts)])

    def forward(self, features_list):
        normalized = [norm(f) for norm, f in zip(self.norms, features_list)]
        fused = torch.stack(normalized, dim=0).prod(dim=0)
        return fused

class MCN_Baseline(nn.Module):
    def __init__(self, expert_backbones, feature_dim, num_classes):
        super().__init__()
        self.expert_backbones = nn.ModuleList(expert_backbones)
        self.fusion_module = SimpleMultiplicativeFusion(len(expert_backbones), feature_dim)
        self.global_head = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        features_list, individual_logits = [], []
        for expert in self.expert_backbones:
            features, logits = expert(x)
            features_list.append(features)
            individual_logits.append(logits)
        fused_features = self.fusion_module(features_list)
        global_logits = self.global_head(fused_features)
        return global_logits, individual_logits

# ====================================================
# Config
# ====================================================
NUM_EXPERTS = 4
NUM_CLASSES = 100
BATCH_SIZE = 128
EPOCHS = 100
BASE_LR = 1e-5
HEAD_LR = 1e-3
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
LAMBDA = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_DIR = "./checkpoints_iid"
TRAIN_FRACTION = 0.30
FUSION_SPLIT_PATH = os.environ.get("FUSION_SPLIT_PATH", "./splits/fusion_holdout_indices.npy")
SEED = 42

# ====================================================
# Data Loaders
# ====================================================
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761))
])
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
# Use fixed fusion subset if available
if os.path.exists(FUSION_SPLIT_PATH):
    fusion_indices = np.load(FUSION_SPLIT_PATH)
    train_subset = torch.utils.data.Subset(trainset, fusion_indices.tolist())
else:
    # Fallback: deterministic fraction split (legacy)
    train_len = int(len(trainset) * TRAIN_FRACTION)
    remain_len = len(trainset) - train_len
    train_subset, _ = random_split(
        trainset,
        [train_len, remain_len],
        generator=torch.Generator().manual_seed(SEED)
    )
trainloader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ====================================================
# Training Function
# ====================================================
def train_mcn():
    print("Initializing MCN training (Dual-Path)...")
    print(f"Experts: {NUM_EXPERTS} | Lambda: {LAMBDA}")

    temp_model = WideResNet(depth=28, widen_factor=10, num_classes=NUM_CLASSES)
    feature_dim = temp_model.fc.in_features
    del temp_model

    expert_backbones = []
    for i in range(NUM_EXPERTS):
        backbone = WideResNet(depth=28, widen_factor=10, num_classes=NUM_CLASSES)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_wrn_expert_{i}.pth")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found for expert {i} at {checkpoint_path}.")
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        backbone.load_state_dict(state_dict, strict=True)
        expert_backbones.append(backbone)

    print(f"✅ Loaded {len(expert_backbones)} experts.")

    model = MCN_Baseline(expert_backbones, feature_dim, NUM_CLASSES).to(DEVICE)
    # Parameter groups: base backbone layers at BASE_LR, heads/fusion (including experts' fc) at HEAD_LR
    backbone_base_params = []
    backbone_head_params = []
    for expert in model.expert_backbones:
        for name, param in expert.named_parameters():
            if not param.requires_grad:
                continue
            if 'fc' in name:
                backbone_head_params.append(param)
            else:
                backbone_base_params.append(param)
    head_params = list(model.fusion_module.parameters()) + list(model.global_head.parameters())
    param_groups = [
        {"params": backbone_base_params, "lr": BASE_LR},
        {"params": backbone_head_params + head_params, "lr": HEAD_LR},
    ]
    optimizer = optim.SGD(param_groups, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    wandb.init(project="MCN_Project", name="MCN_Dual")

    # Pre-training evaluation
    def evaluate(model_to_eval):
        model_to_eval.eval()
        global_correct, total_eval = 0, 0
        individual_correct = [0] * NUM_EXPERTS
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                global_logits, individual_logits = model_to_eval(inputs)
                _, predicted = global_logits.max(1)
                global_correct += predicted.eq(targets).sum().item()
                for j in range(NUM_EXPERTS):
                    _, predicted_ind = individual_logits[j].max(1)
                    individual_correct[j] += predicted_ind.eq(targets).sum().item()
                total_eval += targets.size(0)
        global_acc_eval = 100. * global_correct / max(1, total_eval)
        individual_accs_eval = [100. * c / max(1, total_eval) for c in individual_correct]
        return global_acc_eval, individual_accs_eval

    csv_path = os.path.join(CHECKPOINT_DIR, "report_simple_mcn_10.0.csv")
    fieldnames = [
        "timestamp", "phase", "epoch", "train_loss",
        "global_accuracy",
        *[f"expert_{i}_accuracy" for i in range(NUM_EXPERTS)],
    ]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    pre_global_acc, pre_individual_accs = evaluate(model)
    print(f"[Pre-Train] Global Acc: {pre_global_acc:.2f}%")
    for i, acc in enumerate(pre_individual_accs):
        print(f"[Pre-Train] Expert {i} Acc: {acc:.2f}%")
    wandb.log({
        "epoch": 0,
        "phase": "pretrain",
        "global_accuracy": pre_global_acc,
        **{f"expert_{i}_accuracy": acc for i, acc in enumerate(pre_individual_accs)}
    })
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "phase": "pretrain",
            "epoch": 0,
            "train_loss": "",
            "global_accuracy": pre_global_acc,
            **{f"expert_{i}_accuracy": pre_individual_accs[i] for i in range(NUM_EXPERTS)},
        })

    for epoch in range(EPOCHS):
        model.train()
        total_loss, total = 0, 0
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            global_logits, individual_logits = model(inputs)
            loss_global = criterion(global_logits, targets)
            loss_individual = sum(criterion(logits, targets) for logits in individual_logits)
            loss_total = loss_global + LAMBDA * loss_individual
            loss_total.backward()
            optimizer.step()
            total_loss += loss_total.item() * targets.size(0)
            total += targets.size(0)
            pbar.set_postfix(loss=loss_total.item())
        scheduler.step()

        model.eval()
        global_correct, total_eval = 0, 0
        individual_correct = [0] * NUM_EXPERTS
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                global_logits, individual_logits = model(inputs)
                _, predicted = global_logits.max(1)
                global_correct += predicted.eq(targets).sum().item()
                for i in range(NUM_EXPERTS):
                    _, predicted_ind = individual_logits[i].max(1)
                    individual_correct[i] += predicted_ind.eq(targets).sum().item()
                total_eval += targets.size(0)
        global_acc = 100. * global_correct / total_eval
        individual_accs = [100. * c / total_eval for c in individual_correct]
        print(f"Epoch {epoch+1} | Global Acc: {global_acc:.2f}%")
        for i, acc in enumerate(individual_accs):
            print(f"  -> Expert {i} Acc: {acc:.2f}%")
        wandb.log({
            "epoch": epoch + 1,
            "total_loss": total_loss / total,
            "global_accuracy": global_acc,
            **{f"expert_{i}_accuracy": acc for i, acc in enumerate(individual_accs)}
        })
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "phase": "train",
                "epoch": epoch + 1,
                "train_loss": total_loss / total if total > 0 else "",
                "global_accuracy": global_acc,
                **{f"expert_{i}_accuracy": individual_accs[i] for i in range(NUM_EXPERTS)},
            })
    wandb.finish()
    print("🏁 Training finished.")

if __name__ == "__main__":
    train_mcn()
