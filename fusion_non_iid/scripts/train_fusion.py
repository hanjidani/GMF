import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import sys

# Add project root and expert model paths to system path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../expert_training/models')))

from models.fusion_models import create_mcn_model
from utils.data_loader import get_data_loaders
from utils.helpers import set_seed
from densenet_cifar import densenet121


def load_densenet_specialist_backbones(checkpoint_dir, num_experts, device):
    """Loads Non-IID DenseNet specialists and returns ONLY their backbones."""
    expert_backbones = []
    for i in range(num_experts):
        checkpoint_path = os.path.join(checkpoint_dir, f'best_non_iid_densenet121_expert_{i}.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Specialist checkpoint not found: {checkpoint_path}")

        # Instantiate a temp model with 25 classes to match the checkpoint's architecture
        temp_model = densenet121(num_classes=25)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        temp_model.load_state_dict(state_dict)

        # Extract the feature backbone and discard the rest of the model
        backbone = temp_model.features
        expert_backbones.append(backbone.to(device))
        print(f"Loaded specialist backbone {i} from {checkpoint_path}")
    return expert_backbones


def dual_path_loss(global_logits, individual_logits, targets, criterion, alpha):
    """Calculates L_total = L_global + alpha * sum(L_individual)."""
    global_loss = criterion(global_logits, targets)
    individual_loss_sum = sum(criterion(logits, targets) for logits in individual_logits)
    total_loss = global_loss + alpha * individual_loss_sum
    return total_loss, global_loss, individual_loss_sum


def train_one_epoch(model, loader, optimizer_backbones, optimizer_heads, criterion, alpha, device):
    model.train()
    total_loss, global_correct, total_samples = 0, 0, 0

    for data, targets in tqdm(loader, desc="Training"):
        data, targets = data.to(device), targets.to(device)

        optimizer_backbones.zero_grad()
        optimizer_heads.zero_grad()

        global_logits, individual_logits = model(data)
        loss, _, _ = dual_path_loss(global_logits, individual_logits, targets, criterion, alpha)

        loss.backward()
        optimizer_backbones.step()
        optimizer_heads.step()

        total_loss += loss.item()
        _, predicted = global_logits.max(1)
        total_samples += targets.size(0)
        global_correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100. * global_correct / total_samples
    return avg_loss, accuracy


def validate(model, loader, criterion, alpha, device):
    model.eval()
    total_loss, global_correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for data, targets in tqdm(loader, desc="Validating"):
            data, targets = data.to(device), targets.to(device)

            global_logits, individual_logits = model(data)
            loss, _, _ = dual_path_loss(global_logits, individual_logits, targets, criterion, alpha)

            total_loss += loss.item()
            _, predicted = global_logits.max(1)
            total_samples += targets.size(0)
            global_correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100. * global_correct / total_samples
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Non-IID Specialist Fusion Training from Scratch")
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for CIFAR-100 dataset')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory containing Non-IID specialist checkpoints')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Directory to save trained models')
    parser.add_argument('--fusion_type', type=str, default='multiplicative',
                        choices=['multiplicative', 'multiplicativeAddition', 'TransformerBase', 'concatenation', 'simpleAddition'],
                        help='Type of fusion module to use')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--lr_backbone', type=float, default=1e-5, help='Learning rate for pre-trained backbones')
    parser.add_argument('--lr_heads', type=float, default=1e-4, help='Learning rate for new heads and fusion layers')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for the individual loss component')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    print("--- Setup ---")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Using device: {device}")

    print("\n--- Loading Data (fusion split) ---")
    splits_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../splits'))
    train_loader, val_loader = get_data_loaders(args.data_dir, args.batch_size, splits_dir=splits_dir)

    print("\n--- Building Model ---")
    expert_backbones = load_densenet_specialist_backbones(args.checkpoint_dir, 4, device)
    model = create_mcn_model(
        expert_backbones=expert_backbones,
        input_dim=1024,  # DenseNet-121 backbone output feature dimension
        num_classes=100,
        fusion_type=args.fusion_type
    ).to(device)

    print("\n--- Setting up Optimizers and Loss ---")
    optimizer_backbones = optim.AdamW(model.expert_backbones.parameters(), lr=args.lr_backbone, weight_decay=1e-4)
    optimizer_heads = optim.AdamW(
        list(model.individual_heads.parameters()) + list(model.fusion_module.parameters()) + list(model.global_head.parameters()),
        lr=args.lr_heads, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()

    print(f"Backbone LR: {args.lr_backbone}, New Component LR: {args.lr_heads}, Alpha: {args.alpha}")

    print("\n--- Starting Training ---")
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer_backbones, optimizer_heads, criterion, args.alpha, device)
        val_loss, val_acc = validate(model, val_loader, criterion, args.alpha, device)

        print(f"Epoch {epoch+1} Summary | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.output_dir, f'best_model_{args.fusion_type}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"🎉 New best model saved to {save_path} with accuracy: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()


