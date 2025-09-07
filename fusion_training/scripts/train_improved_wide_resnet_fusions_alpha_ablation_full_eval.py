#!/usr/bin/env python3
"""
Improved WideResNet Fusion Training Script (Alpha Ablation - Full Evaluation)

This script mirrors train_densenet_fusions_alpha_ablation_full_eval.py but is
adapted for Improved WideResNet-28-10 experts. It keeps fixed learning rates
(no LR schedulers) for clean alpha ablations and uses WRN-specific CSV/dir
naming to avoid conflicts with DenseNet runs.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import csv
import time
from datetime import datetime

# Local paths for imports
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
FUSION_ROOT = os.path.dirname(os.path.dirname(SCRIPTS_DIR))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPTS_DIR)))

# Add the scripts directory to import baseline helpers (generic utilities)
sys.path.append(SCRIPTS_DIR)

# Add fusion models path to create generic MCN model
sys.path.append(os.path.join(FUSION_ROOT))

# Add expert models path to import Improved WideResNet
sys.path.append(os.path.join(REPO_ROOT, 'expert_training', 'models'))

# Generic helpers from the baseline trainer (reused; densenet-specific naming helpers are replaced locally)
from train_densenet_fusions import (
    set_seed,
    mixup_data,
    mixup_criterion,
    cutmix_data,
    dual_path_loss,
    evaluate_experts_during_training,
    load_data_splits_with_optional_val,
    # Evaluation utilities (generic)
    evaluate_gaussian_noise_robustness,
    evaluate_corruption_robustness,
    evaluate_ood_detection,
    save_pre_training_results,
    save_post_training_results,
    save_pre_training_ood_results,
    save_post_training_ood_results,
    save_robustness_evaluation_results,
    setup_ood_evaluation_csv,
    save_ood_evaluation_results,
)

from models.fusion_models import create_mcn_model
from improved_wide_resnet import improved_wideresnet28_10


# =============================
# WRN-specific helper functions
# =============================

def load_wideresnet_experts(checkpoint_dir: str, num_experts: int, device: torch.device):
    """Load pre-trained Improved WideResNet-28-10 expert backbones.

    Expects checkpoints named: best_iid_wideresnet28_10_expert_{i}.pth
    """
    expert_backbones = []

    for i in range(num_experts):
        checkpoint_path = os.path.join(checkpoint_dir, f'best_iid_wideresnet28_10_expert_{i}.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"WideResNet expert checkpoint not found: {checkpoint_path}")

        expert = improved_wideresnet28_10(num_classes=100)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            expert.load_state_dict(checkpoint['model_state_dict'])
        else:
            expert.load_state_dict(checkpoint)

        expert = expert.to(device)
        expert_backbones.append(expert)

    return expert_backbones


def load_wrn_baseline_model(checkpoint_path: str, device: torch.device):
    """Load baseline Improved WideResNet-28-10 model for comparison."""
    model = improved_wideresnet28_10(num_classes=100)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    return model


def create_improved_wide_resnet_mcn_model(expert_backbones, fusion_type: str, alpha: float, device: torch.device):
    """Create Improved WideResNet MCN fusion model with proper dimensions.

    WRN-28-10 feature dimension: 640 (64 × widen_factor).
    Hidden dimension is set equal to input_dim for ablation (no expansion).
    """
    input_dim = 640
    num_classes = 100
    hidden_dim = input_dim

    print("Creating Improved WideResNet MCN model:")
    print(f"  - Fusion type: {fusion_type}")
    print(f"  - Input dimension: {input_dim}")
    print(f"  - Hidden dimension: {hidden_dim} (matches feature dimension)")
    print(f"  - Alpha: {alpha}")

    model = create_mcn_model(
        expert_backbones=expert_backbones,
        input_dim=input_dim,
        num_classes=num_classes,
        fusion_type=fusion_type,
        hidden_dim=hidden_dim,
    )

    return model.to(device)


# =============================
# WRN-specific CSV/logging helpers
# =============================

def setup_csv_logging_wrn(fusion_type: str, alpha: float, output_dir: str) -> str:
    csv_dir = Path(output_dir) / 'csv_logs' / 'wideresnet_fusions'
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f'wideresnet_{fusion_type}_alpha_{alpha}_training_log.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            'epoch', 'train_loss', 'val_loss', 'val_accuracy',
            'experts_lr', 'fusion_lr', 'global_head_lr',
            'loss_global', 'loss_individual', 'loss_total',
            'expert_0_accuracy', 'expert_1_accuracy', 'expert_2_accuracy', 'expert_3_accuracy',
            'expert_0_loss', 'expert_1_loss', 'expert_2_loss', 'expert_3_loss',
            'alpha', 'fusion_type', 'model_architecture', 'timestamp'
        ])
        writer.writeheader()
    return str(csv_path)


def log_training_epoch_wrn(
    csv_path: str,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_accuracy: float,
    experts_lr: float,
    fusion_lr: float,
    global_head_lr: float,
    loss_global: float,
    loss_individual: float,
    loss_total: float,
    alpha: float,
    fusion_type: str,
    expert_accuracies=None,
    expert_losses=None,
):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    if expert_accuracies is None:
        expert_accuracies = [0.0] * 4
    if expert_losses is None:
        expert_losses = [0.0] * 4

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            'epoch', 'train_loss', 'val_loss', 'val_accuracy',
            'experts_lr', 'fusion_lr', 'global_head_lr',
            'loss_global', 'loss_individual', 'loss_total',
            'expert_0_accuracy', 'expert_1_accuracy', 'expert_2_accuracy', 'expert_3_accuracy',
            'expert_0_loss', 'expert_1_loss', 'expert_2_loss', 'expert_3_loss',
            'alpha', 'fusion_type', 'model_architecture', 'timestamp'
        ])

        writer.writerow({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'experts_lr': experts_lr,
            'fusion_lr': fusion_lr,
            'global_head_lr': global_head_lr,
            'loss_global': loss_global,
            'loss_individual': loss_individual,
            'loss_total': loss_total,
            'expert_0_accuracy': expert_accuracies[0],
            'expert_1_accuracy': expert_accuracies[1],
            'expert_2_accuracy': expert_accuracies[2],
            'expert_3_accuracy': expert_accuracies[3],
            'expert_0_loss': expert_losses[0],
            'expert_1_loss': expert_losses[1],
            'expert_2_loss': expert_losses[2],
            'expert_3_loss': expert_losses[3],
            'alpha': alpha,
            'fusion_type': fusion_type,
            'model_architecture': 'wideresnet28_10',
            'timestamp': timestamp,
        })


def setup_pre_training_csv_wrn(fusion_type: str, alpha: float, output_dir: str) -> str:
    csv_dir = Path(output_dir) / 'csv_logs' / 'wideresnet_fusions'
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f'wideresnet_{fusion_type}_alpha_{alpha}_pre_training_evaluation.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'timestamp', 'model_type', 'fusion_type', 'alpha', 'expert_id', 'model_architecture',
            'test_type', 'test_param', 'corruption_type', 'severity', 'accuracy', 'loss',
            'correct_predictions', 'total_predictions',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    return str(csv_path)


def setup_post_training_csv_wrn(fusion_type: str, alpha: float, output_dir: str) -> str:
    csv_dir = Path(output_dir) / 'csv_logs' / 'wideresnet_fusions'
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f'wideresnet_{fusion_type}_alpha_{alpha}_post_training_evaluation.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'timestamp', 'model_type', 'fusion_type', 'alpha', 'expert_id', 'model_architecture',
            'test_type', 'test_param', 'corruption_type', 'severity', 'accuracy', 'loss',
            'correct_predictions', 'total_predictions',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    return str(csv_path)


def setup_robustness_evaluation_csv_wrn(fusion_type: str, alpha: float, output_dir: str) -> str:
    csv_dir = Path(output_dir) / 'csv_logs' / 'wideresnet_fusions'
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f'wideresnet_{fusion_type}_alpha_{alpha}_robustness_evaluation.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'model_type', 'fusion_type', 'alpha', 'expert_id', 'model_architecture',
            'test_type', 'attack_type', 'corruption_type', 'severity_level', 'epsilon',
            'accuracy', 'loss', 'correct_predictions', 'total_predictions', 'timestamp',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    return str(csv_path)


def save_fusion_model_components_wrn(model, fusion_type: str, alpha: float, save_dir: str):
    """Save trained fusion model components under WRN naming."""
    experts_dir = os.path.join(save_dir, 'experts')
    fusion_dir = os.path.join(save_dir, 'fusion_module')
    global_dir = os.path.join(save_dir, 'global_head')
    complete_dir = os.path.join(save_dir, 'complete_model')

    os.makedirs(experts_dir, exist_ok=True)
    os.makedirs(fusion_dir, exist_ok=True)
    os.makedirs(global_dir, exist_ok=True)
    os.makedirs(complete_dir, exist_ok=True)

    for i, expert in enumerate(model.expert_backbones):
        expert_path = os.path.join(experts_dir, f'expert_{i}_alpha_{alpha}.pth')
        torch.save({
            'model_state_dict': expert.state_dict(),
            'alpha': alpha,
            'fusion_type': fusion_type,
            'expert_id': i,
            'model_architecture': 'wideresnet28_10',
        }, expert_path)
        print(f"  Saved expert {i}: {expert_path}")

    fusion_path = os.path.join(fusion_dir, f'fusion_module_alpha_{alpha}.pth')
    torch.save({
        'model_state_dict': model.fusion_module.state_dict(),
        'alpha': alpha,
        'fusion_type': fusion_type,
        'model_architecture': 'wideresnet28_10',
    }, fusion_path)
    print(f"  Saved fusion module: {fusion_path}")

    global_path = os.path.join(global_dir, f'global_head_alpha_{alpha}.pth')
    torch.save({
        'model_state_dict': model.global_head.state_dict(),
        'alpha': alpha,
        'fusion_type': fusion_type,
        'model_architecture': 'wideresnet28_10',
    }, global_path)
    print(f"  Saved global head: {global_path}")

    complete_path = os.path.join(complete_dir, f'complete_fusion_model_alpha_{alpha}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'alpha': alpha,
        'fusion_type': fusion_type,
        'model_type': 'wideresnet_fusion',
        'model_architecture': 'wideresnet28_10',
    }, complete_path)
    print(f"  Saved complete model: {complete_path}")
    print(f"✅ All model components saved to: {save_dir}")


# ======================================
# Training loop (identical to DenseNet)
# ======================================

def train_fusion_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    alpha: float = 1.0,
    epochs: int = 100,
    fusion_type: str = "multiplicative",
    save_dir: str | None = None,
    save_freq: int = 10,
    csv_path: str | None = None,
    augmentation_mode: str = "cutmix",
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
    label_smoothing: float = 0.1,
    gradient_clip_norm: float = 1.0,
    base_lr: float = 1e-4,
    head_lr: float = 1e-3,
    experts_lr_scale: float = 0.1,
):
    """Train the fusion model (no LR scheduling; fixed LR for all components)."""
    optim_experts = optim.AdamW(
        model.expert_backbones.parameters(), lr=base_lr * experts_lr_scale, weight_decay=1e-4
    )
    optim_fusion = optim.AdamW(
        model.fusion_module.parameters(), lr=head_lr, weight_decay=1e-4
    )
    optim_global = optim.AdamW(
        model.global_head.parameters(), lr=head_lr, weight_decay=1e-4
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    best_val_acc = 0.0
    best_model_state = None
    best_fusion_val_acc = -float("inf")
    best_global_val_acc = -float("inf")
    best_global_head_path: str | None = None

    num_experts = len(model.expert_backbones)
    best_expert_val_acc = [-float("inf")] * num_experts
    best_expert_paths: list[str | None] = [None] * num_experts
    best_experts_dir = None
    best_fusion_dir = None
    best_global_dir = None
    if save_dir is not None:
        best_experts_dir = os.path.join(save_dir, "experts_best")
        best_fusion_dir = os.path.join(save_dir, "fusion_best")
        best_global_dir = os.path.join(save_dir, "global_best")
        os.makedirs(best_experts_dir, exist_ok=True)
        os.makedirs(best_fusion_dir, exist_ok=True)
        os.makedirs(best_global_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            optim_experts.zero_grad(set_to_none=True)
            optim_fusion.zero_grad(set_to_none=True)
            optim_global.zero_grad(set_to_none=True)

            if augmentation_mode == "mixup":
                data, targets_a, targets_b, lam = mixup_data(data, targets, mixup_alpha, device)
                global_logits, individual_logits = model(data)
                global_loss = mixup_criterion(criterion, global_logits, targets_a, targets_b, lam)
                individual_loss = torch.tensor(0.0, device=device)
                for logits in individual_logits:
                    individual_loss = individual_loss + mixup_criterion(
                        criterion, logits, targets_a, targets_b, lam
                    )
                total_loss = global_loss + alpha * individual_loss
            elif augmentation_mode == "cutmix":
                data, targets_a, targets_b, lam = cutmix_data(data, targets, cutmix_alpha, device)
                global_logits, individual_logits = model(data)
                global_loss = mixup_criterion(criterion, global_logits, targets_a, targets_b, lam)
                individual_loss = torch.tensor(0.0, device=device)
                for logits in individual_logits:
                    individual_loss = individual_loss + mixup_criterion(
                        criterion, logits, targets_a, targets_b, lam
                    )
                total_loss = global_loss + alpha * individual_loss
            else:
                global_logits, individual_logits = model(data)
                total_loss, global_loss, individual_loss = dual_path_loss(
                    global_logits, individual_logits, targets, alpha
                )

            total_loss.backward()

            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

            optim_experts.step()
            optim_fusion.step()
            optim_global.step()

            train_loss += total_loss.item()
            _, predicted = global_logits.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            if batch_idx % 50 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {total_loss.item():.4f} (Global: {global_loss.item():.4f}, "
                    f"Individual: {individual_loss.item():.4f})"
                )

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                global_logits, individual_logits = model(data)
                total_loss, _, _ = dual_path_loss(global_logits, individual_logits, targets, alpha)
                val_loss += total_loss.item()
                _, predicted = global_logits.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total

        # Evaluate individual experts on validation set
        expert_accuracies, expert_losses = evaluate_experts_during_training(
            model.expert_backbones, val_loader, device
        )

        # Save per-expert best checkpoints when they improve
        if best_experts_dir is not None:
            for i, acc in enumerate(expert_accuracies):
                if acc > best_expert_val_acc[i]:
                    best_expert_val_acc[i] = acc
                    expert_path = os.path.join(best_experts_dir, f"expert_{i}_best.pth")
                    torch.save(
                        {
                            'model_state_dict': model.expert_backbones[i].state_dict(),
                            'epoch': epoch + 1,
                            'best_val_acc': acc,
                            'fusion_type': fusion_type,
                            'alpha': alpha,
                            'component': f'expert_{i}',
                        },
                        expert_path,
                    )
                    best_expert_paths[i] = expert_path
                    print(f"  🔸 Saved new best Expert {i} (val acc {acc:.2f}%) to {expert_path}")

        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss/len(val_loader):.4f}, Val   Acc: {val_acc:.2f}%")
        print(
            "  Expert Accuracies: "
            + ", ".join([f"Expert {i}: {acc:.2f}%" for i, acc in enumerate(expert_accuracies)])
        )

        # Save best fusion module and global head when fused validation accuracy improves
        if best_fusion_dir is not None and val_acc > best_fusion_val_acc:
            best_fusion_val_acc = val_acc
            fusion_best_path = os.path.join(best_fusion_dir, "fusion_module_best.pth")
            torch.save(
                {
                    'model_state_dict': model.fusion_module.state_dict(),
                    'epoch': epoch + 1,
                    'best_val_acc': val_acc,
                    'fusion_type': fusion_type,
                    'alpha': alpha,
                    'component': 'fusion_module_best',
                },
                fusion_best_path,
            )
            print(f"  🔸 Saved new best Fusion Module (val acc {val_acc:.2f}%) to {fusion_best_path}")

        if best_global_dir is not None and val_acc > best_global_val_acc:
            best_global_val_acc = val_acc
            global_best_path = os.path.join(best_global_dir, "global_head_best.pth")
            torch.save(
                {
                    'model_state_dict': model.global_head.state_dict(),
                    'epoch': epoch + 1,
                    'best_val_acc': val_acc,
                    'fusion_type': fusion_type,
                    'alpha': alpha,
                    'component': 'global_head_best',
                },
                global_best_path,
            )
            best_global_head_path = global_best_path
            print(f"  🔸 Saved new best Global Head (val acc {val_acc:.2f}%) to {global_best_path}")

        if csv_path:
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            log_training_epoch_wrn(
                csv_path=csv_path,
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                val_accuracy=val_acc,
                experts_lr=base_lr * experts_lr_scale,
                fusion_lr=head_lr,
                global_head_lr=head_lr,
                loss_global=global_loss.item() if "global_loss" in locals() else 0.0,
                loss_individual=individual_loss.item() if "individual_loss" in locals() else 0.0,
                loss_total=total_loss.item() if "total_loss" in locals() else 0.0,
                alpha=alpha,
                fusion_type=fusion_type,
                expert_accuracies=expert_accuracies,
                expert_losses=expert_losses,
            )
            print("  📊 Training progress logged to CSV")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"  New best validation accuracy: {best_val_acc:.2f}%")

        if save_dir and (epoch + 1) % save_freq == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(
                {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_experts_state_dict': optim_experts.state_dict(),
                    'optimizer_fusion_state_dict': optim_fusion.state_dict(),
                    'optimizer_global_state_dict': optim_global.state_dict(),
                    'best_val_acc': best_val_acc,
                    'alpha': alpha,
                    'fusion_type': fusion_type,
                },
                checkpoint_path,
            )
            print(f"  Saved checkpoint: {checkpoint_path}")

    if save_dir and best_model_state is not None:
        best_model_path = os.path.join(save_dir, "best_model.pth")
        torch.save(
            {
                'model_state_dict': best_model_state,
                'best_val_acc': best_val_acc,
                'alpha': alpha,
                'fusion_type': fusion_type,
                'final_epoch': epochs,
            },
            best_model_path,
        )
        print(f"  Saved best model: {best_model_path}")

    return best_val_acc, best_expert_paths, best_global_head_path


# ======================================
# Robustness evaluation (WRN variant)
# ======================================

def run_robustness_evaluation_wrn(
    expert_backbones,
    fusion_model,
    fusion_type: str,
    alpha: float,
    output_dir: str,
    data_dir: str,
    batch_size: int,
    device: torch.device,
):
    print(f"\n{'='*80}")
    print(f"Starting Phase 1, 2 & 3 Robustness Evaluation for WideResNet {fusion_type} (α={alpha})")
    print(f"{'='*80}")

    robustness_csv_path = setup_robustness_evaluation_csv_wrn(fusion_type, alpha, output_dir)
    ood_csv_path = setup_ood_evaluation_csv(fusion_type, alpha, output_dir)

    # Load baseline model (WRN) for comparison if available
    baseline_checkpoint_path = '../../expert_training/scripts/checkpoints_expert_full_dataset_benchmark_250/best_full_dataset_wideresnet28_10_benchmark_250.pth'
    if os.path.exists(baseline_checkpoint_path):
        print(f"\nLoading baseline WideResNet model from: {baseline_checkpoint_path}")
        baseline_model = load_wrn_baseline_model(baseline_checkpoint_path, device)
    else:
        print(f"Warning: Baseline checkpoint not found at {baseline_checkpoint_path}")
        baseline_model = None

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    testset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_transform
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"\n{'='*60}")
    print(f"GAUSSIAN NOISE ROBUSTNESS")
    print(f"{'='*60}")
    noise_sigmas = [0.0, 0.05, 0.1, 0.2, 0.3]
    print(f"Evaluating sigmas: {noise_sigmas}")

    # Experts under noise
    for i, expert in enumerate(expert_backbones):
        print(f"\nTesting Expert {i} under Gaussian noise...")
        noise_results = evaluate_gaussian_noise_robustness(expert, testloader, device, sigmas=noise_sigmas)
        for sigma, res in noise_results.items():
            save_robustness_evaluation_results(
                robustness_csv_path, 'expert', fusion_type, alpha, i, 'wideresnet28_10',
                'gaussian_noise', f'sigma_{sigma}', 'N/A', 'N/A', 'N/A',
                res['accuracy'], res['loss'], res['correct'], res['total']
            )
        print("  Expert noise results: " + ", ".join([f"σ={s}: {r['accuracy']:.2f}%" for s, r in noise_results.items()]))

    # Baseline under noise
    if baseline_model is not None:
        print(f"\nTesting Baseline WideResNet under Gaussian noise...")
        noise_results = evaluate_gaussian_noise_robustness(baseline_model, testloader, device, sigmas=noise_sigmas)
        for sigma, res in noise_results.items():
            save_robustness_evaluation_results(
                robustness_csv_path, 'baseline', fusion_type, alpha, 'N/A', 'wideresnet28_10',
                'gaussian_noise', f'sigma_{sigma}', 'N/A', 'N/A', 'N/A',
                res['accuracy'], res['loss'], res['correct'], res['total']
            )
        print("  Baseline noise results: " + ", ".join([f"σ={s}: {r['accuracy']:.2f}%" for s, r in noise_results.items()]))

    # Fusion model under noise
    if fusion_model is not None:
        print(f"\nTesting Global Fusion Model under Gaussian noise...")
        noise_results = evaluate_gaussian_noise_robustness(fusion_model, testloader, device, sigmas=noise_sigmas)
        for sigma, res in noise_results.items():
            save_robustness_evaluation_results(
                robustness_csv_path, 'fusion', fusion_type, alpha, 'N/A', 'fusion_model',
                'gaussian_noise', f'sigma_{sigma}', 'N/A', 'N/A', 'N/A',
                res['accuracy'], res['loss'], res['correct'], res['total']
            )
        print("  Fusion noise results: " + ", ".join([f"σ={s}: {r['accuracy']:.2f}%" for s, r in noise_results.items()]))

    print(f"\n{'='*60}")
    print(f"PHASE 1: CIFAR-100-C Corruption Robustness")
    print(f"{'='*60}")
    for i, expert in enumerate(expert_backbones):
        print(f"\nTesting Expert {i} on corruptions...")
        corruption_results = evaluate_corruption_robustness(expert, data_dir, batch_size, device)
        for corruption_type, severity_results in corruption_results.items():
            for severity, results in severity_results.items():
                save_robustness_evaluation_results(
                    robustness_csv_path, 'expert', fusion_type, alpha, i, 'wideresnet28_10',
                    'corruption', 'N/A', corruption_type, severity, 'N/A',
                    results['accuracy'], results['loss'], results['correct'], results['total']
                )

    if baseline_model is not None:
        print(f"\nTesting Baseline WideResNet on corruptions...")
        corruption_results = evaluate_corruption_robustness(baseline_model, data_dir, batch_size, device)
        for corruption_type, severity_results in corruption_results.items():
            for severity, results in severity_results.items():
                save_robustness_evaluation_results(
                    robustness_csv_path, 'baseline', fusion_type, alpha, 'N/A', 'wideresnet28_10',
                    'corruption', 'N/A', corruption_type, severity, 'N/A',
                    results['accuracy'], results['loss'], results['correct'], results['total']
                )

    # OOD detection (experts + fusion + baseline)
    print(f"\n{'='*60}")
    print(f"PHASE 2: Out-of-Distribution Detection")
    print(f"{'='*60}")
    for i, expert in enumerate(expert_backbones):
        print(f"\nTesting Expert {i} on OOD detection...")
        ood_results = evaluate_ood_detection(expert, testloader, data_dir, batch_size, device)
        save_ood_evaluation_results(ood_csv_path, 'expert', fusion_type, alpha, i, 'wideresnet28_10', ood_results)

    if fusion_model is not None:
        print(f"\nTesting Global Fusion Model on OOD detection...")
        ood_results = evaluate_ood_detection(fusion_model, testloader, data_dir, batch_size, device)
        save_ood_evaluation_results(ood_csv_path, 'fusion', fusion_type, alpha, 'N/A', 'fusion_model', ood_results)

    if baseline_model is not None:
        print(f"\nTesting Baseline WideResNet on OOD detection...")
        ood_results = evaluate_ood_detection(baseline_model, testloader, data_dir, batch_size, device)
        save_ood_evaluation_results(ood_csv_path, 'baseline', fusion_type, alpha, 'N/A', 'wideresnet28_10', ood_results)


# =============================
# Main
# =============================

def main():
    parser = argparse.ArgumentParser(
        description="Train Improved WideResNet Fusion Models (Alpha Ablation, Full Evaluation, No Schedulers)"
    )
    parser.add_argument(
        "--fusion_type",
        type=str,
        required=True,
        choices=[
            "multiplicative",
            "multiplicativeAddition",
            "TransformerBase",
            "concatenation",
            "simpleAddition",
        ],
        help="Type of fusion to use",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha for dual-path loss")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default='../../expert_training/scripts/checkpoints_expert_iid',
        help="Directory containing WRN expert checkpoints",
    )
    parser.add_argument(
        "--output_dir", type=str, default="../fusion_checkpoints", help="Output directory"
    )
    parser.add_argument("--data_dir", type=str, default="../data", help="Data directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument(
        "--augmentation_mode",
        type=str,
        default="cutmix",
        choices=["mixup", "cutmix", "none"],
    )
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--skip_pre_eval", action="store_true", help="Skip pre-training eval")
    parser.add_argument(
        "--use_train_val_split",
        action="store_true",
        help="Use validation split from fusion holdout",
    )
    parser.add_argument("--val_split_ratio", type=float, default=0.1)
    parser.add_argument(
        "--experts_lr_scale",
        type=float,
        default=0.1,
        help="Scale factor for experts LR (multiplied by base_lr)",
    )
    parser.add_argument(
        "--base_lr", type=float, default=1e-4, help="Base LR for experts before scaling"
    )
    parser.add_argument(
        "--head_lr", type=float, default=1e-3, help="LR for fusion and global head"
    )

    args = parser.parse_args()

    print(
        f"Starting WideResNet {args.fusion_type} fusion training (alpha-ablation, full evaluation, no schedulers)..."
    )
    print(f"Alpha: {args.alpha}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Output directory: {args.output_dir}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load experts
    print("Loading Improved WideResNet expert backbones...")
    expert_backbones = load_wideresnet_experts(args.checkpoint_dir, 4, device)
    print(f"Successfully loaded {len(expert_backbones)} WRN experts")

    # Baseline model for evaluations (if available)
    baseline_checkpoint_path = '../../expert_training/scripts/checkpoints_expert_full_dataset_benchmark_250/best_full_dataset_wideresnet28_10_benchmark_250.pth'
    if os.path.exists(baseline_checkpoint_path):
        print(f"Loading baseline WideResNet model from: {baseline_checkpoint_path}")
        baseline_model = load_wrn_baseline_model(baseline_checkpoint_path, device)
    else:
        print(f"Warning: Baseline checkpoint not found at {baseline_checkpoint_path}")
        baseline_model = None

    # Create model
    print("Creating WideResNet MCN fusion model...")
    fusion_model = create_improved_wide_resnet_mcn_model(expert_backbones, args.fusion_type, args.alpha, device)
    print("✅ Successfully created WideResNet MCN fusion model")

    # A common test loader and noise sigmas used across pre/post-training evals
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    testset = torchvision.datasets.CIFAR100(
        root=args.data_dir, train=False, download=True, transform=test_transform
    )
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    noise_sigmas = [0.0, 0.05, 0.1, 0.2, 0.3]

    # PHASE 1: PRE-TRAINING EVALUATION
    if not args.skip_pre_eval:
        print(f"\n{'='*80}")
        print(f"PHASE 1: PRE-TRAINING EVALUATION (α={args.alpha})")
        print(f"{'='*80}")

        pre_train_csv_path = setup_pre_training_csv_wrn(args.fusion_type, args.alpha, args.output_dir)

        # Gaussian noise robustness (experts + baseline)
        print(f"\n{'='*60}")
        print("PRE-TRAINING: Gaussian Noise Robustness")
        print(f"{'='*60}")
        for i, expert in enumerate(expert_backbones):
            print(f"\nTesting Expert {i} (PRE-TRAINING) under Gaussian noise...")
            noise_results = evaluate_gaussian_noise_robustness(expert, testloader, device, sigmas=noise_sigmas)
            for sigma, res in noise_results.items():
                save_pre_training_results(
                    pre_train_csv_path,
                    'expert',
                    args.fusion_type,
                    args.alpha,
                    i,
                    'wideresnet28_10',
                    'gaussian_noise',
                    f'sigma_{sigma}',
                    'N/A',
                    'N/A',
                    res['accuracy'],
                    res['loss'],
                    res['correct'],
                    res['total'],
                )
            print(
                "  Expert noise results: "
                + ", ".join([f"σ={s}: {r['accuracy']:.2f}%" for s, r in noise_results.items()])
            )

        if baseline_model is not None:
            print(f"\nTesting Baseline WideResNet (PRE-TRAINING) under Gaussian noise...")
            noise_results = evaluate_gaussian_noise_robustness(baseline_model, testloader, device, sigmas=noise_sigmas)
            for sigma, res in noise_results.items():
                save_pre_training_results(
                    pre_train_csv_path,
                    'baseline',
                    args.fusion_type,
                    args.alpha,
                    'N/A',
                    'wideresnet28_10',
                    'gaussian_noise',
                    f'sigma_{sigma}',
                    'N/A',
                    'N/A',
                    res['accuracy'],
                    res['loss'],
                    res['correct'],
                    res['total'],
                )
            print(
                "  Baseline noise results: "
                + ", ".join([f"σ={s}: {r['accuracy']:.2f}%" for s, r in noise_results.items()])
            )

        # CIFAR-100-C corruptions (experts + baseline)
        print(f"\n{'='*60}")
        print("PRE-TRAINING: CIFAR-100-C Corruption Robustness")
        print(f"{'='*60}")
        for i, expert in enumerate(expert_backbones):
            print(f"\nTesting Expert {i} (PRE-TRAINING) on corruptions...")
            corruption_results = evaluate_corruption_robustness(expert, args.data_dir, args.batch_size, device)
            for corruption_type, severity_results in corruption_results.items():
                for severity, results in severity_results.items():
                    save_pre_training_results(
                        pre_train_csv_path,
                        'expert',
                        args.fusion_type,
                        args.alpha,
                        i,
                        'wideresnet28_10',
                        'corruption',
                        'N/A',
                        corruption_type,
                        severity,
                        results['accuracy'],
                        results['loss'],
                        results['correct'],
                        results['total'],
                    )

        if baseline_model is not None:
            print(f"\nTesting Baseline WideResNet (PRE-TRAINING) on corruptions...")
            corruption_results = evaluate_corruption_robustness(baseline_model, args.data_dir, args.batch_size, device)
            for corruption_type, severity_results in corruption_results.items():
                for severity, results in severity_results.items():
                    save_pre_training_results(
                        pre_train_csv_path,
                        'baseline',
                        args.fusion_type,
                        args.alpha,
                        'N/A',
                        'wideresnet28_10',
                        'corruption',
                        'N/A',
                        corruption_type,
                        severity,
                        results['accuracy'],
                        results['loss'],
                        results['correct'],
                        results['total'],
                    )

        # OOD detection (experts + baseline)
        print(f"\n{'='*60}")
        print("PRE-TRAINING: Out-of-Distribution Detection")
        print(f"{'='*60}")
        for i, expert in enumerate(expert_backbones):
            print(f"\nTesting Expert {i} (PRE-TRAINING) on OOD detection...")
            ood_results = evaluate_ood_detection(expert, testloader, args.data_dir, args.batch_size, device)
            save_pre_training_ood_results(pre_train_csv_path, 'expert', args.fusion_type, args.alpha, i, 'wideresnet28_10', ood_results)

        if baseline_model is not None:
            print(f"\nTesting Baseline WideResNet (PRE-TRAINING) on OOD detection...")
            ood_results = evaluate_ood_detection(baseline_model, testloader, args.data_dir, args.batch_size, device)
            save_pre_training_ood_results(pre_train_csv_path, 'baseline', args.fusion_type, args.alpha, 'N/A', 'wideresnet28_10', ood_results)

        print(f"\n✅ PRE-TRAINING evaluation completed!")
        print(f"   Results saved to: {pre_train_csv_path}")
    else:
        print("⏭️  Skipping Phase 1 pre-training evaluation by flag --skip_pre_eval")

    # PHASE 2: TRAINING
    print(f"\n{'='*80}")
    print(f"PHASE 2: FUSION TRAINING (α={args.alpha})")
    print(f"{'='*80}")

    print("Loading data splits for fusion training...")
    train_loader, val_loader = load_data_splits_with_optional_val(
        args.data_dir, args.batch_size, args.use_train_val_split, args.val_split_ratio, seed=args.seed
    )
    print("✅ Data splits loaded successfully")

    csv_path = setup_csv_logging_wrn(args.fusion_type, args.alpha, args.output_dir)
    print(f"CSV logging setup: {csv_path}")

    experiment_output_dir = os.path.join(
        args.output_dir, f'wideresnet_fusions_alpha_{args.alpha}', args.fusion_type
    )
    os.makedirs(experiment_output_dir, exist_ok=True)
    print(f"  - Experiment Output: {experiment_output_dir}")

    print(
        f"\n🚀 Starting WideResNet {args.fusion_type} fusion training with alpha={args.alpha} (no schedulers)..."
    )
    print(
        f"   Augmentation: {args.augmentation_mode.upper()}, MixUp α={args.mixup_alpha}, "
        f"CutMix α={args.cutmix_alpha}, Label Smoothing={args.label_smoothing}, Grad Clip={args.gradient_clip_norm}"
    )

    best_val_acc, best_expert_paths, best_global_head_path = train_fusion_model(
        fusion_model,
        train_loader,
        val_loader,
        device,
        alpha=args.alpha,
        epochs=args.epochs,
        fusion_type=args.fusion_type,
        save_dir=experiment_output_dir,
        save_freq=args.save_freq,
        csv_path=csv_path,
        augmentation_mode=args.augmentation_mode,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        label_smoothing=args.label_smoothing,
        gradient_clip_norm=args.gradient_clip_norm,
        base_lr=args.base_lr,
        head_lr=args.head_lr,
        experts_lr_scale=args.experts_lr_scale,
    )
    print(f"✅ Training completed! Best validation accuracy: {best_val_acc:.2f}%")

    # Save components
    print("\n💾 Saving trained model components...")
    save_fusion_model_components_wrn(
        fusion_model,
        args.fusion_type,
        args.alpha,
        experiment_output_dir,
    )
    print("✅ Model components saved successfully")

    # NEW: Clean Test-set Evaluation using best experts + best global head
    print(f"\n{'='*80}")
    print("CLEAN TEST-SET EVALUATION (Best Experts + Best Global Head)")
    print(f"{'='*80}")

    # Reload best experts
    try:
        if best_expert_paths is not None:
            for i, path in enumerate(best_expert_paths):
                if path is not None and os.path.exists(path):
                    ckpt = torch.load(path, map_location=device)
                    model_state = ckpt.get('model_state_dict', ckpt)
                    fusion_model.expert_backbones[i].load_state_dict(model_state)
                    fusion_model.expert_backbones[i] = fusion_model.expert_backbones[i].to(device)
                    print(f"🔄 Loaded best Expert {i} weights for clean eval from {path}")
                else:
                    print(f"⚠️  Best checkpoint not found for Expert {i}; using current expert weights")
        else:
            print("⚠️  No best expert paths returned; using current expert weights")
    except Exception as e:
        print(f"⚠️  Failed to load best expert checkpoints for clean eval; using current weights. Error: {e}")

    # Reload best global head
    if best_global_head_path is not None and os.path.exists(best_global_head_path):
        try:
            ckpt = torch.load(best_global_head_path, map_location=device)
            model_state = ckpt.get('model_state_dict', ckpt)
            fusion_model.global_head.load_state_dict(model_state)
            fusion_model.global_head = fusion_model.global_head.to(device)
            print(f"🔄 Loaded best Global Head for clean eval from {best_global_head_path}")
        except Exception as e:
            print(f"⚠️  Failed to load best global head for clean eval; using current head. Error: {e}")
    else:
        print("⚠️  No best global head path found; using current global head weights")

    # Evaluate on CIFAR-100 test set (clean)
    fusion_model.eval()
    criterion = nn.CrossEntropyLoss()
    test_total = 0
    test_correct = 0
    test_loss = 0.0
    with torch.no_grad():
        for data, targets in testloader:
            data, targets = data.to(device), targets.to(device)
            outputs, _ = fusion_model(data)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_total += targets.size(0)
            test_correct += (predicted == targets).sum().item()

    clean_test_acc = 100.0 * test_correct / max(1, test_total)
    clean_test_loss = test_loss / max(1, len(testloader))
    print(f"  Clean Test Accuracy (best experts + best head): {clean_test_acc:.2f}% | Loss: {clean_test_loss:.4f}")

    # PHASE 3: POST-TRAINING EVALUATION
    print(f"\n{'='*80}")
    print(f"PHASE 3: POST-TRAINING EVALUATION (α={args.alpha})")
    print(f"{'='*80}")

    # Load per-expert best checkpoints saved during training
    try:
        if best_expert_paths is not None:
            for i, path in enumerate(best_expert_paths):
                if path is not None and os.path.exists(path):
                    ckpt = torch.load(path, map_location=device)
                    model_state = ckpt.get('model_state_dict', ckpt)
                    expert_backbones[i].load_state_dict(model_state)
                    expert_backbones[i] = expert_backbones[i].to(device)
                    print(f"🔄 Loaded best Expert {i} from {path}")
                else:
                    print(f"⚠️  Best checkpoint not found for Expert {i}; using current weights")
        else:
            print("⚠️  No best expert paths returned; using current expert weights")
    except Exception as e:
        print(f"⚠️  Failed to load best expert checkpoints; using current weights. Error: {e}")

    post_train_csv_path = setup_post_training_csv_wrn(args.fusion_type, args.alpha, args.output_dir)

    # Gaussian noise robustness (experts + fusion)
    print(f"\n{'='*60}")
    print("POST-TRAINING: Gaussian Noise Robustness")
    print(f"{'='*60}")
    for i, expert in enumerate(expert_backbones):
        print(f"\nTesting Expert {i} (POST-TRAINING) under Gaussian noise...")
        noise_results = evaluate_gaussian_noise_robustness(expert, testloader, device, sigmas=noise_sigmas)
        for sigma, res in noise_results.items():
            save_post_training_results(
                post_train_csv_path,
                'expert',
                args.fusion_type,
                args.alpha,
                i,
                'wideresnet28_10',
                'gaussian_noise',
                f'sigma_{sigma}',
                'N/A',
                'N/A',
                res['accuracy'],
                res['loss'],
                res['correct'],
                res['total'],
            )
    print(f"\nTesting Trained Fusion Model under Gaussian noise...")
    noise_results = evaluate_gaussian_noise_robustness(fusion_model, testloader, device, sigmas=noise_sigmas)
    for sigma, res in noise_results.items():
        save_post_training_results(
            post_train_csv_path,
            'fusion',
            args.fusion_type,
            args.alpha,
            'N/A',
            'fusion_model',
            'gaussian_noise',
            f'sigma_{sigma}',
            'N/A',
            'N/A',
            res['accuracy'],
            res['loss'],
            res['correct'],
            res['total'],
        )

    # CIFAR-100-C corruptions (experts + fusion)
    print(f"\n{'='*60}")
    print("POST-TRAINING: CIFAR-100-C Corruption Robustness")
    print(f"{'='*60}")
    for i, expert in enumerate(expert_backbones):
        print(f"\nTesting Expert {i} (POST-TRAINING) on corruptions...")
        corruption_results = evaluate_corruption_robustness(expert, args.data_dir, args.batch_size, device)
        for corruption_type, severity_results in corruption_results.items():
            for severity, results in severity_results.items():
                save_post_training_results(
                    post_train_csv_path,
                    'expert',
                    args.fusion_type,
                    args.alpha,
                    i,
                    'wideresnet28_10',
                    'corruption',
                    'N/A',
                    corruption_type,
                    severity,
                    results['accuracy'],
                    results['loss'],
                    results['correct'],
                    results['total'],
                )

    print(f"\nTesting Trained Fusion Model on corruptions...")
    fusion_corruption_results = evaluate_corruption_robustness(
        fusion_model, args.data_dir, args.batch_size, device
    )
    for corruption_type, severity_results in fusion_corruption_results.items():
        for severity, results in severity_results.items():
            save_post_training_results(
                post_train_csv_path,
                'fusion',
                args.fusion_type,
                args.alpha,
                'N/A',
                'fusion_model',
                'corruption',
                'N/A',
                corruption_type,
                severity,
                results['accuracy'],
                results['loss'],
                results['correct'],
                results['total'],
            )

    # OOD detection (experts + fusion)
    print(f"\n{'='*60}")
    print("POST-TRAINING: Out-of-Distribution Detection")
    print(f"{'='*60}")
    for i, expert in enumerate(expert_backbones):
        print(f"\nTesting Expert {i} (POST-TRAINING) on OOD detection...")
        ood_results = evaluate_ood_detection(expert, testloader, args.data_dir, args.batch_size, device)
        save_post_training_ood_results(post_train_csv_path, 'expert', args.fusion_type, args.alpha, i, 'wideresnet28_10', ood_results)

    print(f"\nTesting Trained Fusion Model on OOD detection...")
    fusion_ood_results = evaluate_ood_detection(fusion_model, testloader, args.data_dir, args.batch_size, device)
    save_post_training_ood_results(post_train_csv_path, 'fusion', args.fusion_type, args.alpha, 'N/A', 'fusion_model', fusion_ood_results)

    print(f"\n✅ POST-TRAINING evaluation completed!")
    print(f"   Results saved to: {post_train_csv_path}")

    # PHASE 4: COMPREHENSIVE ROBUSTNESS EVALUATION
    print(f"\n{'='*80}")
    print(f"PHASE 4: COMPREHENSIVE ROBUSTNESS EVALUATION (α={args.alpha})")
    print(f"{'='*80}")
    print(f"\n🔍 Running comprehensive robustness evaluation...")
    run_robustness_evaluation_wrn(
        expert_backbones,
        fusion_model,
        args.fusion_type,
        args.alpha,
        args.output_dir,
        args.data_dir,
        args.batch_size,
        device,
    )


if __name__ == "__main__":
    main()


