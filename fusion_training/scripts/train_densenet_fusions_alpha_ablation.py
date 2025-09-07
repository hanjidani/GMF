#!/usr/bin/env python3
"""
DenseNet Fusion Training Script (Alpha Ablation - No LR Schedulers)

This script mirrors train_densenet_fusions.py but removes learning-rate schedulers
so alpha can be ablated in isolation. Experts, fusion module, and global head
use fixed learning rates throughout training.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add the parent directory to the path to import the base trainer helpers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import helpers from the baseline trainer to avoid duplicating functionality
from train_densenet_fusions import (
    set_seed,
    load_densenet_experts,
    create_densenet_mcn_model,
    mixup_data,
    mixup_criterion,
    cutmix_data,
    dual_path_loss,
    evaluate_experts_during_training,
    setup_csv_logging,
    log_training_epoch,
    save_fusion_model_components,
    load_data_splits_with_optional_val,
)


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
    """Train the fusion model (no LR scheduling; fixed LR for all components).

    Returns:
        best_val_acc: float
        best_expert_paths: list[str | None]
    """
    # Create independent optimizers with different learning rates (fixed)
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

    # Track per-expert best
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
    # Track best global head (main head)
    best_global_dir = None
    best_global_head_path: str | None = None
    if save_dir is not None:
        best_global_dir = os.path.join(save_dir, "global_best")
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
                            "model_state_dict": model.expert_backbones[i].state_dict(),
                            "epoch": epoch + 1,
                            "best_val_acc": acc,
                            "fusion_type": fusion_type,
                            "alpha": alpha,
                            "component": f"expert_{i}",
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
                    "model_state_dict": model.fusion_module.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_acc": val_acc,
                    "fusion_type": fusion_type,
                    "alpha": alpha,
                    "component": "fusion_module_best",
                },
                fusion_best_path,
            )
            print(f"  🔸 Saved new best Fusion Module (val acc {val_acc:.2f}%) to {fusion_best_path}")

        if best_global_dir is not None and val_acc > best_global_val_acc:
            best_global_val_acc = val_acc
            global_best_path = os.path.join(best_global_dir, "global_head_best.pth")
            torch.save(
                {
                    "model_state_dict": model.global_head.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_acc": val_acc,
                    "fusion_type": fusion_type,
                    "alpha": alpha,
                    "component": "global_head_best",
                },
                global_best_path,
            )
            print(f"  🔸 Saved new best Global Head (val acc {val_acc:.2f}%) to {global_best_path}")

        if csv_path:
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            log_training_epoch(
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
            # Save best global head (main head) snapshot
            if best_global_dir is not None:
                best_global_head_path = os.path.join(best_global_dir, "global_head_best.pth")
                torch.save(
                    {
                        "model_state_dict": model.global_head.state_dict(),
                        "epoch": epoch + 1,
                        "best_val_acc": best_val_acc,
                        "fusion_type": fusion_type,
                        "alpha": alpha,
                        "component": "global_head",
                    },
                    best_global_head_path,
                )
                print(f"  🔹 Saved new best Global Head to {best_global_head_path}")

        if save_dir and (epoch + 1) % save_freq == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_experts_state_dict": optim_experts.state_dict(),
                    "optimizer_fusion_state_dict": optim_fusion.state_dict(),
                    "optimizer_global_state_dict": optim_global.state_dict(),
                    "best_val_acc": best_val_acc,
                    "alpha": alpha,
                    "fusion_type": fusion_type,
                },
                checkpoint_path,
            )
            print(f"  Saved checkpoint: {checkpoint_path}")

    if save_dir and best_model_state is not None:
        best_model_path = os.path.join(save_dir, "best_model.pth")
        torch.save(
            {
                "model_state_dict": best_model_state,
                "best_val_acc": best_val_acc,
                "alpha": alpha,
                "fusion_type": fusion_type,
                "final_epoch": epochs,
            },
            best_model_path,
        )
        print(f"  Saved best model: {best_model_path}")

    return best_val_acc, best_expert_paths


def main():
    parser = argparse.ArgumentParser(description="Train DenseNet Fusion Models (Alpha Ablation, No Schedulers)")
    parser.add_argument("--fusion_type", type=str, required=True,
                        choices=["multiplicative", "multiplicativeAddition", "TransformerBase", "concatenation", "simpleAddition"],
                        help="Type of fusion to use")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha for dual-path loss")
    parser.add_argument("--checkpoint_dir", type=str, default='../../expert_training/scripts/checkpoints_expert_iid',
                        help="Directory containing DenseNet expert checkpoints")
    parser.add_argument("--output_dir", type=str, default='../fusion_checkpoints', help="Output directory")
    parser.add_argument("--data_dir", type=str, default='../data', help="Data directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--augmentation_mode", type=str, default='cutmix', choices=['mixup', 'cutmix', 'none'])
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--skip_pre_eval", action='store_true', help='Skip pre-training eval')
    parser.add_argument("--use_train_val_split", action='store_true', help='Use validation split from fusion holdout')
    parser.add_argument("--val_split_ratio", type=float, default=0.1)
    parser.add_argument("--experts_lr_scale", type=float, default=0.1, help='Scale factor for experts LR (multiplied by base_lr)')
    parser.add_argument("--base_lr", type=float, default=1e-4, help='Base LR for experts before scaling')
    parser.add_argument("--head_lr", type=float, default=1e-3, help='LR for fusion and global head')

    args = parser.parse_args()

    print(f"Starting DenseNet {args.fusion_type} fusion training (alpha-ablation, no schedulers)...")
    print(f"Alpha: {args.alpha}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Output directory: {args.output_dir}")

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load experts
    print("Loading DenseNet expert backbones...")
    expert_backbones = load_densenet_experts(args.checkpoint_dir, 4, device)
    print(f"Successfully loaded {len(expert_backbones)} DenseNet experts")

    # Create model
    print("Creating DenseNet MCN fusion model...")
    fusion_model = create_densenet_mcn_model(expert_backbones, args.fusion_type, args.alpha, device)
    print("✅ Successfully created DenseNet MCN fusion model")

    # Data
    print("Loading data splits for fusion training...")
    train_loader, val_loader = load_data_splits_with_optional_val(
        args.data_dir, args.batch_size, args.use_train_val_split, args.val_split_ratio, seed=args.seed
    )
    print("✅ Data splits loaded successfully")

    # CSV logging
    csv_path = setup_csv_logging(args.fusion_type, args.alpha, args.output_dir)
    print(f"CSV logging setup: {csv_path}")

    # Experiment output dir
    experiment_output_dir = os.path.join(args.output_dir, f'densenet_fusions_alpha_{args.alpha}', args.fusion_type)
    os.makedirs(experiment_output_dir, exist_ok=True)
    print(f"  - Experiment Output: {experiment_output_dir}")

    # Train
    print(
        f"\n🚀 Starting DenseNet {args.fusion_type} fusion training with alpha={args.alpha} (no schedulers)..."
    )
    print(
        f"   Augmentation: {args.augmentation_mode.upper()}, MixUp α={args.mixup_alpha}, "
        f"CutMix α={args.cutmix_alpha}, Label Smoothing={args.label_smoothing}, Grad Clip={args.gradient_clip_norm}"
    )

    best_val_acc, best_expert_paths = train_fusion_model(
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
    save_fusion_model_components(
        fusion_model,
        args.fusion_type,
        args.alpha,
        experiment_output_dir,
    )
    print("✅ Model components saved successfully")


if __name__ == "__main__":
    main()



