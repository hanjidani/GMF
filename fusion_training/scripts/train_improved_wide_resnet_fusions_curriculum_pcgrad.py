#!/usr/bin/env python3
"""
Improved WideResNet Fusion Training (Training-Only) with Curriculum + PCGrad

Implements a two-stage schedule:
- Warm-up: freeze experts; train fusion+global head only on global CE
- Joint: unfreeze experts; apply PCGrad to combine gradients from
         global CE and per-expert individual CE (scaled by alpha) for experts

Notes:
- No pre/post-training evaluations. Minimal CSV logging and validation only.
"""

import argparse
import os
import sys
import time
from pathlib import Path
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
FUSION_ROOT = os.path.dirname(os.path.dirname(SCRIPTS_DIR))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPTS_DIR)))

sys.path.append(SCRIPTS_DIR)
sys.path.append(FUSION_ROOT)
sys.path.append(os.path.join(REPO_ROOT, 'expert_training', 'models'))


# -----------------------------------------------------------------------------
# Imports from baseline utilities and models
# -----------------------------------------------------------------------------
from train_densenet_fusions import (
    set_seed,
    load_data_splits_with_optional_val,
)

from models.fusion_models import create_mcn_model
from improved_wide_resnet import improved_wideresnet28_10


# -----------------------------------------------------------------------------
# Expert loading and model creation
# -----------------------------------------------------------------------------
def load_wideresnet_experts(checkpoint_dir: str, num_experts: int, device: torch.device):
    expert_backbones = []
    for i in range(num_experts):
        ckpt_path = os.path.join(checkpoint_dir, f'best_iid_wideresnet28_10_expert_{i}.pth')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"WideResNet expert checkpoint not found: {ckpt_path}")

        model = improved_wideresnet28_10(num_classes=100)
        ckpt = torch.load(ckpt_path, map_location=device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        model = model.to(device)
        expert_backbones.append(model)
    return expert_backbones


def create_improved_wide_resnet_mcn_model(expert_backbones, fusion_type: str, device: torch.device):
    input_dim = 640
    num_classes = 100
    hidden_dim = input_dim
    model = create_mcn_model(
        expert_backbones=expert_backbones,
        input_dim=input_dim,
        num_classes=num_classes,
        fusion_type=fusion_type,
        hidden_dim=hidden_dim,
    )
    return model.to(device)


# -----------------------------------------------------------------------------
# CSV logging (training-only)
# -----------------------------------------------------------------------------
def setup_csv_logging(output_dir: str, fusion_type: str, tag: str) -> str:
    csv_dir = Path(output_dir) / 'csv_logs' / 'wideresnet_fusions_curriculum_pcgrad'
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f'wideresnet_{fusion_type}_{tag}_training_log.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'epoch', 'train_loss', 'val_loss', 'val_acc',
            'experts_lr', 'fusion_lr', 'global_head_lr', 'alpha', 'stage', 'timestamp'
        ])
        writer.writeheader()
    return str(csv_path)


def log_epoch(csv_path: str, epoch: int, train_loss: float, val_loss: float, val_acc: float,
              experts_lr: float, fusion_lr: float, global_head_lr: float, alpha: float, stage: str):
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'epoch', 'train_loss', 'val_loss', 'val_acc',
            'experts_lr', 'fusion_lr', 'global_head_lr', 'alpha', 'stage', 'timestamp'
        ])
        writer.writerow({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'experts_lr': experts_lr,
            'fusion_lr': fusion_lr,
            'global_head_lr': global_head_lr,
            'alpha': alpha,
            'stage': stage,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        })


# -----------------------------------------------------------------------------
# PCGrad utilities (two-task: global CE vs individual CE)
# -----------------------------------------------------------------------------
def _project_conflict(g_i: torch.Tensor, g_j: torch.Tensor) -> torch.Tensor:
    if g_i is None or g_j is None:
        return g_i
    gi = g_i
    gj = g_j
    dot = torch.dot(gi.flatten(), gj.flatten())
    if dot < 0:
        gj_norm_sq = torch.dot(gj.flatten(), gj.flatten()) + 1e-12
        gi = gi - (dot / gj_norm_sq) * gj
    return gi


def pcgrad_combine(g_global_list, g_ind_list):
    combined = []
    for gg, gi in zip(g_global_list, g_ind_list):
        if gg is None and gi is None:
            combined.append(None)
            continue
        if gg is None:
            combined.append(gi)
            continue
        if gi is None:
            combined.append(gg)
            continue
        # Project each to reduce conflicts, then sum
        gg_proj = _project_conflict(gg, gi)
        gi_proj = _project_conflict(gi, gg)
        combined.append(gg_proj + gi_proj)
    return combined


# -----------------------------------------------------------------------------
# Training loop with Curriculum + PCGrad
# -----------------------------------------------------------------------------
def evaluate(model: nn.Module, val_loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            global_logits, _ = model(data)
            loss = criterion(global_logits, targets)
            total_loss += loss.item()
            _, pred = global_logits.max(1)
            total += targets.size(0)
            correct += (pred == targets).sum().item()
    val_loss = total_loss / max(1, len(val_loader))
    val_acc = 100.0 * correct / max(1, total)
    return val_loss, val_acc


def evaluate_with_expert_metrics(model: nn.Module, val_loader: DataLoader, device: torch.device,
                                 compute_expert: bool = False):
    """Evaluate global head; optionally compute per-expert accuracies."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total = 0
    correct_global = 0
    total_loss = 0.0
    expert_correct = None

    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            global_logits, individual_logits = model(data)
            loss = criterion(global_logits, targets)
            total_loss += loss.item()

            _, pred_g = global_logits.max(1)
            total += targets.size(0)
            correct_global += (pred_g == targets).sum().item()

            if compute_expert and isinstance(individual_logits, (list, tuple)):
                if expert_correct is None:
                    expert_correct = [0 for _ in range(len(individual_logits))]
                for idx, logits in enumerate(individual_logits):
                    _, pred_e = logits.max(1)
                    expert_correct[idx] += (pred_e == targets).sum().item()

    val_loss = total_loss / max(1, len(val_loader))
    val_acc = 100.0 * correct_global / max(1, total)
    expert_accs = None
    if compute_expert and expert_correct is not None:
        expert_accs = [100.0 * c / max(1, total) for c in expert_correct]
    return val_loss, val_acc, expert_accs


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    alpha: float,
    epochs: int,
    warmup_epochs: int,
    experts_lr: float,
    head_lr: float,
    gradient_clip_norm: float,
    csv_path: str,
):
    # Parameter groups
    expert_params = list(model.expert_backbones.parameters())
    head_params = list(model.fusion_module.parameters()) + list(model.global_head.parameters())

    optim_experts = optim.AdamW(expert_params, lr=experts_lr, weight_decay=1e-4)
    optim_head = optim.AdamW(head_params, lr=head_lr, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()

    # Warm-up: freeze experts (no grads + keep experts in eval to freeze BN/dropout)
    for p in expert_params:
        p.requires_grad_(False)
    # Ensure module states: experts eval, heads train
    model.eval()
    for expert in model.expert_backbones:
        expert.eval()
    model.fusion_module.train()
    model.global_head.train()

    print(f"Warm-up stage: {warmup_epochs} epochs (experts frozen; BN stats not updating)")
    for epoch in range(1, warmup_epochs + 1):
        # Keep experts eval, heads train each epoch
        for expert in model.expert_backbones:
            expert.eval()
        model.fusion_module.train()
        model.global_head.train()
        train_loss = 0.0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optim_head.zero_grad(set_to_none=True)

            global_logits, _ = model(data)
            loss = criterion(global_logits, targets)
            loss.backward()

            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(head_params, gradient_clip_norm)

            optim_head.step()
            train_loss += loss.item()

        val_loss, val_acc, expert_accs = evaluate_with_expert_metrics(
            model, val_loader, device, compute_expert=True
        )
        avg_train_loss = train_loss / max(1, len(train_loader))
        print(f"[Warm-up] Epoch {epoch}/{warmup_epochs} | Train {avg_train_loss:.4f} | Val {val_loss:.4f} | Acc {val_acc:.2f}%")
        if expert_accs is not None:
            try:
                print(
                    "          Experts: "
                    + ", ".join([f"E{i} {a:.2f}%" for i, a in enumerate(expert_accs)])
                )
            except Exception:
                pass
        log_epoch(csv_path, epoch, avg_train_loss, val_loss, val_acc, 0.0, head_lr, head_lr, alpha, 'warmup')

    # Joint: unfreeze experts + PCGrad on experts (two-task: global vs. individual)
    for p in expert_params:
        p.requires_grad_(True)
    # Restore train mode for all components
    for expert in model.expert_backbones:
        expert.train()
    model.fusion_module.train()
    model.global_head.train()

    print(f"Joint stage: {epochs - warmup_epochs} epochs (PCGrad for experts)")
    for epoch in range(warmup_epochs + 1, epochs + 1):
        # Ensure all submodules are in train during joint phase
        for expert in model.expert_backbones:
            expert.train()
        model.fusion_module.train()
        model.global_head.train()
        train_loss = 0.0

        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)

            # Zero all optimizers
            optim_experts.zero_grad(set_to_none=True)
            optim_head.zero_grad(set_to_none=True)

            # Forward
            global_logits, individual_logits = model(data)
            loss_global = criterion(global_logits, targets)
            loss_individual = torch.tensor(0.0, device=device)
            for logits in individual_logits:
                loss_individual = loss_individual + criterion(logits, targets)

            # PCGrad for experts: grads of two tasks on expert params
            g_global = torch.autograd.grad(loss_global, expert_params, retain_graph=True, allow_unused=True)
            g_ind = torch.autograd.grad(alpha * loss_individual, expert_params, retain_graph=True, allow_unused=True)
            g_comb = pcgrad_combine(g_global, g_ind)

            # Assign combined grads to expert params
            for p, g in zip(expert_params, g_comb):
                if p is None or g is None:
                    continue
                if p.grad is None:
                    p.grad = g.clone()
                else:
                    p.grad.copy_(g)

            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(expert_params, gradient_clip_norm)

            optim_experts.step()

            # Update head (fusion + global head) using global loss only
            # Compute grads explicitly to avoid re-using expert grads
            head_grads = torch.autograd.grad(loss_global, head_params, retain_graph=False, allow_unused=True)
            for p, g in zip(head_params, head_grads):
                if g is None:
                    continue
                if p.grad is None:
                    p.grad = g.clone()
                else:
                    p.grad.copy_(g)

            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(head_params, gradient_clip_norm)

            optim_head.step()

            total_loss = loss_global.item() + alpha * loss_individual.item()
            train_loss += total_loss

        val_loss, val_acc, expert_accs = evaluate_with_expert_metrics(
            model, val_loader, device, compute_expert=True
        )
        avg_train_loss = train_loss / max(1, len(train_loader))
        print(f"[Joint] Epoch {epoch - warmup_epochs}/{epochs - warmup_epochs} | Train {avg_train_loss:.4f} | Val {val_loss:.4f} | Acc {val_acc:.2f}%")
        if expert_accs is not None:
            try:
                print(
                    "        Experts: "
                    + ", ".join([f"E{i} {a:.2f}%" for i, a in enumerate(expert_accs)])
                )
            except Exception:
                pass
        log_epoch(csv_path, epoch, avg_train_loss, val_loss, val_acc, experts_lr, head_lr, head_lr, alpha, 'joint')


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="WRN Fusion Training with Curriculum + PCGrad (training-only)")
    parser.add_argument('--fusion_type', type=str, required=True,
                        choices=['multiplicative', 'multiplicativeAddition', 'TransformerBase', 'concatenation', 'simpleAddition'])
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha for individual branch loss')
    parser.add_argument('--checkpoint_dir', type=str, default='../../expert_training/scripts/checkpoints_expert_iid')
    parser.add_argument('--output_dir', type=str, default='../fusion_checkpoints_curriculum_pcgrad')
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--experts_lr_scale', type=float, default=0.05)
    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--head_lr', type=float, default=1e-3)
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0)
    parser.add_argument('--log_expert_metrics', action='store_true', help='Deprecated flag (expert metrics are printed by default)')

    args = parser.parse_args()

    print(f"Starting WRN curriculum+PCGrad training (training-only)...")
    print(f"Fusion: {args.fusion_type} | Alpha: {args.alpha}")
    print(f"Warm-up epochs: {args.warmup_epochs} | Total epochs: {args.epochs}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Output dir: {args.output_dir}")

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load experts and build model
    expert_backbones = load_wideresnet_experts(args.checkpoint_dir, 4, device)
    model = create_improved_wide_resnet_mcn_model(expert_backbones, args.fusion_type, device)

    # Data
    print("Loading data splits...")
    train_loader, val_loader = load_data_splits_with_optional_val(
        args.data_dir, args.batch_size, True, 0.1, seed=args.seed
    )
    print("✅ Data ready")

    # CSV logging
    tag = f"{args.fusion_type}_alpha_{args.alpha}"
    csv_path = setup_csv_logging(args.output_dir, args.fusion_type, tag)

    # Pre-training evaluation (experts + global)
    print("Pre-training evaluation (experts + global)...")
    pre_val_loss, pre_val_acc, pre_expert_accs = evaluate_with_expert_metrics(
        model, val_loader, device, compute_expert=True
    )
    print(f"[Pre] Val {pre_val_loss:.4f} | Acc {pre_val_acc:.2f}%")
    if pre_expert_accs is not None:
        try:
            print(
                "      Experts: "
                + ", ".join([f"E{i} {a:.2f}%" for i, a in enumerate(pre_expert_accs)])
            )
        except Exception:
            pass
    log_epoch(csv_path, 0, 0.0, pre_val_loss, pre_val_acc, 0.0, args.head_lr, args.head_lr, args.alpha, 'pre')

    experts_lr = args.base_lr * args.experts_lr_scale
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        alpha=args.alpha,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        experts_lr=experts_lr,
        head_lr=args.head_lr,
        gradient_clip_norm=args.gradient_clip_norm,
        csv_path=csv_path,
    )

    print("✅ Training finished (curriculum + PCGrad). Logs saved to:")
    print(csv_path)


if __name__ == '__main__':
    main()


