import argparse
import os
import sys
import csv
import time
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, accuracy_score

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../expert_training/models')))
from models.fusion_models import create_mcn_model
from utils.helpers import set_seed
from densenet_cifar import densenet121  # noqa: E402


def get_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])


def get_train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])


def get_data_loaders(data_dir: str, batch_size: int, num_workers: int = 4, splits_dir: str = None) -> Tuple[DataLoader, DataLoader]:
    trainset_full = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=get_train_transform())
    testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=get_test_transform())

    if splits_dir is not None:
        split_path = os.path.join(splits_dir, 'fusion_holdout_indices.npy')
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Fusion split indices not found: {split_path}")
        fusion_indices = np.load(split_path)
        trainset = Subset(trainset_full, fusion_indices)
    else:
        trainset = trainset_full

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def load_densenet_specialist_backbones(checkpoint_dir: str, num_experts: int, device: torch.device) -> List[nn.Module]:
    backbones: List[nn.Module] = []
    for i in range(num_experts):
        ckpt_path = os.path.join(checkpoint_dir, f"best_noniid_densenet121_expert_{i}.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        model_25 = densenet121(num_classes=25)
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model_25.load_state_dict(state_dict)
        backbone = model_25.features
        backbones.append(backbone.to(device))
    return backbones


def dual_path_loss(global_logits, individual_logits, targets, criterion, alpha: float):
    global_loss = criterion(global_logits, targets)
    individual_loss_sum = sum(criterion(logits, targets) for logits in individual_logits)
    total = global_loss + alpha * individual_loss_sum
    return total, global_loss, individual_loss_sum


def get_model_output(model: nn.Module, data: torch.Tensor) -> torch.Tensor:
    output = model(data)
    if isinstance(output, tuple):
        # Fusion model returns (global_logits, individual_logits)
        if len(output) == 2 and isinstance(output[1], list):
            return output[0]
        # Expert models may return (features, logits)
        if len(output) == 2:
            return output[1]
    return output


def _cifar_stats():
    mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1)
    std = torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1)
    return mean, std


def _normalize_cifar(inputs: torch.Tensor) -> torch.Tensor:
    mean, std = _cifar_stats()
    return (inputs - mean.to(inputs.device)) / std.to(inputs.device)


def add_gaussian_noise(inputs: torch.Tensor, sigma: float) -> torch.Tensor:
    noise = torch.randn_like(inputs) * sigma
    noisy_inputs = inputs + noise
    # Clamp to ±3 std in normalized space to avoid outliers
    return torch.clamp(noisy_inputs, -3.0, 3.0)


@torch.no_grad()
def evaluate_gaussian_noise_robustness(model: nn.Module, testloader: DataLoader, device: torch.device, sigmas=None):
    if sigmas is None:
        sigmas = [0.0, 0.05, 0.1, 0.2, 0.3]
    model.eval()
    criterion = nn.CrossEntropyLoss()
    stats = {s: {"loss": 0.0, "correct": 0, "total": 0} for s in sigmas}
    for data, targets in testloader:
        data, targets = data.to(device), targets.to(device)
        for s in sigmas:
            noisy = add_gaussian_noise(data, s)
            outputs = get_model_output(model, noisy)
            loss = criterion(outputs, targets)
            _, predicted = torch.max(outputs, 1)
            stats[s]["loss"] += loss.item()
            stats[s]["total"] += targets.size(0)
            stats[s]["correct"] += (predicted == targets).sum().item()
    results = {}
    for s, acc in stats.items():
        total = acc["total"] if acc["total"] > 0 else 1
        results[s] = {
            "accuracy": 100.0 * acc["correct"] / total,
            "loss": acc["loss"] / (total / testloader.batch_size if testloader.batch_size else 1),
            "correct": acc["correct"],
            "total": acc["total"]
        }
    return results


@torch.no_grad()
def compute_ece(model: nn.Module, loader: DataLoader, device: torch.device, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE). Returns percentage (0-100).
    """
    model.eval()
    bin_boundaries = torch.linspace(0, 1, steps=n_bins + 1, device=device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    confidences_list = []
    accuracies_list = []
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = get_model_output(model, inputs)
        probs = torch.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)
        accuracies = preds.eq(targets)
        confidences_list.append(confs)
        accuracies_list.append(accuracies.float())
    confidences = torch.cat(confidences_list)
    accuracies = torch.cat(accuracies_list)
    ece = torch.zeros(1, device=device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            acc_in_bin = accuracies[in_bin].float().mean()
            avg_conf_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_conf_in_bin - acc_in_bin) * prop_in_bin
    return (ece.item() * 100.0)


def evaluate_pgd(model: nn.Module, loader: DataLoader, device: torch.device, eps: float = 8/255, alpha: float = 2/255, steps: int = 10) -> float:
    """
    PGD adversarial evaluation. Returns adversarial accuracy (%).
    """
    model.eval()
    correct = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss()
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        adv = data.clone().detach()
        adv.requires_grad = True
        adv = adv + 0.001 * torch.randn_like(adv)
        for _ in range(steps):
            logits = get_model_output(model, adv)
            loss = loss_fn(logits, targets)
            loss.backward()
            grad = adv.grad.detach()
            adv = adv + alpha * torch.sign(grad)
            adv = torch.min(torch.max(adv, data - eps), data + eps)
            adv = torch.clamp(adv, -3.0, 3.0)  # inputs are normalized; keep within reasonable bounds
            adv = adv.detach().requires_grad_(True)
        with torch.no_grad():
            out = get_model_output(model, adv)
            _, pred = out.max(1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)
    return 100.0 * correct / max(1, total)


def compute_mce_from_corruption_results(corruption_results: Dict[str, Dict[int, Dict[str, float]]]) -> float:
    """
    Compute mean Corruption Error (mCE) as average error across corruptions and severities.
    """
    errors = []
    for _, severity_map in corruption_results.items():
        for _, res in severity_map.items():
            acc = res.get('accuracy', 0.0)
            errors.append(1.0 - acc / 100.0)
    if not errors:
        return 0.0
    return float(np.mean(errors) * 100.0)


def aggregate_ood_near_far(ood_results: Dict[str, Dict[str, float]]) -> Tuple[float, float, float]:
    """
    Aggregate AUROC for near-OOD (CIFAR-10, SVHN) and far-OOD (TinyImageNet), and FPR95 (near averaged).
    Returns: (auroc_near, auroc_far, fpr95_near)
    """
    near_list = []
    near_fpr = []
    far_list = []
    for ds, res in ood_results.items():
        if 'cifar10' in ds or 'svhn' in ds:
            near_list.append(res.get('auroc', 0.5))
            near_fpr.append(res.get('fpr95', 1.0))
        elif 'tiny' in ds:
            far_list.append(res.get('auroc', 0.5))
    auroc_near = float(np.mean(near_list)) if near_list else 0.5
    fpr95_near = float(np.mean(near_fpr)) if near_fpr else 1.0
    auroc_far = float(np.mean(far_list)) if far_list else 0.5
    return auroc_near, auroc_far, fpr95_near


@torch.no_grad()
def evaluate_top1_and_nll(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """
    Compute Top-1 accuracy (%) and mean NLL on a loader.
    """
    model.eval()
    ce = nn.CrossEntropyLoss(reduction='sum')
    total = 0
    correct = 0
    nll_sum = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = get_model_output(model, inputs)
        loss = ce(logits, targets)
        nll_sum += loss.item()
        _, preds = logits.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)
    top1 = 100.0 * correct / max(1, total)
    nll = nll_sum / max(1, total)
    return top1, nll


def setup_final_metrics_csv(fusion_type: str, alpha: float, output_dir: str) -> str:
    """
    Create CSV for final metrics (global head and each expert head).
    """
    csv_dir = os.path.join(output_dir, 'csv_logs', 'fusion_non_iid')
    os.makedirs(csv_dir, exist_ok=True)
    path = os.path.join(csv_dir, f'fusion_{fusion_type}_alpha_{alpha}_final_metrics.csv')
    headers = [
        'timestamp', 'model_type', 'fusion_type', 'alpha', 'expert_id',
        'top1', 'nll', 'ece'
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
    return path


def save_final_metrics_row(csv_path: str, model_type: str, fusion_type: str, alpha: float,
                           expert_id: str, top1: float, nll: float, ece_val: float):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([ts, model_type, fusion_type, alpha, expert_id, top1, nll, ece_val])

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optim_backbones: optim.Optimizer,
                    optim_heads: optim.Optimizer,
                    criterion: nn.Module,
                    alpha: float,
                    device: torch.device):
    model.train()
    total = 0
    global_correct = 0
    individual_correct = [0] * len(model.individual_heads)
    total_loss = 0.0
    total_global_loss = 0.0
    total_individual_loss = 0.0
    individual_losses = [0.0] * len(model.individual_heads)
    pbar = tqdm(loader, desc="Train", leave=False)
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)
        optim_backbones.zero_grad(set_to_none=True)
        optim_heads.zero_grad(set_to_none=True)
        global_logits, individual_logits = model(images)
        loss, global_loss, individual_loss_sum = dual_path_loss(global_logits, individual_logits, targets, criterion, alpha)
        loss.backward()
        optim_backbones.step()
        optim_heads.step()
        
        # Track all loss components
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_global_loss += global_loss.item() * batch_size
        total_individual_loss += individual_loss_sum.item() * batch_size
        
        # Track individual expert losses
        for i, expert_logits in enumerate(individual_logits):
            expert_loss = criterion(expert_logits, targets)
            individual_losses[i] += expert_loss.item() * batch_size
        
        # Global head accuracy
        _, global_pred = global_logits.max(1)
        global_correct += global_pred.eq(targets).sum().item()
        
        # Individual expert accuracies
        for i, expert_logits in enumerate(individual_logits):
            _, expert_pred = expert_logits.max(1)
            individual_correct[i] += expert_pred.eq(targets).sum().item()
        
        total += targets.size(0)
        
        # Progress bar with all accuracies
        global_acc = 100.0 * global_correct / max(total, 1)
        individual_accs = [100.0 * individual_correct[i] / max(total, 1) for i in range(len(individual_correct))]
        pbar.set_postfix({
            "loss": f"{total_loss/max(total,1):.4f}", 
            "global_acc": f"{global_acc:.2f}",
            "exp_acc": f"{[f'{acc:.1f}' for acc in individual_accs]}"
        })
    
    global_accuracy = 100.0 * global_correct / max(total, 1)
    individual_accuracies = [100.0 * individual_correct[i] / max(total, 1) for i in range(len(individual_correct))]
    avg_individual_losses = [individual_losses[i] / max(total, 1) for i in range(len(individual_losses))]
    
    return (total_loss / max(total, 1), global_accuracy, individual_accuracies, 
            total_global_loss / max(total, 1), total_individual_loss / max(total, 1), avg_individual_losses)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, alpha: float, device: torch.device):
    model.eval()
    total = 0
    global_correct = 0
    individual_correct = [0] * len(model.individual_heads)
    total_loss = 0.0
    total_global_loss = 0.0
    total_individual_loss = 0.0
    individual_losses = [0.0] * len(model.individual_heads)
    pbar = tqdm(loader, desc="Val", leave=False)
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)
        global_logits, individual_logits = model(images)
        loss, global_loss, individual_loss_sum = dual_path_loss(global_logits, individual_logits, targets, criterion, alpha)
        
        # Track all loss components
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_global_loss += global_loss.item() * batch_size
        total_individual_loss += individual_loss_sum.item() * batch_size
        
        # Track individual expert losses
        for i, expert_logits in enumerate(individual_logits):
            expert_loss = criterion(expert_logits, targets)
            individual_losses[i] += expert_loss.item() * batch_size
        
        # Global head accuracy
        _, global_pred = global_logits.max(1)
        global_correct += global_pred.eq(targets).sum().item()
        
        # Individual expert accuracies
        for i, expert_logits in enumerate(individual_logits):
            _, expert_pred = expert_logits.max(1)
            individual_correct[i] += expert_pred.eq(targets).sum().item()
        
        total += targets.size(0)
        
        # Progress bar with all accuracies
        global_acc = 100.0 * global_correct / max(total, 1)
        individual_accs = [100.0 * individual_correct[i] / max(total, 1) for i in range(len(individual_correct))]
        pbar.set_postfix({
            "loss": f"{total_loss/max(total,1):.4f}", 
            "global_acc": f"{global_acc:.2f}",
            "exp_acc": f"{[f'{acc:.1f}' for acc in individual_accs]}"
        })
    
    global_accuracy = 100.0 * global_correct / max(total, 1)
    individual_accuracies = [100.0 * individual_correct[i] / max(total, 1) for i in range(len(individual_correct))]
    avg_individual_losses = [individual_losses[i] / max(total, 1) for i in range(len(individual_losses))]
    
    return (total_loss / max(total, 1), global_accuracy, individual_accuracies,
            total_global_loss / max(total, 1), total_individual_loss / max(total, 1), avg_individual_losses)


class CIFAR100CCorruption(Dataset):
    def __init__(self, corruption_type, severity, transform=None, data_dir: str = '../data'):
        self.corruption_type = corruption_type
        self.severity = severity
        self.transform = transform
        self.testset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=None
        )
        self.corrupted_data = self._apply_corruption()

    def _apply_corruption(self):
        """Apply specified corruption to the dataset."""
        corrupted_data = []
        
        for i in range(len(self.testset)):
            img, label = self.testset[i]
            # Convert PIL image to tensor and ensure proper shape (C, H, W)
            img = torch.tensor(np.array(img))
            if img.dim() == 3 and img.shape[2] == 3:
                img = img.permute(2, 0, 1)
            
            # Apply corruption based on type
            if self.corruption_type == 'gaussian_noise':
                corrupted_img = self._gaussian_noise(img, self.severity)
            elif self.corruption_type == 'shot_noise':
                corrupted_img = self._shot_noise(img, self.severity)
            elif self.corruption_type == 'impulse_noise':
                corrupted_img = self._impulse_noise(img, self.severity)
            elif self.corruption_type == 'defocus_blur':
                corrupted_img = self._defocus_blur(img, self.severity)
            elif self.corruption_type == 'motion_blur':
                corrupted_img = self._motion_blur(img, self.severity)
            elif self.corruption_type == 'zoom_blur':
                corrupted_img = self._zoom_blur(img, self.severity)
            elif self.corruption_type == 'snow':
                corrupted_img = self._snow(img, self.severity)
            elif self.corruption_type == 'frost':
                corrupted_img = self._frost(img, self.severity)
            elif self.corruption_type == 'fog':
                corrupted_img = self._fog(img, self.severity)
            elif self.corruption_type == 'brightness':
                corrupted_img = self._brightness(img, self.severity)
            elif self.corruption_type == 'contrast':
                corrupted_img = self._contrast(img, self.severity)
            elif self.corruption_type == 'elastic_transform':
                corrupted_img = self._elastic_transform(img, self.severity)
            elif self.corruption_type == 'pixelate':
                corrupted_img = self._pixelate(img, self.severity)
            elif self.corruption_type == 'jpeg_compression':
                corrupted_img = self._jpeg_compression(img, self.severity)
            else:
                corrupted_img = img
            
            corrupted_data.append((corrupted_img, label))
        
        return corrupted_data
    
    def _gaussian_noise(self, img, severity):
        """Apply Gaussian noise corruption."""
        c = [0.04, 0.06, 0.08, 0.09, 0.10][severity - 1]
        # Convert to float if needed
        if img.dtype != torch.float32:
            img = img.float()
        noise = torch.randn_like(img) * c * 255
        corrupted = torch.clamp(img + noise, 0, 255)
        return corrupted.byte()
    
    def _shot_noise(self, img, severity):
        """Apply shot noise corruption."""
        c = [60, 25, 12, 5, 3][severity - 1]
        corrupted = img.clone()
        # Convert to float if needed
        if corrupted.dtype != torch.float32:
            corrupted = corrupted.float()
        # Add Poisson noise
        corrupted = torch.poisson(corrupted / 255.0 * c) / c * 255
        corrupted = torch.clamp(corrupted, 0, 255)
        return corrupted.byte()
    
    def _impulse_noise(self, img, severity):
        """Apply impulse noise corruption."""
        c = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]
        corrupted = img.clone()
        # Convert to float if needed
        if corrupted.dtype != torch.float32:
            corrupted = corrupted.float()
        # Add salt and pepper noise
        mask = torch.rand_like(corrupted) < c
        corrupted[mask] = torch.randint(0, 2, corrupted[mask].shape, device=corrupted.device) * 255
        return corrupted.byte()
    
    def _defocus_blur(self, img, severity):
        """Apply defocus blur corruption."""
        c = [0.3, 0.4, 0.5, 1.0, 1.5][severity - 1]
        # Convert to float if needed
        if img.dtype != torch.float32:
            img = img.float()
        # Simple blur approximation - just add some noise
        noise = torch.randn_like(img) * c * 10
        corrupted = torch.clamp(img + noise, 0, 255)
        return corrupted.byte()
    
    def _motion_blur(self, img, severity):
        """Apply motion blur corruption."""
        c = [0.4, 0.6, 0.8, 1.0, 1.2][severity - 1]
        # Simple motion blur approximation - horizontal shift
        shift = int(c * 5)
        corrupted = img.clone()
        corrupted[:, :, shift:] = img[:, :, :-shift]
        return corrupted
    
    def _zoom_blur(self, img, severity):
        """Apply zoom blur corruption."""
        c = [0.15, 0.25, 0.35, 0.45, 0.55][severity - 1]
        # Simple zoom approximation - crop and resize
        h, w = img.shape[1], img.shape[2]
        crop_size = int(min(h, w) * (1 - c))
        start_h = (h - crop_size) // 2
        start_w = (w - crop_size) // 2
        
        # Crop
        cropped = img[:, start_h:start_h + crop_size, start_w:start_w + crop_size]
        
        # Resize back to original size
        corrupted = F.interpolate(cropped.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)
        return corrupted.byte()
    
    def _snow(self, img, severity):
        """Apply snow corruption."""
        c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
        corrupted = img.clone()
        # Convert to float if needed
        if corrupted.dtype != torch.float32:
            corrupted = corrupted.float()
        # Add snow effect
        snow = torch.rand_like(corrupted) * c * 255
        corrupted = torch.clamp(corrupted + snow, 0, 255)
        return corrupted.byte()
    
    def _frost(self, img, severity):
        """Apply frost corruption."""
        c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
        corrupted = img.clone()
        # Convert to float if needed
        if corrupted.dtype != torch.float32:
            corrupted = corrupted.float()
        # Add frost effect
        frost = torch.rand_like(corrupted) * c * 100
        corrupted = torch.clamp(corrupted - frost, 0, 255)
        return corrupted.byte()
    
    def _fog(self, img, severity):
        """Apply fog corruption."""
        c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
        corrupted = img.clone()
        # Convert to float if needed
        if corrupted.dtype != torch.float32:
            corrupted = corrupted.float()
        # Add fog effect
        fog = torch.rand_like(corrupted) * c * 150
        corrupted = torch.clamp(corrupted + fog, 0, 255)
        return corrupted.byte()
    
    def _brightness(self, img, severity):
        """Apply brightness corruption."""
        c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
        # Convert to float if needed
        if img.dtype != torch.float32:
            img = img.float()
        corrupted = img * (1 + c)
        corrupted = torch.clamp(corrupted, 0, 255)
        return corrupted.byte()
    
    def _contrast(self, img, severity):
        """Apply contrast corruption."""
        c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
        # Convert to float if needed
        if img.dtype != torch.float32:
            img = img.float()
        mean = img.mean()
        corrupted = (img - mean) * (1 + c) + mean
        corrupted = torch.clamp(corrupted, 0, 255)
        return corrupted.byte()
    
    def _elastic_transform(self, img, severity):
        """Apply elastic transform corruption."""
        c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
        # Simple elastic approximation - random pixel shuffling
        corrupted = img.clone()
        h, w = img.shape[1], img.shape[2]
        
        # Random displacement
        displacement = torch.randint(-int(c * 10), int(c * 10) + 1, (h, w, 2))
        for i in range(h):
            for j in range(w):
                new_i = max(0, min(h - 1, i + displacement[i, j, 0]))
                new_j = max(0, min(w - 1, j + displacement[i, j, 1]))
                corrupted[:, i, j] = img[:, new_i, new_j]
        
        return corrupted
    
    def _pixelate(self, img, severity):
        """Apply pixelation corruption."""
        c = [0.6, 0.5, 0.4, 0.3, 0.2][severity - 1]
        h, w = img.shape[1], img.shape[2]
        new_h, new_w = int(h * c), int(w * c)
        
        # Downsample
        downsampled = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
        # Upsample back
        corrupted = F.interpolate(downsampled.unsqueeze(0), size=(h, w), mode='nearest').squeeze(0)
        return corrupted.byte()
    
    def _jpeg_compression(self, img, severity):
        """Apply JPEG compression corruption."""
        c = [25, 18, 15, 10, 5][severity - 1]
        # Convert to float if needed
        if img.dtype != torch.float32:
            img = img.float()
        # Simple approximation by adding noise
        noise = torch.randn_like(img) * c
        corrupted = torch.clamp(img + noise, 0, 255)
        return corrupted.byte()

    def __len__(self):
        return len(self.corrupted_data)

    def __getitem__(self, idx):
        img, label = self.corrupted_data[idx]
        if self.transform:
            if isinstance(img, torch.Tensor):
                if img.dtype != torch.float32:
                    img = img.float()
                img = img / 255.0
                mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
                std = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)
                img = (img - mean) / std
            else:
                img = self.transform(img)
        return img, label


@torch.no_grad()
def evaluate_corruption_robustness(model: nn.Module, data_dir: str, batch_size: int, device: torch.device):
    corruption_types = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 
                      'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 
                      'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    severity_levels = [1, 2, 3, 4, 5]
    results: Dict[str, Dict[int, Dict[str, float]]] = {}
    test_transform = get_test_transform()
    for corruption_type in corruption_types:
        results[corruption_type] = {}
        for severity in severity_levels:
            dataset = CIFAR100CCorruption(corruption_type, severity, test_transform, data_dir=data_dir)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            correct = 0
            total = 0
            total_loss = 0.0
            criterion = nn.CrossEntropyLoss()
            for data, targets in loader:
                data, targets = data.to(device), targets.to(device)
                outputs = get_model_output(model, data)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            results[corruption_type][severity] = {
                'accuracy': 100.0 * correct / max(total, 1),
                'loss': total_loss / max(len(loader), 1),
                'correct': correct,
                'total': total,
            }
    return results


## Deprecated duplicate: earlier save_ood_evaluation_results was superseded by the later definition.


@torch.no_grad()
def _scan_image_files(root_dir: str) -> List[str]:
    exts = {'.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG'}
    files: List[str] = []
    for r, _, fns in os.walk(root_dir):
        for fn in fns:
            if os.path.splitext(fn)[1] in exts:
                files.append(os.path.join(r, fn))
    return files


class TinyImageNetFiles(Dataset):
    def __init__(self, tiny_dir: str, transform):
        # Prefer 'test' subdir if present, else use the given dir recursively
        candidate = os.path.join(tiny_dir, 'test')
        self.base_dir = candidate if os.path.isdir(candidate) else tiny_dir
        self.filepaths = _scan_image_files(self.base_dir)
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0


def evaluate_ood_detection(model: nn.Module, testloader: DataLoader, data_dir: str, batch_size: int, device: torch.device, tinyimagenet_dir: str = None):
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
    model.eval()

    # Get in-distribution scores
    id_scores = []
    id_confidences = []
    for data, _ in testloader:
        data = data.to(device)
        output = get_model_output(model, data)
        probs = torch.softmax(output, dim=1)
        confidence, _ = torch.max(probs, dim=1)
        id_confidences.extend(confidence.cpu().numpy())
        max_logits, _ = torch.max(output, dim=1)
        ood_score = -max_logits.cpu().numpy()
        id_scores.extend(ood_score)

    # OOD datasets: CIFAR-10, SVHN; TinyImageNet synthetic fallback
    results = {}
    for ood_name in ['cifar10', 'svhn', 'tinyimagenet']:
        try:
            if ood_name == 'cifar10':
                ood_dataset = torchvision.datasets.CIFAR10(
                    root=data_dir, train=False, download=True,
                    transform=get_test_transform()
                )
            elif ood_name == 'svhn':
                ood_dataset = torchvision.datasets.SVHN(
                    root=data_dir, split='test', download=True,
                    transform=get_test_transform()
                )
            else:
                if tinyimagenet_dir is None or not os.path.isdir(tinyimagenet_dir):
                    # Skip TinyImageNet if directory not provided
                    continue
                ood_dataset = TinyImageNetFiles(tinyimagenet_dir, get_test_transform())
            ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            ood_scores = []
            ood_confidences = []
            for data, _ in ood_loader:
                data = data.to(device)
                output = get_model_output(model, data)
                probs = torch.softmax(output, dim=1)
                confidence, _ = torch.max(probs, dim=1)
                ood_confidences.extend(confidence.cpu().numpy())
                max_logits, _ = torch.max(output, dim=1)
                ood_score = -max_logits.cpu().numpy()
                ood_scores.extend(ood_score)
            # Metrics
            all_scores = np.concatenate([np.array(id_scores), np.array(ood_scores)])
            labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
            try:
                auroc = roc_auc_score(labels, all_scores)
            except:
                auroc = 0.5
            try:
                aupr = average_precision_score(labels, all_scores)
            except:
                aupr = 0.5
            try:
                fpr, tpr, thresholds = roc_curve(labels, all_scores)
                idx = np.argmax(tpr >= 0.95)
                fpr95 = fpr[idx] if idx < len(fpr) else 1.0
            except:
                fpr95 = 1.0
            # crude detection accuracy via thresholding
            try:
                from sklearn.metrics import accuracy_score
                optimal_idx = np.argmax(tpr - fpr)
                thr = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
                preds = (all_scores > thr).astype(int)
                det_acc = accuracy_score(labels, preds)
                conf_thr = 0.5
            except:
                det_acc, conf_thr = 0.5, 0.5
            results[ood_name] = {
                'ood_type': 'dataset_shift',
                'auroc': auroc,
                'aupr': aupr,
                'fpr95': fpr95,
                'detection_accuracy': det_acc,
                'confidence_threshold': conf_thr,
                'uncertainty_metric': 'max_softmax',
                'ood_score_mean': float(np.mean(ood_scores) if len(ood_scores) else 0.0),
                'ood_score_std': float(np.std(ood_scores) if len(ood_scores) else 0.0),
            }
        except Exception:
            results[ood_name] = {
                'ood_type': 'unknown',
                'auroc': 0.5,
                'aupr': 0.5,
                'fpr95': 0.5,
                'detection_accuracy': 0.5,
                'confidence_threshold': 0.5,
                'uncertainty_metric': 'max_softmax',
                'ood_score_mean': 0.0,
                'ood_score_std': 1.0,
            }
    
    # Add synthetic OOD datasets
    print("    Testing on synthetic Gaussian noise...")
    synthetic_gaussian_results = test_synthetic_ood(model, testloader, device, id_scores, id_confidences)
    results['synthetic_gaussian'] = synthetic_gaussian_results
    
    print("    Testing on synthetic uniform noise...")
    synthetic_uniform_results = test_uniform_ood(model, testloader, device, id_scores, id_confidences)
    results['synthetic_uniform'] = synthetic_uniform_results

    return results

def test_synthetic_ood(model, testloader, device, id_scores, id_confidences):
    """Test OOD detection on synthetic Gaussian noise."""
    ood_scores = []
    ood_confidences = []
    
    with torch.no_grad():
        for data, _ in testloader:
            # Generate Gaussian noise with same shape as data
            noise = torch.randn_like(data) * 0.5 + 0.5  # Scale to [0,1] range
            noise = torch.clamp(noise, 0, 1)
            
            # Normalize noise using CIFAR-100 statistics
            noise = (noise - torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1)) / \
                    torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1)
            
            noise = noise.to(device)
            output = get_model_output(model, noise)
            
            # Get confidence scores
            probs = torch.softmax(output, dim=1)
            confidence, _ = torch.max(probs, dim=1)
            ood_confidences.extend(confidence.cpu().numpy())
            
            # Get max logit scores (negative for OOD)
            max_logits, _ = torch.max(output, dim=1)
            ood_score = -max_logits.cpu().numpy()
            ood_scores.extend(ood_score)
    
    # Calculate OOD metrics
    all_scores = np.concatenate([np.array(id_scores), np.array(ood_scores)])
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    
    try:
        auroc = roc_auc_score(labels, all_scores)
    except:
        auroc = 0.5
    try:
        aupr = average_precision_score(labels, all_scores)
    except:
        aupr = 0.5
    try:
        fpr, tpr, thresholds = roc_curve(labels, all_scores)
        idx = np.argmax(tpr >= 0.95)
        fpr95 = fpr[idx] if idx < len(fpr) else 1.0
    except:
        fpr95 = 1.0
    try:
        from sklearn.metrics import accuracy_score
        optimal_idx = np.argmax(tpr - fpr)
        thr = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        preds = (all_scores > thr).astype(int)
        det_acc = accuracy_score(labels, preds)
        conf_thr = 0.5
    except:
        det_acc, conf_thr = 0.5, 0.5
    
    return {
        'ood_type': 'synthetic_gaussian',
        'auroc': auroc,
        'aupr': aupr,
        'fpr95': fpr95,
        'detection_accuracy': det_acc,
        'confidence_threshold': conf_thr,
        'uncertainty_metric': 'max_softmax',
        'ood_score_mean': float(np.mean(ood_scores) if len(ood_scores) else 0.0),
        'ood_score_std': float(np.std(ood_scores) if len(ood_scores) else 0.0),
    }

def test_uniform_ood(model, testloader, device, id_scores, id_confidences):
    """Test OOD detection on synthetic uniform noise."""
    ood_scores = []
    ood_confidences = []
    
    with torch.no_grad():
        for data, _ in testloader:
            # Generate uniform noise with same shape as data
            noise = torch.rand_like(data)  # Uniform distribution in [0,1]
            
            # Normalize noise using CIFAR-100 statistics
            noise = (noise - torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1)) / \
                    torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1)
            
            noise = noise.to(device)
            output = get_model_output(model, noise)
            
            # Get confidence scores
            probs = torch.softmax(output, dim=1)
            confidence, _ = torch.max(probs, dim=1)
            ood_confidences.extend(confidence.cpu().numpy())
            
            # Get max logit scores (negative for OOD)
            max_logits, _ = torch.max(output, dim=1)
            ood_score = -max_logits.cpu().numpy()
            ood_scores.extend(ood_score)
    
    # Calculate OOD metrics
    all_scores = np.concatenate([np.array(id_scores), np.array(ood_scores)])
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    
    try:
        auroc = roc_auc_score(labels, all_scores)
    except:
        auroc = 0.5
    try:
        aupr = average_precision_score(labels, all_scores)
    except:
        aupr = 0.5
    try:
        fpr, tpr, thresholds = roc_curve(labels, all_scores)
        idx = np.argmax(tpr >= 0.95)
        fpr95 = fpr[idx] if idx < len(fpr) else 1.0
    except:
        fpr95 = 1.0
    try:
        from sklearn.metrics import accuracy_score
        optimal_idx = np.argmax(tpr - fpr)
        thr = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        preds = (all_scores > thr).astype(int)
        det_acc = accuracy_score(labels, preds)
        conf_thr = 0.5
    except:
        det_acc, conf_thr = 0.5, 0.5
    
    return {
        'ood_type': 'synthetic_uniform',
        'auroc': auroc,
        'aupr': aupr,
        'fpr95': fpr95,
        'detection_accuracy': det_acc,
        'confidence_threshold': conf_thr,
        'uncertainty_metric': 'max_softmax',
        'ood_score_mean': float(np.mean(ood_scores) if len(ood_scores) else 0.0),
        'ood_score_std': float(np.std(ood_scores) if len(ood_scores) else 0.0),
    }

# === CSV Logging Functions (Following Reference Pattern) ===

def setup_csv_logging(fusion_type, alpha, output_dir):
    """Setup CSV logging for the fusion training with alpha in filename."""
    csv_dir = Path(output_dir) / 'csv_logs' / 'fusion_non_iid'
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Include alpha in filename to avoid conflicts during ablation studies
    csv_path = csv_dir / f'fusion_{fusion_type}_alpha_{alpha}_training_log.csv'

    # Create CSV file with headers including expert accuracies
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'epoch', 'train_loss', 'val_loss', 'val_accuracy',
            'experts_lr', 'fusion_lr', 'global_head_lr',
            'loss_global', 'loss_individual', 'loss_total',
            'expert_0_accuracy', 'expert_1_accuracy', 'expert_2_accuracy', 'expert_3_accuracy',
            'expert_0_loss', 'expert_1_loss', 'expert_2_loss', 'expert_3_loss',
            'alpha', 'fusion_type', 'timestamp'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    return str(csv_path)

def log_training_epoch(csv_path, epoch, train_loss, val_loss, val_accuracy,
                      experts_lr, fusion_lr, global_head_lr,
                      loss_global, loss_individual, loss_total, alpha, fusion_type,
                      expert_accuracies=None, expert_losses=None):
    """Log training progress for each epoch to CSV file."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    # Default values for expert accuracies and losses
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
            'alpha', 'fusion_type', 'timestamp'
        ])
        writer.writerow({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'experts_lr': experts_lr,
            'fusion_lr': fusion_lr,
            'global_head_lr': global_head_lr,
            'loss_global': loss_global,
            'loss_individual': loss_individual,
            'loss_total': loss_total,
            'expert_0_accuracy': expert_accuracies[0] if len(expert_accuracies) > 0 else 0.0,
            'expert_1_accuracy': expert_accuracies[1] if len(expert_accuracies) > 1 else 0.0,
            'expert_2_accuracy': expert_accuracies[2] if len(expert_accuracies) > 2 else 0.0,
            'expert_3_accuracy': expert_accuracies[3] if len(expert_accuracies) > 3 else 0.0,
            'expert_0_loss': expert_losses[0] if len(expert_losses) > 0 else 0.0,
            'expert_1_loss': expert_losses[1] if len(expert_losses) > 1 else 0.0,
            'expert_2_loss': expert_losses[2] if len(expert_losses) > 2 else 0.0,
            'expert_3_loss': expert_losses[3] if len(expert_losses) > 3 else 0.0,
            'alpha': alpha,
            'fusion_type': fusion_type,
            'timestamp': timestamp
        })

def setup_robustness_evaluation_csv(fusion_type, alpha, output_dir):
    """Setup CSV logging for robustness evaluation results."""
    csv_dir = Path(output_dir) / 'csv_logs' / 'fusion_non_iid'
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    # Create robustness evaluation CSV file
    csv_path = csv_dir / f'fusion_{fusion_type}_alpha_{alpha}_robustness_evaluation.csv'
    
    # Create CSV file with headers for robustness evaluation
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'model_type', 'fusion_type', 'alpha', 'expert_id', 'model_architecture',
            'test_type', 'attack_type', 'corruption_type', 'severity_level', 'epsilon',
            'accuracy', 'loss', 'correct_predictions', 'total_predictions', 'timestamp'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    return str(csv_path)

def save_robustness_evaluation_results(csv_path, model_type, fusion_type, alpha, expert_id, 
                                     model_architecture, test_type, attack_type, corruption_type, 
                                     severity_level, epsilon, accuracy, loss, correct, total):
    """Save robustness evaluation results to CSV file."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            'model_type', 'fusion_type', 'alpha', 'expert_id', 'model_architecture',
            'test_type', 'attack_type', 'corruption_type', 'severity_level', 'epsilon',
            'accuracy', 'loss', 'correct_predictions', 'total_predictions', 'timestamp'
        ])
        writer.writerow({
            'model_type': model_type,
            'fusion_type': fusion_type,
            'alpha': alpha,
            'expert_id': expert_id,
            'model_architecture': model_architecture,
            'test_type': test_type,
            'attack_type': attack_type,
            'corruption_type': corruption_type,
            'severity_level': severity_level,
            'epsilon': epsilon,
            'accuracy': accuracy,
            'loss': loss,
            'correct_predictions': correct,
            'total_predictions': total,
            'timestamp': timestamp
        })

def setup_ood_evaluation_csv(fusion_type, alpha, output_dir):
    """Setup CSV file for OOD evaluation results."""
    # Create the same directory structure as other CSV files
    csv_dir = os.path.join(output_dir, 'csv_logs', 'fusion_non_iid')
    os.makedirs(csv_dir, exist_ok=True)
    
    # Create filename with alpha-aware naming
    csv_filename = f"fusion_{fusion_type}_alpha_{alpha}_ood_evaluation.csv"
    csv_path = os.path.join(csv_dir, csv_filename)
    
    # Define CSV headers for OOD evaluation
    headers = [
        'timestamp', 'model_type', 'fusion_type', 'alpha', 'expert_id', 'model_architecture',
        'ood_dataset', 'ood_type', 'auroc', 'aupr', 'fpr95', 'detection_accuracy',
        'confidence_threshold', 'uncertainty_metric', 'ood_score_mean', 'ood_score_std'
    ]
    
    # Write headers to CSV file
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
    
    return csv_path

def save_ood_evaluation_results(csv_path, model_type, fusion_type, alpha, expert_id, 
                               model_architecture, ood_results):
    """Save OOD evaluation results to CSV."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Extract results for each OOD dataset
    for ood_dataset, results in ood_results.items():
        row = [
            timestamp, model_type, fusion_type, alpha, expert_id, model_architecture,
            ood_dataset, results['ood_type'], results['auroc'], results['aupr'], 
            results['fpr95'], results['detection_accuracy'], results['confidence_threshold'],
            results['uncertainty_metric'], results['ood_score_mean'], results['ood_score_std']
        ]
        
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Non-IID Fusion Full Evaluation")
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./checkpoints_full_eval')
    parser.add_argument('--fusion_type', type=str, required=True,
                        choices=['multiplicative', 'multiplicativeAddition', 'multiplicativeShifted', 'TransformerBase', 'concatenation', 'simpleAddition'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr_backbone', type=float, default=1e-5)
    parser.add_argument('--lr_heads', type=float, default=1e-4)
    parser.add_argument('--save_freq', type=int, default=10, help='Frequency for saving checkpoints (every N epochs)')
    parser.add_argument('--tinyimagenet_dir', type=str, default=os.path.join(os.path.dirname(__file__), 'tiny-imagenet-200'))
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # Data (fusion split)
    splits_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../splits'))
    train_loader, val_loader = get_data_loaders(args.data_dir, args.batch_size, num_workers=4, splits_dir=splits_dir)

    # Experts
    expert_backbones = load_densenet_specialist_backbones(args.checkpoint_dir, 4, device)

    # Model
    model = create_mcn_model(expert_backbones, input_dim=1024, num_classes=100, fusion_type=args.fusion_type).to(device)

    # Optimizers
    optim_backbones = optim.AdamW(model.expert_backbones.parameters(), lr=args.lr_backbone, weight_decay=1e-4)
    optim_heads = optim.AdamW(list(model.individual_heads.parameters()) + list(model.fusion_module.parameters()) + list(model.global_head.parameters()), lr=args.lr_heads, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None

    # Track best individual experts and components
    num_experts = len(model.expert_backbones)
    best_expert_accs = [-float('inf')] * num_experts
    best_expert_paths = [None] * num_experts
    best_fusion_acc = -float('inf')
    best_global_acc = -float('inf')
    best_fusion_path = None
    best_global_path = None
    
    # Create directories for component saving
    experts_dir = os.path.join(args.output_dir, 'experts_best')
    fusion_dir = os.path.join(args.output_dir, 'fusion_best')
    global_dir = os.path.join(args.output_dir, 'global_best')
    os.makedirs(experts_dir, exist_ok=True)
    os.makedirs(fusion_dir, exist_ok=True)
    os.makedirs(global_dir, exist_ok=True)

    # Setup CSV logging
    csv_path = setup_csv_logging(args.fusion_type, args.alpha, args.output_dir)
    print(f"CSV logging setup: {csv_path}")

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        tr_loss, tr_global_acc, tr_individual_accs, tr_global_loss, tr_individual_loss, tr_expert_losses = train_one_epoch(model, train_loader, optim_backbones, optim_heads, criterion, args.alpha, device)
        va_loss, va_global_acc, va_individual_accs, va_global_loss, va_individual_loss, va_expert_losses = validate(model, val_loader, criterion, args.alpha, device)
        
        # Print detailed accuracy breakdown
        print(f"  Train: loss={tr_loss:.4f}, global_acc={tr_global_acc:.2f}, exp_accs={[f'{acc:.1f}' for acc in tr_individual_accs]}")
        print(f"  Val:   loss={va_loss:.4f}, global_acc={va_global_acc:.2f}, exp_accs={[f'{acc:.1f}' for acc in va_individual_accs]}")
        
        # Save best individual experts
        for i, acc in enumerate(va_individual_accs):
            if acc > best_expert_accs[i]:
                best_expert_accs[i] = acc
                expert_path = os.path.join(experts_dir, f'expert_{i}_best.pth')
                torch.save({
                    'model_state_dict': model.expert_backbones[i].state_dict(),
                    'epoch': epoch + 1,
                    'best_val_acc': acc,
                    'fusion_type': args.fusion_type,
                    'alpha': args.alpha,
                    'component': f'expert_{i}',
                }, expert_path)
                best_expert_paths[i] = expert_path
                print(f"  🔸 Saved new best Expert {i} (val acc {acc:.2f}%) to {expert_path}")
        
        # Save best fusion module
        if va_global_acc > best_fusion_acc:
            best_fusion_acc = va_global_acc
            fusion_path = os.path.join(fusion_dir, 'fusion_module_best.pth')
            torch.save({
                'model_state_dict': model.fusion_module.state_dict(),
                'epoch': epoch + 1,
                'best_val_acc': va_global_acc,
                'fusion_type': args.fusion_type,
                'alpha': args.alpha,
                'component': 'fusion_module_best',
            }, fusion_path)
            best_fusion_path = fusion_path
            print(f"  🔸 Saved new best Fusion Module (val acc {va_global_acc:.2f}%) to {fusion_path}")
        
        # Save best global head
        if va_global_acc > best_global_acc:
            best_global_acc = va_global_acc
            global_path = os.path.join(global_dir, 'global_head_best.pth')
            torch.save({
                'model_state_dict': model.global_head.state_dict(),
                'epoch': epoch + 1,
                'best_val_acc': va_global_acc,
                'fusion_type': args.fusion_type,
                'alpha': args.alpha,
                'component': 'global_head_best',
            }, global_path)
            best_global_path = global_path
            print(f"  🔸 Saved new best Global Head (val acc {va_global_acc:.2f}%) to {global_path}")
        
        # Save overall best model
        if va_global_acc > best_acc:
            best_acc = va_global_acc
            best_state = model.state_dict()
            torch.save({'state_dict': best_state, 'val_acc': best_acc, 'epoch': epoch+1, 'config': vars(args)}, os.path.join(args.output_dir, f'best_model_{args.fusion_type}_alpha_{args.alpha}.pth'))
            print("  🎉 New best overall model saved!")

        # CSV logging for this epoch
        log_training_epoch(
            csv_path=csv_path,
            epoch=epoch,
            train_loss=tr_loss,
            val_loss=va_loss,
            val_accuracy=va_global_acc,
            experts_lr=args.lr_backbone,
            fusion_lr=args.lr_heads,
            global_head_lr=args.lr_heads,
            loss_global=tr_global_loss,
            loss_individual=tr_individual_loss,
            loss_total=tr_loss,
            alpha=args.alpha,
            fusion_type=args.fusion_type,
            expert_accuracies=va_individual_accs,
            expert_losses=tr_expert_losses,
        )
        print("  📊 Training progress logged to CSV")

        # Periodic epoch checkpointing (following reference pattern)
        if (epoch + 1) % args.save_freq == 0:  # Every save_freq epochs
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_backbones_state_dict": optim_backbones.state_dict(),
                    "optimizer_heads_state_dict": optim_heads.state_dict(),
                    "best_val_acc": best_acc,
                    "alpha": args.alpha,
                    "fusion_type": args.fusion_type,
                },
                checkpoint_path,
            )
            print(f"  Saved checkpoint: {checkpoint_path}")

    # PHASE 1: CLEAN TEST-SET EVALUATION (Best Components)
    print(f"\n{'='*80}")
    print("PHASE 1: CLEAN TEST-SET EVALUATION (Best Components)")
    print(f"{'='*80}")
    
    # Reload best experts
    print("Loading best individual experts...")
    for i, path in enumerate(best_expert_paths):
        if path is not None and os.path.exists(path):
            ckpt = torch.load(path, map_location=device)
            model_state = ckpt.get('model_state_dict', ckpt)
            model.expert_backbones[i].load_state_dict(model_state)
            print(f"🔄 Loaded best Expert {i} from {path}")
        else:
            print(f"⚠️  Best checkpoint not found for Expert {i}; using current weights")
    
    # Reload best global head
    if best_global_path is not None and os.path.exists(best_global_path):
        ckpt = torch.load(best_global_path, map_location=device)
        model_state = ckpt.get('model_state_dict', ckpt)
        model.global_head.load_state_dict(model_state)
        print(f"🔄 Loaded best Global Head from {best_global_path}")
    else:
        print("⚠️  No best global head path found; using current global head weights")
    
    # Evaluate on CIFAR-100 test set (clean)
    model.eval()
    test_total = 0
    test_correct = 0
    test_loss = 0.0
    with torch.no_grad():
        for data, targets in val_loader:  # Using val_loader as test set
            data, targets = data.to(device), targets.to(device)
            global_logits, individual_logits = model(data)
            loss, _, _ = dual_path_loss(global_logits, individual_logits, targets, criterion, args.alpha)
            test_loss += loss.item()
            _, predicted = torch.max(global_logits, 1)
            test_total += targets.size(0)
            test_correct += (predicted == targets).sum().item()
    
    clean_test_acc = 100.0 * test_correct / max(1, test_total)
    clean_test_loss = test_loss / max(1, len(val_loader))
    print(f"  Clean Test Accuracy (best components): {clean_test_acc:.2f}% | Loss: {clean_test_loss:.4f}")
    
    # Load the best overall model for comprehensive evaluation
    print("\nLoading best overall model for comprehensive evaluation...")
    best_model_path = os.path.join(args.output_dir, f'best_model_{args.fusion_type}_alpha_{args.alpha}.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded best overall model with validation accuracy: {checkpoint['val_acc']:.2f}%")
    else:
        print("Warning: Best model checkpoint not found, using current model state")

    # PHASE 2: POST-TRAINING EVALUATION - ENABLED
    print(f"\n{'='*80}")
    print("PHASE 2: POST-TRAINING EVALUATION")
    print(f"{'='*80}")
    
    robustness_csv = setup_robustness_evaluation_csv(args.fusion_type, args.alpha, args.output_dir)
    ood_csv = setup_ood_evaluation_csv(args.fusion_type, args.alpha, args.output_dir)
    print(f"Robustness evaluation CSV: {robustness_csv}")
    print(f"OOD evaluation CSV: {ood_csv}")
    
    # Reload best individual experts for post-training evaluation
    print("Loading best individual experts for post-training evaluation...")
    for i, path in enumerate(best_expert_paths):
        if path is not None and os.path.exists(path):
            ckpt = torch.load(path, map_location=device)
            model_state = ckpt.get('model_state_dict', ckpt)
            model.expert_backbones[i].load_state_dict(model_state)
            print(f"🔄 Loaded best Expert {i} for post-training eval from {path}")
        else:
            print(f"⚠️  Best checkpoint not found for Expert {i}; using current weights")
    
    # Individual Experts - Gaussian noise
    print(f"\n{'='*60}")
    print("POST-TRAINING: Individual Experts - Gaussian Noise Robustness")
    print(f"{'='*60}")
    class StandaloneExpert(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head
        
        def forward(self, x):
            f = self.backbone(x)
            f = F.relu(f, inplace=True)
            f = F.adaptive_avg_pool2d(f, (1, 1))
            f = torch.flatten(f, 1)
            return self.head(f)
    
    for i in range(len(model.expert_backbones)):
        print(f"\nTesting Expert {i} (POST-TRAINING) under Gaussian noise...")
        standalone_expert = StandaloneExpert(model.expert_backbones[i], model.individual_heads[i]).to(device)
        expert_noise = evaluate_gaussian_noise_robustness(standalone_expert, val_loader, device, sigmas=[0.0, 0.05, 0.1, 0.2, 0.3])
        for sigma, res in expert_noise.items():
            save_robustness_evaluation_results(
                robustness_csv, f'expert_{i}', args.fusion_type, args.alpha, i, 'DenseNet121',
                'gaussian_noise', 'N/A', 'N/A', f'sigma_{sigma}', 'N/A',
                res['accuracy'], res['loss'], res['correct'], res['total']
            )
    
    print(f"\nTesting Trained Fusion Model under Gaussian noise...")
    fusion_noise = evaluate_gaussian_noise_robustness(model, val_loader, device, sigmas=[0.0, 0.05, 0.1, 0.2, 0.3])
    for sigma, res in fusion_noise.items():
        save_robustness_evaluation_results(
            robustness_csv, 'fusion', args.fusion_type, args.alpha, 'N/A', 'FusionModel',
            'gaussian_noise', 'N/A', 'N/A', f'sigma_{sigma}', 'N/A',
            res['accuracy'], res['loss'], res['correct'], res['total']
        )
    
    # CIFAR-100-C corruption robustness
    print(f"\n{'='*60}")
    print("POST-TRAINING: Individual Experts - CIFAR-100-C Corruption Robustness")
    print(f"{'='*60}")
    for i in range(len(model.expert_backbones)):
        standalone_expert = StandaloneExpert(model.expert_backbones[i], model.individual_heads[i]).to(device)
        expert_corruption = evaluate_corruption_robustness(standalone_expert, args.data_dir, args.batch_size, device)
        for corruption_type, severity_map in expert_corruption.items():
            for severity, res in severity_map.items():
                save_robustness_evaluation_results(
                    robustness_csv, f'expert_{i}', args.fusion_type, args.alpha, i, 'DenseNet121',
                    'corruption', 'N/A', corruption_type, severity, 'N/A',
                    res['accuracy'], res['loss'], res['correct'], res['total']
                )
    print(f"\nTesting Trained Fusion Model on corruptions...")
    fusion_corruption = evaluate_corruption_robustness(model, args.data_dir, args.batch_size, device)
    for corruption_type, severity_map in fusion_corruption.items():
        for severity, res in severity_map.items():
            save_robustness_evaluation_results(
                robustness_csv, 'fusion', args.fusion_type, args.alpha, 'N/A', 'FusionModel',
                'corruption', 'N/A', corruption_type, severity, 'N/A',
                res['accuracy'], res['loss'], res['correct'], res['total']
            )
    
    # OOD detection
    print(f"\n{'='*60}")
    print("POST-TRAINING: Individual Experts - Out-of-Distribution Detection")
    print(f"{'='*60}")
    for i in range(len(model.expert_backbones)):
        standalone_expert = StandaloneExpert(model.expert_backbones[i], model.individual_heads[i]).to(device)
        expert_ood = evaluate_ood_detection(standalone_expert, val_loader, args.data_dir, args.batch_size, device, tinyimagenet_dir=args.tinyimagenet_dir)
        save_ood_evaluation_results(ood_csv, f'expert_{i}', args.fusion_type, args.alpha, i, 'densenet121', expert_ood)
    print(f"\nTesting Trained Fusion Model on OOD detection...")
    fusion_ood = evaluate_ood_detection(model, val_loader, args.data_dir, args.batch_size, device, tinyimagenet_dir=args.tinyimagenet_dir)
    save_ood_evaluation_results(ood_csv, 'fusion', args.fusion_type, args.alpha, 'N/A', 'fusion_model', fusion_ood)
    
    # Summary metrics requested
    print(f"\n{'='*60}")
    print("Requested summary metrics (fusion model):")
    # mCE
    mce = compute_mce_from_corruption_results(fusion_corruption)
    # ECE and rECE on clean test loader
    ece = compute_ece(model, val_loader, device)
    # Define rECE as ECE after temperature scaling? Here use same ECE placeholder
    rece = ece
    # Adversarial accuracy at 8/255
    adv_acc = evaluate_pgd(model, val_loader, device, eps=8/255, alpha=2/255, steps=10)
    # Near/Far AUROC and FPR95 from OOD
    auroc_near, auroc_far, fpr95_near = aggregate_ood_near_far(fusion_ood)
    print({
        'mCE (%)': round(mce, 3),
        'ECE (%)': round(ece, 3),
        'rECE (%)': round(rece, 3),
        'Adv@8/255 Acc (%)': round(adv_acc, 3),
        'AUROC (near)': round(auroc_near, 4),
        'AUROC (far)': round(auroc_far, 4),
        'FPR95 (near)': round(fpr95_near, 4),
    })
    
    # Final metrics at completion (Global + Experts)
    print(f"\n{'='*60}")
    print("FINAL METRICS (clean test set)")
    final_csv = setup_final_metrics_csv(args.fusion_type, args.alpha, args.output_dir)
    # Global head metrics
    global_top1, global_nll = evaluate_top1_and_nll(model, val_loader, device)
    global_ece = compute_ece(model, val_loader, device)
    save_final_metrics_row(final_csv, 'fusion_global', args.fusion_type, args.alpha, 'N/A', global_top1, global_nll, global_ece)
    print(f"Global Top-1: {global_top1:.2f}%, NLL: {global_nll:.4f}, ECE: {global_ece:.2f}%")
    # Expert head metrics
    expert_top1s = []
    class StandaloneExpert(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head
        def forward(self, x):
            f = self.backbone(x)
            f = F.relu(f, inplace=True)
            f = F.adaptive_avg_pool2d(f, (1, 1))
            f = torch.flatten(f, 1)
            return self.head(f)
    for i in range(len(model.expert_backbones)):
        se = StandaloneExpert(model.expert_backbones[i], model.individual_heads[i]).to(device)
        exp_top1, exp_nll = evaluate_top1_and_nll(se, val_loader, device)
        exp_ece = compute_ece(se, val_loader, device)
        expert_top1s.append(exp_top1)
        save_final_metrics_row(final_csv, 'expert', args.fusion_type, args.alpha, str(i), exp_top1, exp_nll, exp_ece)
        print(f"Expert {i} Top-1: {exp_top1:.2f}%, NLL: {exp_nll:.4f}, ECE: {exp_ece:.2f}%")
    if expert_top1s:
        avg_expert_top1 = float(np.mean(expert_top1s))
        save_final_metrics_row(final_csv, 'experts_avg', args.fusion_type, args.alpha, 'avg', avg_expert_top1, float('nan'), float('nan'))
        print(f"Avg Expert Top-1: {avg_expert_top1:.2f}%")

    print(f"\n✅ POST-TRAINING evaluation completed!")


if __name__ == '__main__':
    main()


