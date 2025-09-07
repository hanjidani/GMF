#!/usr/bin/env python3
"""
Full Dataset Benchmark Training Script
Trains the same model on the entire CIFAR-100 dataset for benchmarking against IID experts
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import wandb
import numpy as np
import os
import csv
import json
import argparse
from datetime import datetime
import pandas as pd

# Add model and config paths
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'configs'))

from improved_wide_resnet import improved_wideresnet28_10, improved_wideresnet40_2
from resnext_cifar import resnext29_8x64d, resnext29_16x64d, preact_resnext29_8x64d
from resnet_cifar import resnet18, resnet34, resnet50, preact_resnet18
from densenet_cifar import densenet121, densenet169, densenet201, efficient_densenet
from training_config import MODEL_CONFIGS, TRAINING_CONFIG, DATA_CONFIG, LOGGING_CONFIG
from augmentation_strategies import get_model_augmentation, get_test_transform


def get_model(model_name, num_classes=100, drop_rate=0.3):
    """Get model by name"""
    models = {
        # WideResNet variants
        'wideresnet28_10': lambda: improved_wideresnet28_10(num_classes, drop_rate),
        'wideresnet40_2': lambda: improved_wideresnet40_2(num_classes, drop_rate),
        
        # ResNeXt variants
        'resnext29_8x64d': lambda: resnext29_8x64d(num_classes, drop_rate),
        'resnext29_16x64d': lambda: resnext29_16x64d(num_classes, drop_rate),
        'preact_resnext29_8x64d': lambda: preact_resnext29_8x64d(num_classes, drop_rate),
        
        # ResNet variants
        'resnet18': lambda: resnet18(num_classes, drop_rate),
        'resnet34': lambda: resnet34(num_classes, drop_rate),
        'resnet50': lambda: resnet50(num_classes, drop_rate),
        'preact_resnet18': lambda: preact_resnet18(num_classes, drop_rate),
        
        # DenseNet variants
        'densenet121': lambda: densenet121(num_classes, drop_rate),
        'densenet169': lambda: densenet169(num_classes, drop_rate),
        'densenet201': lambda: densenet201(num_classes, drop_rate),
        'efficient_densenet': lambda: efficient_densenet(num_classes, drop_rate),
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not supported. Available: {list(models.keys())}")
    
    return models[model_name]()


# ====================================================
# Augmentation Functions (same as IID experts)
# ====================================================
def mixup_data(x, y, alpha=1.0, device='cuda'):
    """Returns mixed inputs, pairs of targets, and lambda"""
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

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0, device='cuda'):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


# ====================================================
# Full Dataset Training Function
# ====================================================
def train_full_dataset_benchmark(model_name, config, device, rank=None, world_size=None, distributed=False):
    """Train model on the entire CIFAR-100 dataset for benchmarking"""
    
    # Get model-specific config and augmentation
    model_config = MODEL_CONFIGS.get(model_name, {})
    aug_config = get_model_augmentation(model_name)
    effective_config = config.copy()
    
    # Override with model-specific settings
    for key in ['batch_size', 'drop_rate', 'lr', 'weight_decay', 'optimal_epochs']:
        if key in model_config:
            effective_config[key] = model_config[key]
            if key == 'lr':
                effective_config['learning_rate'] = model_config[key]
            elif key == 'optimal_epochs':
                effective_config['epochs'] = 250
    
    # Override with augmentation-specific settings
    for key in ['mixup_alpha', 'cutmix_alpha', 'augmentation_mode', 'label_smoothing']:
        if key in aug_config:
            effective_config[key] = aug_config[key]
    
    print(f"Training Full Dataset Benchmark with {model_name}:")
    print(f"  Architecture: {model_config.get('name', model_name)}")
    print(f"  Dataset: Full CIFAR-100 ({50000} training samples)")
    print(f"  Epochs: {effective_config.get('epochs', config['epochs'])}")
    print(f"  Batch size: {effective_config.get('batch_size', config['batch_size'])}")
    print(f"  Learning rate: {effective_config.get('learning_rate', config['learning_rate'])}")
    print(f"  LR Schedule: {config['scheduler_type']} (T_max: {effective_config.get('epochs', config['epochs'])})")
    print(f"  Augmentation: {aug_config['rationale']}")
    print(f"  Expected accuracy: {model_config.get('expected_accuracy', 'Unknown')}")
    print(f"  🎯 Target: ≥75% accuracy (no early stopping until reached)")
    
    # Setup data loaders with SOTA augmentation
    train_transform = aug_config['transform']()
    test_transform = get_test_transform()
    
    # Full training dataset
    full_trainset = torchvision.datasets.CIFAR100(
        root=DATA_CONFIG['data_root'], 
        train=True, 
        download=True, 
        transform=train_transform
    )
    
    if distributed:
        train_sampler = DistributedSampler(full_trainset, num_replicas=world_size, rank=rank)
        shuffle = False  # DistributedSampler handles shuffling
    else:
        train_sampler = None
        shuffle = True
    
    trainloader = DataLoader(
        full_trainset, 
        batch_size=effective_config.get('batch_size', config['batch_size']), 
        shuffle=shuffle, 
        sampler=train_sampler,
        num_workers=DATA_CONFIG['num_workers'],
        pin_memory=DATA_CONFIG['pin_memory']
    )
    
    # Test loader
    testset = torchvision.datasets.CIFAR100(
        root=DATA_CONFIG['data_root'], 
        train=False, 
        download=True, 
        transform=test_transform
    )
    testloader = DataLoader(
        testset, 
        batch_size=effective_config.get('batch_size', config['batch_size']), 
        shuffle=False, 
        num_workers=DATA_CONFIG['num_workers']
    )
    
    # Setup model
    model = get_model(model_name, drop_rate=effective_config['drop_rate']).to(device)
    
    if distributed:
        model = DDP(model, device_ids=[rank])
    
    # Setup optimizer and scheduler
    optimizer = optim.SGD(
        model.parameters(), 
        lr=effective_config.get('learning_rate', config['learning_rate']),
        momentum=config['momentum'],
        weight_decay=effective_config.get('weight_decay', config['weight_decay']),
        nesterov=config['nesterov']
    )
    
    if config['scheduler_type'] == 'cosine_warm_restart':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config['cosine_t0'], T_mult=config['cosine_t_mult']
        )
    else:
        # Use simple cosine annealing with proper T_max
        total_epochs_for_scheduler = effective_config.get('epochs', config['epochs'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs_for_scheduler)   
    
    criterion = nn.CrossEntropyLoss(label_smoothing=effective_config.get('label_smoothing', config['label_smoothing']))
    
    # Setup logging
    wandb.init(
        project="MCN_Full_Dataset_Benchmark", 
        name=f"full_dataset_{model_name}_benchmark",
        config={
            'experiment_type': 'full_dataset_benchmark',
            'model_name': model_name,
            'train_samples': len(full_trainset),
            'augmentation_strategy': aug_config['rationale'],
            **effective_config,
            **model_config
        }
    )
    
    # Training loop setup
    best_acc = 0.0
    best_train_acc = 0.0
    best_epoch = -1
    epochs_no_improve = 0
    
    checkpoint_dir = f"{DATA_CONFIG['checkpoint_dir']}_full_dataset_benchmark_250"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # CSV logging setup
    csv_dir = os.path.join(checkpoint_dir, "per_epoch_logs")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"full_dataset_{model_name}_benchmark_250_epochs.csv")
    
    # Initialize CSV with headers
    csv_columns = ['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'lr', 'best_test_acc_so_far']
    epoch_data = []
    
    total_epochs = effective_config.get('epochs', config['epochs'])
    
    # Enhanced early stopping criteria
    min_accuracy_threshold = 75.0  # Minimum 75% accuracy requirement
    patience_after_threshold = config['patience']  # Original patience after reaching threshold
    
    for epoch in range(total_epochs):
        # Set epoch for distributed sampler
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
            
        # Training phase
        model.train()
        total_loss, correct, total = 0, 0, 0
        pbar = tqdm(trainloader, desc=f"[Full Dataset Benchmark] Epoch {epoch+1}/{total_epochs}")
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Apply augmentation based on strategy
            augmentation_mode = effective_config.get('augmentation_mode', 'cutmix')
            if augmentation_mode == 'mixup':
                inputs, targets_a, targets_b, lam = mixup_data(
                    inputs, targets, effective_config.get('mixup_alpha', 0.2), device
                )
                optimizer.zero_grad()
                _, outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            elif augmentation_mode == 'cutmix':
                inputs, targets_a, targets_b, lam = cutmix_data(
                    inputs, targets, effective_config.get('cutmix_alpha', 1.0), device
                )
                optimizer.zero_grad()
                _, outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                optimizer.zero_grad()
                _, outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            loss.backward()
            
            # Gradient clipping
            if config['gradient_clip_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
            
            optimizer.step()
            
            total_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            pbar.set_postfix(loss=loss.item(), acc=100.*correct/total)
        
        train_acc = 100. * correct / total
        train_loss = total_loss / total
        scheduler.step()
        
        if train_acc > best_train_acc:
            best_train_acc = train_acc
        
        # Evaluation phase
        model.eval()
        test_loss, test_correct, test_total = 0, 0, 0
        
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                features, logits = model(inputs)
                loss = criterion(logits, targets)
                
                test_loss += loss.item() * targets.size(0)
                _, predicted = logits.max(1)
                test_correct += predicted.eq(targets).sum().item()
                test_total += targets.size(0)
        
        test_acc = 100. * test_correct / test_total
        test_loss /= test_total
        
        print(f"✅ Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Update best accuracy tracking
        current_best_acc = max(best_acc, test_acc)
        
        # Only log and save on main process (rank 0) or when not distributed
        if not distributed or rank == 0:
            # CSV logging - record every epoch
            epoch_record = {
                'epoch': epoch + 1,
                'train_loss': round(train_loss, 4),
                'train_acc': round(train_acc, 2),
                'test_loss': round(test_loss, 4),
                'test_acc': round(test_acc, 2),
                'lr': round(scheduler.get_last_lr()[0], 6),
                'best_test_acc_so_far': round(current_best_acc, 2)
            }
            epoch_data.append(epoch_record)
            
            # Save CSV every 10 epochs and at the end
            if (epoch + 1) % 10 == 0 or epoch == total_epochs - 1:
                df = pd.DataFrame(epoch_data)
                df.to_csv(csv_path, index=False)
            
            # Wandb logging
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
                # Save the underlying model state dict for DDP
                if distributed:
                    torch.save(
                        model.module.state_dict(), 
                        os.path.join(checkpoint_dir, f"best_full_dataset_{model_name}_benchmark_250.pth")
                    )
                else:
                    torch.save(
                        model.state_dict(), 
                        os.path.join(checkpoint_dir, f"best_full_dataset_{model_name}_benchmark_250.pth")
                    )
            else:
                epochs_no_improve += 1
        
        # Enhanced early stopping logic - NO early stopping until 75% test accuracy
        if epoch + 1 >= config['min_epochs']:
            # NEVER stop early if test accuracy < 75% - keep training!
            if best_acc < min_accuracy_threshold:
                # Only stop if we've trained for a very long time without any progress
                if epochs_no_improve >= patience_after_threshold * 4:  # Very extended patience (120 epochs)
                    print(f"⚠️  Early stopping at epoch {epoch+1}: accuracy {best_acc:.2f}% < {min_accuracy_threshold}% threshold after very extended patience ({patience_after_threshold * 4} epochs).")
                    break
                # Otherwise, keep training regardless of patience
                elif epochs_no_improve >= patience_after_threshold:
                    print(f"📈 Continuing training despite {epochs_no_improve} epochs without improvement - target is {min_accuracy_threshold}% accuracy (current: {best_acc:.2f}%)")
            else:
                # Only after reaching 75% accuracy, use normal early stopping
                if epochs_no_improve >= patience_after_threshold:
                    print(f"✅ Early stopping at epoch {epoch+1}: reached {best_acc:.2f}% accuracy (≥{min_accuracy_threshold}%) with no improvement for {patience_after_threshold} epochs.")
                    break
    
    # Only save and log on main process (rank 0) or when not distributed
    if not distributed or rank == 0:
        # Final CSV save
        df = pd.DataFrame(epoch_data)
        df.to_csv(csv_path, index=False)
        
        # Final summary
        print(f"🏁 Full Dataset Benchmark ({model_name}) finished.")
        print(f"📈 Best Test Accuracy: {best_acc:.2f}% at Epoch {best_epoch}")
        print(f"🧠 Best Train Accuracy: {best_train_acc:.2f}%")
        print(f"📊 Per-epoch log saved to: {csv_path}")
        
        # Check if accuracy threshold was met
        if best_acc >= min_accuracy_threshold:
            print(f"✅ Accuracy threshold met: {best_acc:.2f}% ≥ {min_accuracy_threshold}%")
        else:
            print(f"⚠️  Accuracy threshold not met: {best_acc:.2f}% < {min_accuracy_threshold}%")
        
        wandb.log({
            "best_test_acc": best_acc,
            "best_train_acc": best_train_acc,
            "best_epoch": best_epoch,
            "threshold_met": best_acc >= min_accuracy_threshold
        })
        
        wandb.finish()
    
    return {
        'model_name': model_name,
        'best_test_acc': best_acc,
        'best_train_acc': best_train_acc,
        'best_epoch': best_epoch,
        'train_samples': len(full_trainset),
        'threshold_met': best_acc >= min_accuracy_threshold,
        'csv_log_path': csv_path,
        'total_epochs_trained': epoch + 1
    }


def train_distributed(model_name, config, world_size):
    """Distributed training entry point"""
    mp.spawn(
        _train_distributed_worker,
        args=(model_name, config, world_size),
        nprocs=world_size,
        join=True
    )


def _train_distributed_worker(rank, model_name, config, world_size):
    """Worker function for distributed training"""
    # Set up distributed environment
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    
    # Set device for this process
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    # Train the model
    result = train_full_dataset_benchmark(
        model_name, config, device, 
        rank=rank, world_size=world_size, distributed=True
    )
    
    # Clean up distributed process group
    dist.destroy_process_group()
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Train model on full CIFAR-100 dataset for benchmarking')
    parser.add_argument('--model', type=str, default='wideresnet28_10',
                        choices=['wideresnet28_10', 'wideresnet40_2', 
                                'resnext29_8x64d', 'resnext29_16x64d', 'preact_resnext29_8x64d',
                                'resnet18', 'resnet34', 'resnet50', 'preact_resnet18',
                                'densenet121', 'densenet169', 'densenet201', 'efficient_densenet'],
                        help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')
    parser.add_argument('--distributed', action='store_true',
                        help='Enable distributed training across multiple GPUs')
    parser.add_argument('--world_size', type=int, default=4,
                        help='Number of GPUs for distributed training')
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    config = TRAINING_CONFIG.copy()
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.lr is not None:
        config['learning_rate'] = args.lr
    
    # Disable wandb if requested
    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    
    if args.distributed:
        print(f"Using distributed training with {args.world_size} GPUs")
        print(f"Training full dataset benchmark with model: {args.model}")
        
        # Train on full dataset with distributed training
        print(f"\n{'='*60}")
        print(f"Full Dataset Benchmark Training (Distributed)")
        print(f"{'='*60}")
        
        result = train_distributed(args.model, config, args.world_size)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        print(f"Training full dataset benchmark with model: {args.model}")
        
        # Train on full dataset
        print(f"\n{'='*60}")
        print(f"Full Dataset Benchmark Training")
        print(f"{'='*60}")
        
        result = train_full_dataset_benchmark(args.model, config, device)
    
    # Save results summary
    results_path = os.path.join(f"{DATA_CONFIG['checkpoint_dir']}_full_dataset_benchmark_250", f"full_dataset_benchmark_250_results_{args.model}.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Full Dataset Benchmark Summary")
    print(f"{'='*60}")
    threshold_status = "✅" if result['threshold_met'] else "⚠️"
    print(f"Model: {result['model_name']}")
    print(f"Best Test Accuracy: {result['best_test_acc']:.2f}% {threshold_status}")
    print(f"Best Train Accuracy: {result['best_train_acc']:.2f}%")
    print(f"Best Epoch: {result['best_epoch']}")
    print(f"Total Epochs Trained: {result['total_epochs_trained']}")
    print(f"Training Samples: {result['train_samples']}")
    print(f"75%+ Threshold Met: {result['threshold_met']}")
    print(f"📊 CSV log: {result['csv_log_path']}")
    print(f"📁 Results saved to: {results_path}")
    
    csv_logs_dir = os.path.join(f"{DATA_CONFIG['checkpoint_dir']}_full_dataset_benchmark_250", 'per_epoch_logs')
    print(f"\n📊 Per-epoch CSV log saved in: {csv_logs_dir}")


if __name__ == "__main__":
    main()
