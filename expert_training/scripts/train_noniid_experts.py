#!/usr/bin/env python3
"""
Non-IID Expert Training Script
Based on train_iid_experts.py but calls train_noniid.py for class-based distribution
Each expert sees only 25 classes (labels 0-24) with 1x25 output
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import json
import argparse

# Add model and config paths
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'configs'))

from training_config import TRAINING_CONFIG, DATA_CONFIG
from augmentation_strategies import get_model_augmentation
from train_noniid import train_noniid_expert, get_class_based_indices_from_existing


def main():
    parser = argparse.ArgumentParser(description='Train Non-IID experts with class-based distribution')
    parser.add_argument('--model', type=str, default='wideresnet28_10',
                        choices=['wideresnet28_10', 'wideresnet40_2', 
                                'resnext29_8x64d', 'resnext29_16x64d', 'preact_resnext29_8x64d',
                                'resnet18', 'resnet34', 'resnet50', 'preact_resnet18',
                                'densenet121', 'densenet169', 'densenet201', 'efficient_densenet'],
                        help='Model architecture to use')
    parser.add_argument('--expert_id', type=int, default=None,
                        help='Train only specific expert (0-3), or all if None')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    config = TRAINING_CONFIG.copy()
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.lr is not None:
        config['learning_rate'] = args.lr
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training Non-IID experts with model: {args.model}")
    print(f"Class-based distribution: Each expert sees 25 classes (0-24, 25-49, 50-74, 75-99)")
    
    # Disable wandb if requested
    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    
    # Load data with SOTA augmentation
    aug_config = get_model_augmentation(args.model)
    train_transform = aug_config['transform']()
    
    full_trainset = torchvision.datasets.CIFAR100(
        root=DATA_CONFIG['data_root'], 
        train=True, 
        download=True, 
        transform=train_transform
    )
    
    # Load existing expert indices from splits directory
    print(f"\nLoading existing expert indices from splits directory...")
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    expert_indices_path = os.path.join(project_root, DATA_CONFIG['splits_dir'], "expert_train_indices.npy")
    if not os.path.exists(expert_indices_path):
        raise FileNotFoundError(f"Expert indices file not found at: {expert_indices_path}")
    
    master_expert_indices = np.load(expert_indices_path, allow_pickle=True)
    
    # Create class-based non-IID splits from existing expert indices
    print(f"Creating class-based non-IID distribution from existing splits...")
    expert_indices_list = get_class_based_indices_from_existing(
        full_trainset, 
        master_expert_indices,
        num_classes_per_expert=25, 
        num_experts=4, 
        seed=config['seed']
    )
    
    print(f"\nNon-IID Class Distribution:")
    for i, indices in enumerate(expert_indices_list):
        start_class = i * 25
        end_class = start_class + 25
        print(f"  Expert {i}: Classes {start_class}-{end_class-1} ({len(indices)} samples)")
    
    # Determine which experts to train
    if args.expert_id is not None:
        if 0 <= args.expert_id < 4:
            expert_range = [args.expert_id]
        else:
            raise ValueError("Expert ID must be between 0 and 3")
    else:
        expert_range = list(range(4))
    
    # Train experts
    results = []
    for expert_id in expert_range:
        print(f"\n{'='*60}")
        print(f"Training Non-IID Expert {expert_id} with {args.model}")
        print(f"Classes: {expert_id * 25}-{(expert_id + 1) * 25 - 1}")
        print(f"{'='*60}")
        
        expert_subset = Subset(full_trainset, expert_indices_list[expert_id])
        result = train_noniid_expert(expert_subset, expert_id, args.model, config, device)
        results.append(result)
    
    # Save results summary
    results_path = os.path.join(f"{DATA_CONFIG['checkpoint_dir']}_noniid", f"noniid_training_results_{args.model}.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Non-IID Training Summary")
    print(f"{'='*60}")
    threshold_met_count = 0
    for result in results:
        threshold_status = "✅" if result['threshold_met'] else "⚠️"
        print(f"Expert {result['expert_id']} ({result['classes']}): {result['best_test_acc']:.2f}% accuracy {threshold_status} (epochs: {result['total_epochs_trained']})")
        print(f"  📊 CSV log: {result['csv_log_path']}")
        if result['threshold_met']:
            threshold_met_count += 1
    
    if len(results) > 1:
        avg_acc = sum(r['best_test_acc'] for r in results) / len(results)
        print(f"\nAverage accuracy: {avg_acc:.2f}%")
        print(f"Experts meeting 75%+ threshold: {threshold_met_count}/{len(results)}")
    
    csv_logs_dir = os.path.join(f"{DATA_CONFIG['checkpoint_dir']}_noniid", 'per_epoch_logs')
    print(f"\n📊 All per-epoch CSV logs saved in: {csv_logs_dir}")
    print(f"💾 Results summary saved to: {results_path}")


if __name__ == "__main__":
    main()
