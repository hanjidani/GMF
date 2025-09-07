import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os

def get_data_loaders(data_dir, batch_size, num_workers=4, splits_dir: str = None):
    """
    Creates and returns the CIFAR-100 train and validation DataLoaders
    with standard, simple augmentations.

    If splits_dir is provided, use fusion_holdout_indices.npy from that dir to
    build the training subset (fusion-only training data).
    """
    # Simple augmentations for the training set
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    # Normalization only for the validation set
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=val_transform)

    if splits_dir is not None:
        split_path = os.path.join(splits_dir, 'fusion_holdout_indices.npy')
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Fusion split indices not found: {split_path}")
        fusion_indices = np.load(split_path)
        train_dataset = Subset(train_dataset, fusion_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


