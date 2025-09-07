"""
Training configuration for expert models
"""

# ====================================================
# Model Configuration
# ====================================================
MODEL_CONFIGS = {
    # WideResNet variants
    'wideresnet28_10': {
        'name': 'WideResNet-28-10',
        'expected_accuracy': '84-87%',
        'parameters': '36.5M',
        'drop_rate': 0.3,
        'batch_size': 128,
        'lr': 0.1,
        'weight_decay': 5e-4,
        'optimal_epochs': 250,
        'convergence_time': '~150-200 epochs',
    },
    'wideresnet40_2': {
        'name': 'WideResNet-40-2', 
        'expected_accuracy': '83-86%',
        'parameters': '2.2M',
        'drop_rate': 0.3,
        'batch_size': 128,
        'lr': 0.1,
        'weight_decay': 5e-4,
        'optimal_epochs': 250,
        'convergence_time': '~200-250 epochs',
    },
    
    # ResNeXt variants
    'resnext29_8x64d': {
        'name': 'ResNeXt-29-8x64d',
        'expected_accuracy': '85-88%',
        'parameters': '34.4M',
        'drop_rate': 0.3,
        'batch_size': 128,
        'lr': 0.1,
        'weight_decay': 5e-4,
        'optimal_epochs': 250,
        'convergence_time': '~150-200 epochs',
    },
    'preact_resnext29_8x64d': {
        'name': 'PreAct-ResNeXt-29-8x64d',
        'expected_accuracy': '86-89%',
        'parameters': '34.4M',
        'drop_rate': 0.3,
        'batch_size': 128,
        'lr': 0.1,
        'weight_decay': 5e-4,
        'optimal_epochs': 250,
        'convergence_time': '~150-200 epochs',
    },
    
    # ResNet variants - Lightweight and fast
    'resnet18': {
        'name': 'ResNet-18',
        'expected_accuracy': '78-82%',
        'parameters': '11.2M',
        'drop_rate': 0.0,
        'batch_size': 128,
        'lr': 0.1,  # Higher LR for lighter model
        'weight_decay': 1e-4,  # Reduced weight decay
        'optimal_epochs': 250,
        'convergence_time': '~180-220 epochs',
    },
    'resnet34': {
        'name': 'ResNet-34',
        'expected_accuracy': '80-84%',
        'parameters': '21.3M',
        'drop_rate': 0.0,
        'batch_size': 128,
        'lr': 0.1,
        'weight_decay': 1e-4,
        'optimal_epochs': 250,
        'convergence_time': '~200-230 epochs',
    },
    'resnet50': {
        'name': 'ResNet-50',
        'expected_accuracy': '82-86%',
        'parameters': '23.5M',
        'drop_rate': 0.1,
        'batch_size': 128,
        'lr': 0.1,
        'weight_decay': 1e-4,
        'optimal_epochs': 250,
        'convergence_time': '~200-240 epochs',
    },
    'preact_resnet18': {
        'name': 'PreAct-ResNet-18',
        'expected_accuracy': '79-83%',
        'parameters': '11.2M',
        'drop_rate': 0.0,
        'batch_size': 128,
        'lr': 0.2,
        'weight_decay': 1e-4,
        'optimal_epochs': 250,
        'convergence_time': '~180-220 epochs',
    },
    
    # DenseNet variants - Memory efficient
    'densenet121': {
        'name': 'DenseNet-121',
        'expected_accuracy': '80-84%',
        'parameters': '7.0M',
        'drop_rate': 0.2,  # DenseNet benefits from dropout
        'batch_size': 64,  # Smaller batch due to memory
        'lr': 0.1,
        'weight_decay': 1e-4,
        'optimal_epochs': 250,
        'convergence_time': '~200-230 epochs',
    },
    'densenet169': {
        'name': 'DenseNet-169',
        'expected_accuracy': '81-85%',
        'parameters': '12.5M',
        'drop_rate': 0.2,
        'batch_size': 64,
        'lr': 0.1,
        'weight_decay': 1e-4,
        'optimal_epochs': 250,
        'convergence_time': '~200-240 epochs',
    },
    'densenet201': {
        'name': 'DenseNet-201',
        'expected_accuracy': '82-86%',
        'parameters': '18.1M',
        'drop_rate': 0.2,
        'batch_size': 32,  # Even smaller batch
        'lr': 0.1,
        'weight_decay': 1e-4,
        'optimal_epochs': 250,
        'convergence_time': '~200-240 epochs',
    },
    'efficient_densenet': {
        'name': 'Efficient-DenseNet',
        'expected_accuracy': '79-83%',
        'parameters': '2.5M',
        'drop_rate': 0.1,
        'batch_size': 128,
        'lr': 0.1,
        'weight_decay': 1e-4,
        'optimal_epochs': 250,
        'convergence_time': '~200-240 epochs',
    }
}

# ====================================================
# Training Hyperparameters
# ====================================================
TRAINING_CONFIG = {
    'epochs': 250,  # Increased to 250 for fair comparison
    'batch_size': 128,
    'learning_rate': 0.1,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'nesterov': True,
    'patience': 30,
    'min_epochs': 120,
    'scheduler_type': 'cosine',
    'cosine_t0': 150,
    'cosine_t_mult': 2,  # Added: multiplier for cosine warm restart
    'label_smoothing': 0.1,
    'grad_clip': 1.0,
    'gradient_clip_norm': 1.0,  # Added: alias for grad_clip (script compatibility)
    'amp': True,
    'num_workers': 4,
    'pin_memory': True,
    'mixed_precision': True,
    
    # Data Augmentation Parameters
    'mixup_alpha': 0.2,        # Added: mixup alpha parameter
    'cutmix_alpha': 1.0,       # Added: cutmix alpha parameter
    'augmentation_mode': 'cutmix',  # Added: default augmentation mode
    
    # IID Data Distribution Parameters
    'shared_ratio': 0.40,      # 40% of data shared between all experts
    'unique_ratio': 0.15,      # 15% unique data per expert
    'seed': 42,                # Random seed for reproducibility
}

# ====================================================
# Data Configuration
# ====================================================
DATA_CONFIG = {
    # CIFAR-100 normalization values
    'mean': (0.5071, 0.4865, 0.4409),
    'std': (0.2673, 0.2564, 0.2762),
    
    # Data paths
    'data_root': './data',
    'checkpoint_dir': './checkpoints_expert',
    'splits_dir': './splits',
    
    # Data loading
    'num_workers': 4,
    'pin_memory': True,
}

# ====================================================
# Logging Configuration  
# ====================================================
LOGGING_CONFIG = {
    'wandb_project': 'PuGA_Expert_Training',
    'log_interval': 100,
    'save_best_only': True,
    'save_last': True,
}

# ====================================================
# Hardware Configuration
# ====================================================
HARDWARE_CONFIG = {
    'device': 'cuda',
    'mixed_precision': True,
    'compile_model': False,  # Set to True for PyTorch 2.0+
}
