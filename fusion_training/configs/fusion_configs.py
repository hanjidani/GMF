"""
Configuration presets for different fusion models with optimized learning rates
"""

import torch
import torch.nn as nn

# Base configurations for different fusion types
MULTIPLICATIVE_CONFIG = {
    'base_lr': 1e-4,        # Increased from 1e-5 for better expert adaptation
    'head_lr': 5e-4,        # Moderate fusion learning
    'weight_decay': 1e-4,
    'lambda_loss': 1.0,      # Balance between global and individual paths
    'rationale': 'Simple multiplicative fusion with balanced expert-fusion learning'
}

MULTIPLICATIVE_ADDITION_CONFIG = {
    'base_lr': 1e-4,        # Increased for expert knowledge transfer
    'head_lr': 1e-3,        # Standard MLP learning rate
    'weight_decay': 1e-4,
    'lambda_loss': 1.0,
    'rationale': 'MLP-based fusion with enhanced expert adaptation'
}

TRANSFORMER_BASE_CONFIG = {
    'base_lr': 5e-5,        # Moderate expert learning for complex fusion
    'head_lr': 2e-4,        # Lower LR for attention stability
    'weight_decay': 5e-4,   # Higher regularization for transformer
    'lambda_loss': 1.0,
    'rationale': 'Attention-based fusion with careful expert learning'
}

CONCATENATION_CONFIG = {
    'base_lr': 1e-4,        # Increased for expert adaptation
    'head_lr': 1e-3,        # Standard MLP learning rate
    'weight_decay': 1e-4,
    'lambda_loss': 1.0,
    'rationale': 'Feature concatenation with balanced learning'
}

SIMPLE_ADDITION_CONFIG = {
    'base_lr': 1e-4,        # Increased for expert knowledge transfer
    'head_lr': 1e-3,        # Standard MLP learning rate
    'weight_decay': 1e-4,
    'lambda_loss': 1.0,
    'rationale': 'Direct feature addition with MLP processing for nonlinearity'
}

# All fusion configurations
FUSION_CONFIGS = {
    'multiplicative': MULTIPLICATIVE_CONFIG,
    'multiplicativeAddition': MULTIPLICATIVE_ADDITION_CONFIG,
    'TransformerBase': TRANSFORMER_BASE_CONFIG,
    'concatenation': CONCATENATION_CONFIG,
    'simpleAddition': SIMPLE_ADDITION_CONFIG,
}

def get_fusion_config(fusion_type):
    """Get configuration for a specific fusion type."""
    if fusion_type not in FUSION_CONFIGS:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
    return FUSION_CONFIGS[fusion_type].copy()

def get_optimal_learning_rates(fusion_type, input_dim):
    """
    Get optimal learning rates based on fusion type and input dimension.
    Optimized for knowledge transfer and dual-path learning.
    """
    config = get_fusion_config(fusion_type)
    
    # Base learning rates for dual-path architecture
    base_lr = config['base_lr']      # Expert backbone learning rate
    head_lr = config['head_lr']      # Fusion and global head learning rate
    
    # Adaptive learning rate based on input dimension
    if input_dim > 1000:
        # Large feature dimensions: reduce learning rates for stability
        base_lr *= 0.7
        head_lr *= 0.8
        print(f"Large input_dim ({input_dim}): Reducing LRs for stability")
    elif input_dim < 100:
        # Small feature dimensions: increase learning rates for faster convergence
        base_lr *= 1.3
        head_lr *= 1.2
        print(f"Small input_dim ({input_dim}): Increasing LRs for faster convergence")
    
    # Fusion-specific adaptive adjustments
    if fusion_type == 'TransformerBase':
        # Transformer needs careful tuning
        if input_dim > 500:
            base_lr *= 0.8  # More conservative for large features
    elif fusion_type == 'multiplicative':
        # Simple fusion can handle higher rates
        if input_dim < 500:
            base_lr *= 1.1  # Slightly higher for small features
    
    return {
        'base_lr': base_lr,
        'head_lr': head_lr,
        'input_dim': input_dim,
        'fusion_type': fusion_type,
        'rationale': config['rationale']
    }

def get_adaptive_scheduler_config(fusion_type, input_dim, total_epochs):
    """
    Get adaptive scheduler configuration for different components.
    Enables progressive learning rate adjustment during training.
    """
    lr_config = get_optimal_learning_rates(fusion_type, input_dim)
    
    # Different scheduler strategies for different components
    scheduler_config = {
        'experts': {
            'type': 'CosineAnnealingWarmRestarts',
            'T_0': total_epochs // 4,  # Restart every quarter
            'T_mult': 2,               # Double restart interval
            'eta_min': lr_config['base_lr'] * 0.1,  # Minimum LR
            'rationale': 'Warm restarts help experts adapt to fusion changes'
        },
        'fusion': {
            'type': 'CosineAnnealingLR',
            'T_max': total_epochs,
            'eta_min': lr_config['head_lr'] * 0.1,
            'rationale': 'Smooth decay for fusion stability'
        },
        'global_head': {
            'type': 'CosineAnnealingLR',
            'T_max': total_epochs,
            'eta_min': lr_config['head_lr'] * 0.1,
            'rationale': 'Consistent with fusion learning'
        }
    }
    
    return scheduler_config

def get_knowledge_transfer_config(fusion_type):
    """
    Get knowledge transfer specific configurations.
    Optimizes the dual-path architecture for mutual learning.
    """
    config = get_fusion_config(fusion_type)
    
    # Knowledge transfer specific settings
    kt_config = {
        'lambda_loss': config['lambda_loss'],
        'expert_gradient_accumulation': 2,  # Accumulate gradients for stable expert updates
        'fusion_gradient_clipping': 1.0,    # Clip fusion gradients for stability
        'expert_gradient_clipping': 5.0,    # Allow larger expert gradient updates
        'knowledge_distillation_weight': 0.1,  # Soft target learning between experts
        'cross_expert_attention': fusion_type == 'TransformerBase',  # Enable for transformer
        'rationale': 'Enhanced knowledge transfer between individual and collaborative paths'
    }
    
    return kt_config
