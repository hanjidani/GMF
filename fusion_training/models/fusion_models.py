import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleAdditionFusion(nn.Module):
    """
    Weighted addition fusion: weighted element-wise addition of features with hidden layer processing
    """
    def __init__(self, num_experts, input_dim, hidden_dim=None):
        super().__init__()
        # No normalization layers - keep features as-is
        self.num_experts = num_experts
        self.input_dim = input_dim
        
        if hidden_dim is None:
            hidden_dim = input_dim  # Same size for consistency with other methods
        else:
            hidden_dim = input_dim
        
        # Trainable weights for each expert
        self.expert_weights = nn.Parameter(torch.ones(num_experts))
        
        # MLP with same hidden dimension as input for nonlinearity
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, features_list):
        # Weighted element-wise addition of all features
        # Apply softmax to ensure weights sum to 1 and are positive
        weights = F.softmax(self.expert_weights, dim=0)
        
        # Weighted sum of features
        fused = weights[0] * features_list[0]
        for i in range(1, len(features_list)):
            fused = fused + weights[i] * features_list[i]
        
        # Process through MLP for nonlinearity
        fused = self.mlp(fused)
        
        return fused


class MultiplicativeFusion(nn.Module):
    """
    Multiplicative fusion: element-wise multiplication of normalized features with hidden layer processing
    """
    def __init__(self, num_experts, input_dim, hidden_dim=None):
        super().__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        
        if hidden_dim is None:
            hidden_dim = input_dim  # Same size for consistency with other methods
        else:
            hidden_dim = input_dim
        
        self.norms = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_experts)])
        
        # MLP with same hidden dimension as input for nonlinearity
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, features_list):
        normalized = [norm(f) for norm, f in zip(self.norms, features_list)]
        fused = torch.stack(normalized, dim=0).prod(dim=0)
        
        # Process through MLP for nonlinearity
        fused = self.mlp(fused)
        
        return fused


class MultiplicativeAdditionFusion(nn.Module):
    """
    Multiplicative + Weighted Addition fusion: combines both approaches for better nonlinearity
    """
    def __init__(self, num_experts, input_dim, hidden_dim=None):
        super().__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        
        if hidden_dim is None:
            hidden_dim = input_dim  # Same size for consistency with other methods
        else:
            hidden_dim = input_dim
        
        self.norms = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_experts)])
        
        # Trainable weights for each expert in addition
        self.expert_weights = nn.Parameter(torch.ones(num_experts))
        
        # MLP with same hidden dimension as input for nonlinearity
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, features_list):
        normalized = [norm(f) for norm, f in zip(self.norms, features_list)]
        
        # Multiplicative fusion
        multiplicative = torch.stack(normalized, dim=0).prod(dim=0)
        
        # Weighted additive fusion
        weights = F.softmax(self.expert_weights, dim=0)
        weighted_sum = weights[0] * normalized[0]
        for i in range(1, len(normalized)):
            weighted_sum = weighted_sum + weights[i] * normalized[i]
        
        # Combine both with MLP for nonlinearity
        combined = multiplicative + weighted_sum
        fused = self.mlp(combined)
        
        return fused


class TransformerBaseFusion(nn.Module):
    """
    Transformer-based fusion: uses attention mechanism for feature fusion
    """
    def __init__(self, num_experts, input_dim, hidden_dim=None, num_heads=8, dropout=0.2):
        super().__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        
        if hidden_dim is None:
            hidden_dim = input_dim  # Same size for consistency with other methods
        else:
            hidden_dim = input_dim
        
        self.norms = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_experts)])
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=0.2,
            batch_first=True
        )
        
        # Feed-forward network with same hidden dimension
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, features_list):
        normalized = [norm(f) for norm, f in zip(self.norms, features_list)]
        
        # Stack features for attention (batch_size, num_experts, input_dim)
        stacked_features = torch.stack(normalized, dim=1)
        
        # Apply self-attention
        attended_features, _ = self.attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # Average across experts
        averaged = attended_features.mean(dim=1)
        
        # Apply feed-forward network
        fused = self.ffn(averaged)
        
        return fused


class ConcatenationFusion(nn.Module):
    """
    Concatenation fusion: concatenate features and process through MLP
    """
    def __init__(self, num_experts, input_dim, hidden_dim=None):
        super().__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        
        # Hidden equals feature size (input_dim) to match feature layers
        hidden_dim = input_dim
        
        self.norms = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_experts)])
        
        # MLP: project concatenated features down to feature size via hidden=feature size
        self.mlp = nn.Sequential(
            nn.Linear(num_experts * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, features_list):
        normalized = [norm(f) for norm, f in zip(self.norms, features_list)]
        
        # Concatenate all features
        concatenated = torch.cat(normalized, dim=1)
        
        # Process through MLP
        fused = self.mlp(concatenated)
        
        return fused


class MCNFusionModel(nn.Module):
    """
    Base MCN model with configurable fusion module
    """
    def __init__(self, expert_backbones, input_dim, num_classes, fusion_type="multiplicative", hidden_dim=None):
        super().__init__()
        self.expert_backbones = nn.ModuleList(expert_backbones)
        self.input_dim = input_dim
        
        # Select fusion module based on type
        if fusion_type == "multiplicative":
            self.fusion_module = MultiplicativeFusion(len(expert_backbones), input_dim, hidden_dim)
        elif fusion_type == "multiplicativeAddition":
            self.fusion_module = MultiplicativeAdditionFusion(len(expert_backbones), input_dim, hidden_dim)
        elif fusion_type == "TransformerBase":
            self.fusion_module = TransformerBaseFusion(len(expert_backbones), input_dim, hidden_dim)
        elif fusion_type == "concatenation":
            self.fusion_module = ConcatenationFusion(len(expert_backbones), input_dim, hidden_dim)
        elif fusion_type == "simpleAddition":
            self.fusion_module = SimpleAdditionFusion(len(expert_backbones), input_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}. Available: ['multiplicative', 'multiplicativeAddition', 'TransformerBase', 'concatenation', 'simpleAddition']")
        
        self.global_head = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        features_list, individual_logits = [], []
        for expert in self.expert_backbones:
            features, logits = expert(x)
            features_list.append(features)
            individual_logits.append(logits)
        
        fused_features = self.fusion_module(features_list)
        global_logits = self.global_head(fused_features)
        
        return global_logits, individual_logits
    



# Factory function to create MCN models with different fusion types
def create_mcn_model(expert_backbones, input_dim, num_classes, fusion_type="multiplicative", hidden_dim=None):
    """
    Create an MCN model with the specified fusion type
    
    Args:
        expert_backbones: List of expert backbone models
        input_dim: Dimension of the input feature vectors (configurable)
        num_classes: Number of output classes
        fusion_type: One of ["multiplicative", "multiplicativeAddition", "TransformerBase", "concatenation", "simpleAddition"]
        hidden_dim: Hidden dimension for MLP-based fusions (default: same as input_dim for nonlinearity)
    
    Returns:
        MCNFusionModel instance
    """
    return MCNFusionModel(expert_backbones, input_dim, num_classes, fusion_type, hidden_dim)
