# fusion_non_iid/models/fusion_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# === Part 1: Fusion Modules (Building Blocks) ===
# These modules operate on a list of feature tensors.

class SimpleAdditionFusion(nn.Module):
    def __init__(self, num_experts, input_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        self.norms = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_experts)])
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, features_list):
        normalized = [norm(f) for norm, f in zip(self.norms, features_list)]
        fused = torch.stack(normalized, dim=0).sum(dim=0)
        return self.mlp(fused)

class MultiplicativeFusion(nn.Module):
    def __init__(self, num_experts, input_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        self.norms = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_experts)])
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, features_list):
        normalized = [norm(f) for norm, f in zip(self.norms, features_list)]
        fused = torch.stack(normalized, dim=0).prod(dim=0)
        return self.mlp(fused)

class MultiplicativeAdditionFusion(nn.Module):
    def __init__(self, num_experts, input_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        self.norms = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_experts)])
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, features_list):
        normalized = [norm(f) for norm, f in zip(self.norms, features_list)]
        # Multiplicative + Addition combination
        multiplicative = torch.stack(normalized, dim=0).prod(dim=0)
        addition = torch.stack(normalized, dim=0).sum(dim=0)
        fused = multiplicative + addition
        return self.mlp(fused)

class MultiplicativeShiftedFusion(nn.Module):
    def __init__(self, num_experts, input_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        self.norms = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_experts)])
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, features_list):
        normalized = [norm(f) for norm, f in zip(self.norms, features_list)]
        shifted = [f + 1 for f in normalized]
        fused = torch.stack(shifted, dim=0).prod(dim=0)
        return self.mlp(fused)

class TransformerBaseFusion(nn.Module):
    def __init__(self, num_experts, input_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        self.norms = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_experts)])
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=8, dim_feedforward=hidden_dim, batch_first=True),
            num_layers=2
        )
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, features_list):
        normalized = [norm(f) for norm, f in zip(self.norms, features_list)]
        # Stack features as sequence
        stacked = torch.stack(normalized, dim=1)  # [batch, num_experts, input_dim]
        # Apply transformer
        transformed = self.transformer(stacked)
        # Average pool across experts
        fused = transformed.mean(dim=1)
        return self.mlp(fused)

class ConcatenationFusion(nn.Module):
    def __init__(self, num_experts, input_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        self.norms = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_experts)])
        self.mlp = nn.Sequential(
            nn.Linear(num_experts * input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, features_list):
        normalized = [norm(f) for norm, f in zip(self.norms, features_list)]
        concatenated = torch.cat(normalized, dim=1)
        return self.mlp(concatenated)

# === Part 2: The Main MCN Model (The "Head Swap" Implementation) ===
class MCNFusionModel(nn.Module):
    """
    Base MCN model for Non-IID specialists.
    This model correctly separates backbones from newly created individual heads.
    """
    def __init__(self, expert_backbones, input_dim, num_classes, fusion_type="multiplicative", hidden_dim=None):
        super().__init__()
        self.expert_backbones = nn.ModuleList(expert_backbones)
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # CRITICAL: Create new, trainable 100-class heads for each expert's individual path.
        # These are initialized randomly and will be trained from scratch.
        self.individual_heads = nn.ModuleList(
            [nn.Linear(input_dim, num_classes) for _ in range(len(self.expert_backbones))]
        )
        
        # Select the fusion module based on the provided type
        if fusion_type == "multiplicative":
            self.fusion_module = MultiplicativeFusion(len(expert_backbones), input_dim, hidden_dim)
        elif fusion_type == "concatenation":
            self.fusion_module = ConcatenationFusion(len(expert_backbones), input_dim, hidden_dim)
        elif fusion_type == "multiplicativeAddition":
            self.fusion_module = MultiplicativeAdditionFusion(len(expert_backbones), input_dim, hidden_dim)
        elif fusion_type == "multiplicativeShifted":
            self.fusion_module = MultiplicativeShiftedFusion(len(expert_backbones), input_dim, hidden_dim)
        elif fusion_type == "TransformerBase":
            self.fusion_module = TransformerBaseFusion(len(expert_backbones), input_dim, hidden_dim)
        elif fusion_type == "simpleAddition":
            self.fusion_module = SimpleAdditionFusion(len(expert_backbones), input_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # The global head that operates on the fused features
        self.global_head = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        features_list = []
        # Step 1: Get features from each specialist backbone
        for backbone in self.expert_backbones:
            f = backbone(x)  # Pass input through the feature extractor
            # This post-processing is specific to DenseNet's backbone output
            f = F.relu(f, inplace=True)
            f = F.adaptive_avg_pool2d(f, (1, 1))
            f = torch.flatten(f, 1)
            features_list.append(f)
        
        # Step 2: Get individual logits from the NEW, separate 100-class heads
        individual_logits = [head(f) for head, f in zip(self.individual_heads, features_list)]
        
        # Step 3: Fuse the features for the global path
        fused_features = self.fusion_module(features_list)
        global_logits = self.global_head(fused_features)
        
        # Step 4: Return both global and individual predictions for the dual-path loss
        return global_logits, individual_logits

# === Part 3: Factory Function ===
def create_mcn_model(expert_backbones, input_dim, num_classes, fusion_type="multiplicative", hidden_dim=None):
    """Factory function to instantiate the MCNFusionModel."""
    return MCNFusionModel(expert_backbones, input_dim, num_classes, fusion_type, hidden_dim)