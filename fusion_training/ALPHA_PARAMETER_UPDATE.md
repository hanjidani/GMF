# Alpha Parameter Update for Fusion Training Scripts

## Overview

All fusion training scripts have been updated to use `alpha` instead of `lambda_loss` as the parameter name for the dual-path loss balance. Additionally, all Slurm scripts now use 5 GPUs to accommodate all 5 fusion types including the additive fusion. This change makes the parameter naming more intuitive and consistent across all scripts while ensuring complete coverage of all fusion methods.

## Changes Made

### 1. **Parameter Renaming**
- **Before**: `--lambda_loss <value>`
- **After**: `--alpha <value>`

### 2. **GPU Configuration Update**
- **Before**: 4-GPU setup (missing additive fusion)
- **After**: 5-GPU setup (all fusion types covered)

### 3. **Checkpoint Path Updates**
- **Expert Checkpoints**: Now loaded from `../expert_training/scripts/checkpoints_expert_iid/`
- **Baseline Checkpoints**: Now loaded from `../expert_training/scripts/checkpoints_expert_full_dataset_benchmark_250/`

### 4. **File Path Updates**
All file paths and directories now include the alpha value to prevent conflicts when running multiple experiments:

#### **CSV Logging Files**
- Training logs: `{model}_{fusion_type}_alpha_{alpha}_training_log.csv`
- Expert evaluation: `{model}_{fusion_type}_alpha_{alpha}_expert_evaluation.csv`
- Robustness evaluation: `{model}_{fusion_type}_alpha_{alpha}_robustness_evaluation.csv`
- OOD evaluation: `ood_evaluation_{fusion_type}_alpha_{alpha}_{timestamp}.csv`

#### **Model Checkpoint Directories**
- Base directory: `{model}_fusions_alpha_{alpha}/`
- Component subdirectories: `experts/`, `fusion/`, `global/`
- All checkpoint files include alpha in naming: `{model}_{component}_alpha_{alpha}_epoch_{epoch}_acc_{accuracy}.pth`

### 5. **Slurm Script Updates**
All 5-GPU Slurm scripts now require alpha as a command line argument and cover all 5 fusion types:

```bash
# Usage
./train_densenet_fusions_5gpu.sh 1.0
./train_resnet_fusions_5gpu.sh 0.5
./train_resnext_fusions_5gpu.sh 2.0
./train_improved_wide_resnet_fusions_5gpu.sh 1.5

# Error handling
if [ $# -eq 0 ]; then
    echo "Error: Alpha parameter is required"
    echo "Usage: $0 <alpha_value>"
    echo "Example: $0 1.0"
    exit 1
fi
```

## Updated Scripts

### **Python Training Scripts**
1. `train_densenet_fusions.py` - DenseNet fusion training
2. `train_resnet_fusions.py` - ResNet fusion training  
3. `train_resnext_fusions.py` - ResNeXt fusion training
4. `train_improved_wide_resnet_fusions.py` - Improved WideResNet fusion training

### **Slurm Scripts (5-GPU)**
1. `train_densenet_fusions_5gpu.sh` - 5-GPU DenseNet training
2. `train_resnet_fusions_5gpu.sh` - 5-GPU ResNet training
3. `train_resnext_fusions_5gpu.sh` - 5-GPU ResNeXt training
4. `train_improved_wide_resnet_fusions_5gpu.sh` - 5-GPU Improved WideResNet training

## Checkpoint Paths

### **Expert Checkpoints (IID Training)**
All scripts now load expert models from the IID training checkpoints:
```
../expert_training/scripts/checkpoints_expert_iid/
├── best_iid_densenet121_expert_0.pth
├── best_iid_densenet121_expert_1.pth
├── best_iid_densenet121_expert_2.pth
├── best_iid_densenet121_expert_3.pth
├── best_iid_resnet18_expert_0.pth
├── best_iid_resnet18_expert_1.pth
├── best_iid_resnet18_expert_2.pth
├── best_iid_resnet18_expert_3.pth
├── best_iid_preact_resnext29_8x64d_expert_0.pth
├── best_iid_preact_resnext29_8x64d_expert_1.pth
├── best_iid_preact_resnext29_8x64d_expert_2.pth
├── best_iid_preact_resnext29_8x64d_expert_3.pth
├── best_iid_wideresnet28_10_expert_0.pth
├── best_iid_wideresnet28_10_expert_1.pth
├── best_iid_wideresnet28_10_expert_2.pth
└── best_iid_wideresnet28_10_expert_3.pth
```

### **Baseline Checkpoints (Full Dataset Benchmark)**
All scripts now load baseline models from the full dataset benchmark checkpoints:
```
../expert_training/scripts/checkpoints_expert_full_dataset_benchmark_250/
├── best_full_dataset_densenet121_benchmark_250.pth
├── best_full_dataset_resnet18_benchmark_250.pth
├── best_full_dataset_resnext29_8x64d_benchmark_250.pth
└── best_full_dataset_wideresnet28_10_benchmark_250.pth
```

## Fusion Types Covered

All 5 fusion types are now covered with dedicated GPU assignments:

| Task ID | GPU | Fusion Type | Description |
|---------|-----|-------------|-------------|
| 0 | GPU 0 | `multiplicative` | Element-wise multiplication fusion |
| 1 | GPU 1 | `multiplicativeAddition` | Combined multiplication and addition |
| 2 | GPU 2 | `TransformerBase` | Transformer-based fusion mechanism |
| 3 | GPU 3 | `concatenation` | Feature concatenation fusion |
| 4 | GPU 4 | `additive` | Element-wise addition fusion |

## Benefits

### **1. Complete Coverage**
- All 5 fusion types are now trained simultaneously
- No fusion method is left out
- Efficient use of all available GPUs

### **2. Correct Checkpoint Sources**
- Experts loaded from IID training (proper expert diversity)
- Baselines loaded from full dataset benchmark (proper comparison)
- Consistent checkpoint naming across all architectures

### **3. Conflict Prevention**
- Multiple experiments with different alpha values can run simultaneously
- No file overwriting or directory conflicts
- Clear separation of results by alpha value

### **4. Better Organization**
- All outputs are grouped by alpha value
- Easy to compare results across different alpha settings
- Clear audit trail for experiments

### **5. Improved Usability**
- Alpha parameter is now required (prevents accidental default runs)
- Clear error messages guide users on proper usage
- Consistent parameter naming across all scripts

## Usage Examples

### **Single GPU Training**
```bash
python train_densenet_fusions.py --fusion_type multiplicative --alpha 1.0
python train_resnet_fusions.py --fusion_type additive --alpha 0.5
```

### **5-GPU Parallel Training**
```bash
# Submit to Slurm with alpha=1.0
sbatch train_densenet_fusions_5gpu.sh 1.0

# Submit to Slurm with alpha=0.5  
sbatch train_resnet_fusions_5gpu.sh 0.5

# Submit to Slurm with alpha=2.0
sbatch train_resnext_fusions_5gpu.sh 2.0
```

### **Multiple Alpha Experiments**
```bash
# Run multiple alpha values simultaneously
sbatch train_densenet_fusions_5gpu.sh 0.5
sbatch train_densenet_fusions_5gpu.sh 1.0
sbatch train_densenet_fusions_5gpu.sh 2.0

# Each will create separate directories and files
# - densenet_fusions_alpha_0.5/
# - densenet_fusions_alpha_1.0/
# - densenet_fusions_alpha_2.0/
```

## File Structure After Update

```
fusion_checkpoints/
├── densenet_fusions_alpha_1.0/
│   ├── multiplicative/
│   │   ├── experts/
│   │   │   ├── densenet_expert_0_alpha_1.0_epoch_50_acc_85.23.pth
│   │   │   └── densenet_expert_0_alpha_1.0_best.pth
│   │   ├── fusion/
│   │   │   └── densenet_multiplicative_alpha_1.0_fusion_epoch_50_acc_85.23.pth
│   │   └── global/
│   │       └── densenet_multiplicative_alpha_1.0_global_head_epoch_50_acc_85.23.pth
│   ├── multiplicativeAddition/
│   ├── TransformerBase/
│   ├── concatenation/
│   └── additive/
├── densenet_fusions_alpha_0.5/
│   └── ...
└── csv_logs/
    └── densenet_fusions/
        ├── densenet_multiplicative_alpha_1.0_training_log.csv
        ├── densenet_additive_alpha_1.0_training_log.csv
        └── ...
```

## Migration Notes

### **For Existing Users**
- Update any scripts that call the training scripts
- Change `--lambda_loss` to `--alpha` in all calls
- Update any post-processing scripts that parse CSV files
- Check for any hardcoded file paths that include "lambda"
- Note that scripts now use 5 GPUs instead of 4
- Checkpoint paths have been updated to use IID experts and full dataset baselines

### **For New Users**
- Always specify alpha when running training
- Use descriptive alpha values for easy identification
- Check output directories to ensure proper separation
- Use consistent alpha values across related experiments
- Ensure your compute node has at least 5 GPUs available
- Verify that checkpoint directories exist and contain the required files

## Technical Details

### **Hidden Dimensions**
All scripts maintain the proper hidden dimensions for each architecture:
- **DenseNet**: 1024 → 1536
- **ResNet**: 512 → 768  
- **ResNeXt**: 256 → 384
- **Improved WideResNet**: 640 → 960

### **CSV Headers Updated**
All CSV files now use `alpha` instead of `lambda_loss` in their column headers and data.

### **GPU Requirements**
- **Minimum**: 5 GPUs per node
- **Recommended**: V100 or better for optimal performance
- **Memory**: Each GPU should have sufficient memory for the model architecture

### **Checkpoint Requirements**
- **Expert Checkpoints**: Must exist in `checkpoints_expert_iid/` directory
- **Baseline Checkpoints**: Must exist in `checkpoints_expert_full_dataset_benchmark_250/` directory
- **File Naming**: Must follow the expected naming convention for each architecture

### **Backward Compatibility**
This is a breaking change - existing scripts using `--lambda_loss` will need to be updated to use `--alpha`.

## Summary

The alpha parameter update and 5-GPU configuration provides:
1. **Better naming convention** (alpha vs lambda_loss)
2. **Complete fusion coverage** (all 5 fusion types)
3. **Correct checkpoint sources** (IID experts + full dataset baselines)
4. **Conflict-free file organization** (alpha in all paths)
5. **Required parameter validation** (prevents accidental defaults)
6. **Consistent interface** across all fusion training scripts
7. **Improved experiment management** (clear separation by alpha value)
8. **Efficient resource utilization** (5 GPUs for 5 fusion types)

All scripts are now ready for production use with proper alpha parameter handling, conflict-free file organization, complete coverage of all fusion methods, and correct checkpoint loading from the specified directories.
