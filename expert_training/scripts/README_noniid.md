# Non-IID Expert Training

This directory contains the Non-IID (Non-Independent and Identically Distributed) expert training implementation for the MCN project.

## Overview

The Non-IID training implements a **class-based distribution** strategy where:
- **4 experts** are trained, each specializing in **25 different classes**
- **Expert 0**: Classes 0-24 (labels 0-24)
- **Expert 1**: Classes 25-49 (labels 0-24, remapped)
- **Expert 2**: Classes 50-74 (labels 0-24, remapped)
- **Expert 3**: Classes 75-99 (labels 0-24, remapped)

Each expert model outputs **1x25** logits, and the labels are remapped to the 0-24 range during training.

**Important**: This implementation uses the **entire expert set** from the `splits/expert_train_indices.npy` file and filters by class to create the non-IID distribution. Each expert sees **ALL available samples** from their assigned 25 classes, resulting in approximately 9,000-10,000 samples per expert.

## Key Differences from IID Training

| Aspect | IID Training | Non-IID Training |
|--------|--------------|------------------|
| Data Distribution | Shared + Unique samples | Class-based separation |
| Expert Overlap | 40% shared data | No shared data |
| Model Output | 1x100 (all classes) | 1x25 (expert-specific classes) |
| Label Range | 0-99 | 0-24 (remapped) |
| Checkpoint Directory | `checkpoints_expert_iid` | `checkpoints_expert_noniid` |
| Wandb Project | `MCN_IID_Experts` | `MCN_NonIID_Experts` |

## Files

### Core Training Files
- **`train_noniid.py`** - Core non-IID training function
- **`train_noniid_experts.py`** - Main script that calls train_noniid.py
- **`train_noniid_16gpu.sh`** - Slurm script for 16-GPU training

### Testing and Documentation
- **`test_noniid_setup.py`** - Test script to verify setup
- **`README_noniid.md`** - This documentation file

## Data Source

The non-IID training uses the **entire expert set** from `splits/expert_train_indices.npy`:
- **Source**: Complete expert indices from the original data splitting
- **Method**: Filter entire expert set by class to create class-based distribution
- **Benefit**: Each expert sees all available samples from their assigned classes
- **Result**: Each expert gets approximately 9,000-10,000 samples from their 25 assigned classes

## Usage

### 1. Test the Setup
```bash
cd expert_training/scripts
python test_noniid_setup.py
```

### 2. Train a Single Expert
```bash
python train_noniid_experts.py --model resnet18 --expert_id 0
```

### 3. Train All Experts with a Model
```bash
python train_noniid_experts.py --model wideresnet28_10
```

### 4. Train on Cluster (16 GPUs)
```bash
# Submit to Slurm
sbatch train_noniid_16gpu.sh wideresnet28_10

# Or run directly (if you have 16 GPUs)
bash train_noniid_16gpu.sh resnet18
```

## Supported Models

All models from the IID training are supported:
- **WideResNet**: `wideresnet28_10`, `wideresnet40_2`
- **ResNeXt**: `resnext29_8x64d`, `resnext29_16x64d`, `preact_resnext29_8x64d`
- **ResNet**: `resnet18`, `resnet34`, `resnet50`, `preact_resnet18`
- **DenseNet**: `densenet121`, `densenet169`, `densenet201`, `efficient_densenet`

## Output Structure

```
checkpoints_expert_noniid/
├── best_noniid_wideresnet28_10_expert_0.pth
├── best_noniid_wideresnet28_10_expert_1.pth
├── best_noniid_wideresnet28_10_expert_2.pth
├── best_noniid_wideresnet28_10_expert_3.pth
├── per_epoch_logs/
│   ├── noniid_wideresnet28_10_expert_0_epochs.csv
│   ├── noniid_wideresnet28_10_expert_1_epochs.csv
│   ├── noniid_wideresnet28_10_expert_2_epochs.csv
│   └── noniid_wideresnet28_10_expert_3_epochs.csv
└── noniid_training_results_wideresnet28_10.json
```

## Training Configuration

The training uses the same configuration as IID training but with:
- **Model output**: 25 classes instead of 100
- **Label remapping**: Global labels (0-99) → Expert labels (0-24)
- **Test set filtering**: Only classes relevant to each expert
- **Checkpoint naming**: `noniid_` prefix to avoid conflicts

## Class Distribution Details

### Expert 0 (Classes 0-24)
- **Original labels**: 0, 1, 2, ..., 23, 24
- **Remapped labels**: 0, 1, 2, ..., 23, 24
- **Samples**: ~10,000 (all available from expert set)

### Expert 1 (Classes 25-49)
- **Original labels**: 25, 26, 27, ..., 48, 49
- **Remapped labels**: 0, 1, 2, ..., 23, 24
- **Samples**: ~10,000 (all available from expert set)

### Expert 2 (Classes 50-74)
- **Original labels**: 50, 51, 52, ..., 73, 74
- **Remapped labels**: 0, 1, 2, ..., 23, 24
- **Samples**: ~10,000 (all available from expert set)

### Expert 3 (Classes 75-99)
- **Original labels**: 75, 76, 77, ..., 98, 99
- **Remapped labels**: 0, 1, 2, ..., 23, 24
- **Samples**: ~10,000 (all available from expert set)

## Monitoring and Logging

- **Wandb**: Project `MCN_NonIID_Experts` with expert-specific runs
- **CSV Logs**: Per-epoch training metrics saved every 10 epochs
- **Checkpoints**: Best model saved based on test accuracy
- **Results Summary**: JSON file with training results for all experts

## Example Training Command

```bash
# Train all experts with WideResNet-28-10
python train_noniid_experts.py \
    --model wideresnet28_10 \
    --epochs 250 \
    --batch_size 128 \
    --lr 0.1

# Train only expert 2 with ResNet-18
python train_noniid_experts.py \
    --model resnet18 \
    --expert_id 2 \
    --epochs 200 \
    --batch_size 128 \
    --lr 0.1
```

## Notes

1. **No Data Overlap**: Unlike IID training, experts have completely separate data
2. **Class Specialization**: Each expert becomes specialized in its 25 classes
3. **Label Consistency**: All experts use 0-24 labels internally
4. **Independent Training**: Experts can be trained in parallel or sequentially
5. **Conflict Prevention**: Separate checkpoint directories and naming prevent conflicts with IID training
