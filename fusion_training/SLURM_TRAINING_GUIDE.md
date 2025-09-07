# Slurm Training Guide for MCN Fusion Models

This guide explains how to train the fusion models using Slurm on your cluster.

## 🚀 Quick Start

### 1. Submit All Fusion Jobs
```bash
cd fusion_training
./scripts/submit_jobs.sh all
```

### 2. Submit Individual Fusion Jobs
```bash
# Train multiplicative fusion
./scripts/submit_jobs.sh multiplicative

# Train TransformerBase fusion with custom epochs
./scripts/submit_jobs.sh TransformerBase 150

# Train concatenation fusion with custom epochs and batch size
./scripts/submit_jobs.sh concatenation 100 64
```

## 📋 Available Fusion Models

| Fusion Type | Description | Default Epochs | Expected Accuracy | Training Time |
|-------------|-------------|----------------|-------------------|---------------|
| **multiplicative** | Simple element-wise multiplication | 80 | 80-85% | ~2-3 hours |
| **multiplicativeAddition** | Multiplicative + Additive + MLP | 100 | 82-87% | ~3-4 hours |
| **concatenation** | Feature concatenation + MLP | 100 | 81-86% | ~3-4 hours |
| **TransformerBase** | Attention-based fusion | 120 | 84-89% | ~4-5 hours |

## 🔧 Slurm Configuration

### Resource Requirements
- **Time**: 12 hours (configurable)
- **Memory**: 32GB
- **CPU**: 4 cores
- **GPU**: 1 GPU
- **Partition**: gpu

### Job Submission Scripts

#### 1. `submit_fusion_slurm.sh`
Main Slurm job script that handles:
- Resource allocation
- Environment setup
- Training execution
- Logging and checkpointing

#### 2. `submit_jobs.sh`
Quick submission script for:
- Individual fusion model training
- Batch submission of all models
- Custom epoch and batch size configuration

## 📊 Training Process

### 🚨 **IMPORTANT: Experts are NOT Frozen!**
- ✅ **Experts + Fusion Trained Together**: All components learn simultaneously
- ✅ **Independent Component Saving**: Each component saved separately for standalone use
- ✅ **Dual Learning Rates**: Different rates for experts vs. fusion/global head
- ✅ **Component Reusability**: Load experts, fusion, or global head independently

### Training Strategy
1. **Load Pre-trained Experts**: Uses expert checkpoints from `../../checkpoints_iid/` as initialization
2. **Joint Training**: Experts, fusion, and global head train together on fusion split data
3. **Dual Learning Rates**: 
   - Experts: `base_lr` (e.g., 1e-5) - slower learning
   - Fusion + Global Head: `head_lr` (e.g., 1e-3) - faster learning
4. **Independent Saving**: Each component saved separately every 10 epochs
5. **Complete Models**: Also saves complete models for convenience

### Automatic Features
- ✅ **Input Dimension Detection**: Automatically detects feature dimensions from expert models
- ✅ **Hidden Dimension Optimization**: Sets hidden_dim = input_dim for optimal nonlinearity
- ✅ **Learning Rate Optimization**: Auto-optimizes learning rates based on fusion type and input dimension
- ✅ **Independent Checkpointing**: Saves experts, fusion, and global head separately
- ✅ **CSV Logging**: Detailed per-epoch logging for analysis
- ✅ **WandB Integration**: Experiment tracking and visualization

## 📁 File Structure

```
fusion_training/
├── scripts/
│   ├── train_fusion_slurm.py      # Main training script (experts NOT frozen)
│   ├── submit_fusion_slurm.sh     # Slurm job script
│   ├── submit_jobs.sh             # Quick submission script
│   ├── load_independent_components.py  # Load components independently
│   └── ...                        # Other training scripts
├── models/
│   └── fusion_models.py           # Fusion model implementations
├── configs/
│   └── fusion_configs.py          # Configuration and learning rates
└── logs/                          # Slurm job logs
```

## 💾 Component Saving Strategy

### Independent Component Structure
```
fusion_checkpoints/
├── experts/                        # Individual expert models
│   ├── expert_0_epoch_100.pth
│   ├── expert_1_epoch_100.pth
│   ├── expert_2_epoch_100.pth
│   └── expert_3_epoch_100.pth
├── fusion/                         # Fusion modules
│   ├── fusion_multiplicative_epoch_100.pth
│   ├── fusion_multiplicativeAddition_epoch_100.pth
│   ├── fusion_TransformerBase_epoch_100.pth
│   └── fusion_concatenation_epoch_100.pth
├── global/                         # Global heads
│   ├── global_head_epoch_100.pth
│   └── ...
└── complete_fusion_*.pth          # Complete models (for convenience)
```

### Component Loading
```bash
# List available components
python scripts/load_independent_components.py --checkpoint_dir ../../fusion_checkpoints --action list

# Reconstruct MCN model from components
python scripts/load_independent_components.py --checkpoint_dir ../../fusion_checkpoints --action reconstruct --fusion_type multiplicative

# Load complete model
python scripts/load_independent_components.py --checkpoint_dir ../../fusion_checkpoints --action load_complete --fusion_type TransformerBase
```

## 🎯 Job Management

### Submit Jobs
```bash
# Submit all fusion models
./scripts/submit_jobs.sh all

# Submit specific model
./scripts/submit_jobs.sh multiplicative

# Submit with custom parameters
./scripts/submit_jobs.sh TransformerBase 150 64
```

### Monitor Jobs
```bash
# Check job status
squeue -u $USER

# Monitor specific job
squeue -j <job_id>

# Check job output
tail -f logs/fusion_<job_id>.out

# Check job errors
tail -f logs/fusion_<job_id>.err
```

### Cancel Jobs
```bash
# Cancel specific job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER
```

## 📈 Expected Results

### Performance Comparison
- **Multiplicative**: Fastest training, good baseline performance
- **MultiplicativeAddition**: Balanced complexity and performance
- **Concatenation**: Good performance with MLP processing
- **TransformerBase**: Best expected performance, longer training time

### Output Files
- **Independent Components**: `../../fusion_checkpoints/{experts,fusion,global}/`
  - `expert_{i}_epoch_{N}.pth` - Individual expert models
  - `fusion_{type}_epoch_{N}.pth` - Fusion modules
  - `global_head_epoch_{N}.pth` - Global heads
- **Complete Models**: `../../fusion_checkpoints/`
  - `complete_fusion_{type}_epoch_{N}.pth` - Complete models
  - `best_complete_fusion_{type}.pth` - Best performing complete model
- **CSV Logs**: `../../fusion_checkpoints/fusion_{type}_report.csv`
- **Results Summary**: `../../fusion_checkpoints/fusion_training_results_{type}.json`
- **Slurm Logs**: `logs/fusion_{job_id}.out`

## 🔍 Troubleshooting

### Common Issues

#### 1. Job Fails to Submit
```bash
# Check Slurm status
sinfo

# Check partition availability
sinfo -p gpu

# Verify script permissions
ls -la scripts/submit_fusion_slurm.sh
```

#### 2. Training Fails
```bash
# Check expert checkpoints exist
ls -la ../../checkpoints_iid/best_wrn_expert_*.pth

# Check fusion split exists
ls -la ../../splits/fusion_holdout_indices.npy

# Check GPU availability
nvidia-smi
```

#### 3. Out of Memory
```bash
# Reduce batch size
./scripts/submit_jobs.sh multiplicative 80 64

# Check memory usage
tail -f logs/fusion_<job_id>.out
```

### Performance Optimization

#### 1. Batch Size Tuning
- **32GB Memory**: Batch size 64-128
- **64GB Memory**: Batch size 128-256
- **Multi-GPU**: Increase batch size proportionally

#### 2. Epoch Optimization
- **Simple Models**: 80-100 epochs
- **Complex Models**: 120-150 epochs
- **Early Stopping**: Monitor validation accuracy

#### 3. Learning Rate Tuning
- **Experts**: 1e-5 (slower learning, stable)
- **Fusion + Global Head**: 1e-3 (faster learning, adaptation)
- **Model-specific adjustments**: Based on fusion complexity

## 🚀 Advanced Usage

### Custom Training Configuration
```bash
# Train with custom parameters
python scripts/train_fusion_slurm.py \
    --fusion_type TransformerBase \
    --epochs 200 \
    --batch_size 64 \
    --checkpoint_dir /path/to/experts \
    --output_dir /path/to/output
```

### Multi-GPU Training
```bash
# Modify submit_fusion_slurm.sh
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
```

### Custom Slurm Parameters
```bash
# Submit with custom Slurm options
sbatch --time=24:00:00 --mem=64G scripts/submit_fusion_slurm.sh multiplicative
```

## 📊 Post-Training Analysis

### Evaluate All Models
```bash
cd fusion_training
python scripts/evaluate_fusions.py
```

### Load Components Independently
```bash
# List all available components
python scripts/load_independent_components.py --checkpoint_dir ../../fusion_checkpoints --action list

# Test individual expert
python scripts/load_independent_components.py --checkpoint_dir ../../fusion_checkpoints --action reconstruct --fusion_type multiplicative --num_experts 1
```

### Compare Performance
```bash
# Check CSV logs
head -20 ../../fusion_checkpoints/fusion_multiplicative_report.csv

# Analyze results
python -c "
import pandas as pd
df = pd.read_csv('../../fusion_checkpoints/fusion_multiplicative_report.csv')
print(df[df['phase']=='train'][['epoch', 'global_accuracy']].tail(10))
"
```

### Generate Reports
```bash
# Create performance summary
python scripts/evaluate_fusions.py --output_file fusion_performance_summary.csv
```

## 🎉 Success Criteria

### Training Completion
- ✅ All fusion models trained successfully with experts
- ✅ Independent components saved for standalone use
- ✅ Complete models saved for convenience
- ✅ CSV logs generated for analysis
- ✅ WandB experiments tracked

### Performance Targets
- **Multiplicative**: ≥80% accuracy
- **MultiplicativeAddition**: ≥82% accuracy
- **Concatenation**: ≥81% accuracy
- **TransformerBase**: ≥84% accuracy

### Quality Checks
- Training loss decreases steadily
- Validation accuracy improves
- No overfitting (validation > training)
- All components saved successfully
- Independent loading works correctly

## 🔧 Component Independence Benefits

### Standalone Usage
- ✅ **Individual Experts**: Load and use any expert independently
- ✅ **Fusion Modules**: Swap fusion strategies without retraining experts
- ✅ **Global Heads**: Adapt to different classification tasks
- ✅ **Component Mixing**: Combine different epoch checkpoints

### Research Flexibility
- ✅ **Ablation Studies**: Test individual components
- ✅ **Transfer Learning**: Use experts in other tasks
- ✅ **Model Analysis**: Examine fusion vs. expert contributions
- ✅ **Incremental Training**: Continue training from any checkpoint

---

**Ready to train your fusion models with experts? Start with:**
```bash
cd fusion_training
./scripts/submit_jobs.sh all
```

**After training, explore components independently:**
```bash
python scripts/load_independent_components.py --checkpoint_dir ../../fusion_checkpoints --action list
```
