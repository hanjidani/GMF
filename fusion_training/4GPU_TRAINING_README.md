# 🚀 4-GPU Parallel Fusion Training System

## 📋 **Overview**

This system enables **parallel training of all 4 fusion types across all 4 model architectures** using 4 GPUs simultaneously. Each GPU handles one model-fusion combination, maximizing resource utilization and reducing total training time.

## 🏗️ **Architecture**

### **GPU Assignment**
| GPU | Model | Fusion Type | Task ID |
|-----|-------|-------------|---------|
| **GPU 0** | **DenseNet** | `multiplicative` | Task 0 |
| **GPU 1** | **ResNet** | `multiplicativeAddition` | Task 1 |
| **GPU 2** | **Improved WideResNet** | `TransformerBase` | Task 2 |
| **GPU 3** | **ResNeXt** | `concatenation` | Task 3 |

### **Model Architectures**
1. **DenseNet** (`densenet_cifar.py`) - Dense connections, memory efficient
2. **ResNet** (`resnet_cifar.py`) - Residual connections, proven architecture  
3. **Improved WideResNet** (`improved_wide_resnet.py`) - Wide residual networks with improvements
4. **ResNeXt** (`resnext_cifar.py`) - Grouped convolutions, cardinality-based

### **Fusion Types**
1. **multiplicative** - Element-wise multiplication with LayerNorm
2. **multiplicativeAddition** - MLP-based fusion with nonlinearity
3. **TransformerBase** - Multi-head attention mechanism
4. **concatenation** - Feature concatenation with MLP processing

## 🚀 **Quick Start**

### **1. Submit 4-GPU Training Job**
```bash
cd fusion_training/scripts
bash submit_fusion_training_4gpu.sh
```

### **2. Monitor Training**
```bash
# Check job status
squeue -u $USER

# Monitor logs
tail -f logs/train_all_fusions_4gpu_<JOB_ID>.out

# Check individual task logs
tail -f logs/train_all_fusions_4gpu_<JOB_ID>.out | grep "\[task-0-densenet-multiplicative\]"
tail -f logs/train_all_fusions_4gpu_<JOB_ID>.out | grep "\[task-1-resnet-multiplicativeAddition\]"
tail -f logs/train_all_fusions_4gpu_<JOB_ID>.out | grep "\[task-2-improved_wide_resnet-TransformerBase\]"
tail -f logs/train_all_fusions_4gpu_<JOB_ID>.out | grep "\[task-3-resnext-concatenation\]"
```

## 📁 **File Structure**

```
fusion_training/
├── scripts/
│   ├── train_all_fusions_4gpu.sh          # Main 4-GPU Slurm script
│   ├── submit_fusion_training_4gpu.sh     # Submission script
│   ├── train_densenet_fusions.py          # DenseNet fusion training
│   ├── train_resnet_fusions.py            # ResNet fusion training
│   ├── train_improved_wide_resnet_fusions.py  # Improved WideResNet fusion training
│   └── train_resnext_fusions.py           # ResNeXt fusion training
├── models/                                 # Fusion model implementations
├── configs/                                # Configuration files
└── 4GPU_TRAINING_README.md                # This file
```

## 🔧 **Training Scripts**

### **Individual Model Training Scripts**

Each script is designed to:
- **Load the appropriate model architecture** from `expert_training/models/`
- **Load 4 expert checkpoints** for that specific model
- **Train the specified fusion type** with adaptive learning rates
- **Save components independently** for standalone use
- **Log training progress to CSV** for analysis

#### **DenseNet Fusion Training**
```bash
python scripts/train_densenet_fusions.py \
    --fusion_type multiplicative \
    --checkpoint_dir ../expert_training/checkpoints/densenet \
    --output_dir ../fusion_checkpoints \
    --epochs 100 \
    --batch_size 128
```

#### **ResNet Fusion Training**
```bash
python scripts/train_resnet_fusions.py \
    --fusion_type multiplicativeAddition \
    --checkpoint_dir ../expert_training/checkpoints/resnet \
    --output_dir ../fusion_checkpoints \
    --epochs 100 \
    --batch_size 128
```

#### **Improved WideResNet Fusion Training**
```bash
python scripts/train_improved_wide_resnet_fusions.py \
    --fusion_type TransformerBase \
    --checkpoint_dir ../expert_training/checkpoints/improved_wide_resnet \
    --output_dir ../fusion_checkpoints \
    --epochs 100 \
    --batch_size 128
```

#### **ResNeXt Fusion Training**
```bash
python scripts/train_resnext_fusions.py \
    --fusion_type concatenation \
    --checkpoint_dir ../expert_training/checkpoints/resnext \
    --output_dir ../fusion_checkpoints \
    --epochs 100 \
    --batch_size 128
```

## 📊 **Expected Results**

### **Performance Improvements**
| Model | Baseline | With Fusion | Expected Gain |
|-------|----------|-------------|---------------|
| **DenseNet** | 80-84% | 82-87% | +2-3% |
| **ResNet** | 80-84% | 82-87% | +2-3% |
| **Improved WideResNet** | 82-85% | 85-90% | +3-5% |
| **ResNeXt** | 83-86% | 86-91% | +3-5% |

### **Training Time**
- **Sequential Training**: ~8-12 hours (4 models × 2-3 hours each)
- **4-GPU Parallel**: ~2-4 hours (depending on model complexity)
- **Speedup**: **3-4x faster** than sequential training

## 🎯 **Key Features**

### **1. Parallel Processing**
- **4 GPUs simultaneously** - No idle resources
- **Independent training** - Each model-fusion pair trains separately
- **Resource optimization** - Maximum GPU utilization

### **2. Adaptive Learning Rates**
- **Component-specific LRs** - Different rates for experts vs fusion
- **Input dimension adaptation** - Automatic LR adjustment
- **Fusion-specific optimization** - Tailored for each fusion type

### **3. Enhanced Knowledge Transfer**
- **Dual-path architecture** - Individual + collaborative objectives
- **Expert training** - Experts learn alongside fusion (NOT frozen)
- **Gradient control** - Clipping for stability and flexibility

### **4. Comprehensive Logging**
- **CSV logging** - Per-epoch training metrics
- **WandB integration** - Separate projects per model
- **Component tracking** - Individual expert and fusion performance

### **5. Independent Component Saving**
- **Standalone usage** - Each component saved separately
- **Modular deployment** - Mix and match components
- **Easy integration** - Use in other projects

## 📈 **Monitoring & Analysis**

### **Real-time Monitoring**
```bash
# Monitor all tasks
watch -n 5 'squeue -u $USER && echo "---" && nvidia-smi'

# Check specific task progress
grep "\[task-0-densenet-multiplicative\]" logs/train_all_fusions_4gpu_<JOB_ID>.out | tail -10
```

### **CSV Log Analysis**
```python
import pandas as pd

# Load training logs
densenet_log = pd.read_csv('fusion_checkpoints/csv_logs/densenet_fusions/densenet_multiplicative_training_log.csv')
resnet_log = pd.read_csv('fusion_checkpoints/csv_logs/resnet_fusions/resnet_multiplicativeAddition_training_log.csv')

# Analyze learning rates
print("DenseNet Expert LR:", densenet_log['experts_lr'].iloc[-1])
print("ResNet Fusion LR:", resnet_log['fusion_lr'].iloc[-1])
```

### **WandB Projects**
- **MCN_DenseNet_Fusion** - DenseNet fusion training
- **MCN_ResNet_Fusion** - ResNet fusion training  
- **MCN_ImprovedWideResNet_Fusion** - Improved WideResNet fusion training
- **MCN_ResNeXt_Fusion** - ResNeXt fusion training

## 🚨 **Important Notes**

### **Prerequisites**
1. **Expert checkpoints** must exist in expected directories:
   ```
   expert_training/checkpoints/
   ├── densenet/expert_0_best.pth, ..., expert_3_best.pth
   ├── resnet/expert_0_best.pth, ..., expert_3_best.pth
   ├── improved_wide_resnet/expert_0_best.pth, ..., expert_3_best.pth
   └── resnext/expert_0_best.pth, ..., expert_3_best.pth
   ```

2. **Fusion split indices** must exist:
   ```
   splits/fusion_holdout_indices.npy
   ```

3. **CIFAR-100 dataset** must be available in:
   ```
   data/
   ```

### **Resource Requirements**
- **4 GPUs** (V100 or equivalent)
- **16 CPUs** (4 per GPU task)
- **Memory**: 32GB+ per GPU
- **Storage**: 100GB+ for checkpoints and logs

### **Expected Output Structure**
```
fusion_checkpoints/
├── densenet_fusions/
│   ├── multiplicative/
│   │   ├── experts/
│   │   ├── fusion/
│   │   └── global/
│   └── ...
├── resnet_fusions/
├── improved_wide_resnet_fusions/
├── resnext_fusions/
└── csv_logs/
    ├── densenet_fusions/
    ├── resnet_fusions/
    ├── improved_wide_resnet_fusions/
    └── resnext_fusions/
```

## 🔍 **Troubleshooting**

### **Common Issues**

#### **1. Import Errors**
```bash
# Ensure you're in the right directory
cd fusion_training/scripts

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

#### **2. Checkpoint Not Found**
```bash
# Verify checkpoint directories exist
ls -la ../expert_training/checkpoints/
ls -la ../expert_training/checkpoints/densenet/
```

#### **3. GPU Memory Issues**
```bash
# Reduce batch size
--batch_size 64  # Instead of 128

# Check GPU memory usage
nvidia-smi
```

#### **4. Slurm Job Issues**
```bash
# Check job status
scontrol show job <JOB_ID>

# Cancel and resubmit
scancel <JOB_ID>
bash submit_fusion_training_4gpu.sh
```

## 🎉 **Success Indicators**

### **Training Completion**
- All 4 tasks show "completed" status
- No error messages in logs
- All model components saved successfully

### **Performance Metrics**
- Validation accuracy > 80% for all models
- Training loss decreasing over time
- Learning rates adapting properly

### **Output Verification**
- 4 model directories created
- CSV logs generated for each fusion
- WandB projects updated with results

## 🚀 **Next Steps**

After successful training:

1. **Evaluate fusion models** using `evaluate_fusions.py`
2. **Load independent components** using `load_independent_components.py`
3. **Compare performance** across different fusion types
4. **Deploy best models** in production systems

---

**Ready to train all fusion types in parallel? Run `bash submit_fusion_training_4gpu.sh` and watch the magic happen! 🎯**
