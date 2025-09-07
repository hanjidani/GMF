# Experiment Strategy for Multi-Expert Collaborative Network (MCN)

## 🎯 **Updated Training Protocol: 250 Epochs for All Models**

### **Why This Standard?**

- **Fair Comparison**: All models get identical training time
- **Academic Consistency**: Matches CIFAR-100 research protocols  
- **Full Convergence**: Ensures models reach their potential
- **Eliminates Excuses**: No "not enough training time" arguments

## 🚀 **Recommended Model Combination for Experiments**

### **Goal 1: Same Architecture Fusion (ResNet-18 + ResNet-18)**
```bash
# Expert 1: ResNet-18
python train_iid_experts.py --model resnet18 --expert_id 0

# Expert 2: ResNet-18  
python train_iid_experts.py --model resnet18 --expert_id 1

# Expected Results: 75-78% accuracy each, 250 epochs
```

### **Goal 2: Weak-Strong Guidance (ResNet-18 + PreAct-ResNeXt-29)**
```bash
# Expert 1: ResNet-18 (Weak)
python train_iid_experts.py --model resnet18 --expert_id 0

# Expert 2: PreAct-ResNeXt-29 (Strong)
python train_iid_experts.py --model preact_resnext29_8x64d --expert_id 1

# Expected Results: 
# - ResNet-18: 75-78% accuracy, 250 epochs
# - PreAct-ResNeXt: 82-85% accuracy, 250 epochs
```

## 📊 **Training Configuration**

### **Standard Parameters (All Models):**
- **Epochs**: 250 (fixed for fair comparison)
- **Batch Size**: 128
- **Learning Rate**: 0.1 (model-specific overrides available)
- **Early Stopping**: 75% accuracy threshold + 30 epochs patience
- **Minimum Epochs**: 120 before early stopping consideration

### **Model-Specific Optimizations:**
- **ResNet-18**: LR=0.2 (faster convergence)
- **DenseNet**: Weight decay=1e-4 (better regularization)
- **WideResNet**: Dropout=0.3 (prevent overfitting)
- **ResNeXt**: Cosine LR schedule (smooth convergence)

## 🔬 **Experimental Design**

### **Phase 1: Individual Expert Training (Current)**
- Train each expert independently for 250 epochs
- Use IID data distribution (40% shared, 15% unique per expert)
- Apply SOTA augmentation for each architecture
- Monitor convergence and early stopping

### **Phase 2: Fusion Experiments (Future)**
- Combine trained experts using `simple_mcn.py`
- Test different fusion strategies
- Evaluate collaborative vs individual performance
- Analyze expert specialization patterns

## 📈 **Expected Outcomes**

### **Individual Expert Performance:**
| **Model** | **Expected Accuracy** | **Training Time** | **Convergence** |
|-----------|----------------------|-------------------|-----------------|
| **ResNet-18** | 75-78% | ~62 hours | ~180-220 epochs |
| **DenseNet-121** | 78-82% | ~83 hours | ~200-230 epochs |
| **WideResNet-28-10** | 80-84% | ~104 hours | ~200-230 epochs |
| **PreAct-ResNeXt-29** | 82-85% | ~125 hours | ~200-240 epochs |

### **Collaborative Benefits:**
- **Same Architecture**: Expected 1-3% improvement
- **Weak-Strong**: Expected 2-5% improvement for weak model
- **Knowledge Transfer**: Better feature representations

## 🎛️ **Training Commands**

### **Quick Start (All Experts):**
```bash
cd expert_training/scripts

# Train all 4 experts with ResNet-18
for expert_id in {0..3}; do
    python train_iid_experts.py --model resnet18 --expert_id $expert_id --no_wandb
done
```

### **Best Performance Setup:**
```bash
# Expert 0: ResNet-18 (fast, baseline)
python train_iid_experts.py --model resnet18 --expert_id 0

# Expert 1: DenseNet-121 (balanced)
python train_iid_experts.py --model densenet121 --expert_id 1

# Expert 2: WideResNet-28-10 (high performance)
python train_iid_experts.py --model wideresnet28_10 --expert_id 2

# Expert 3: PreAct-ResNeXt-29 (best overall)
python train_iid_experts.py --model preact_resnext29_8x64d --expert_id 3
```

### **Slurm Cluster Training:**
```bash
# Submit 16-GPU parallel training
sbatch train_iid_16gpu.sh

# This will train all models in parallel on pascal-node10
# Each model gets 4 GPUs, total training time: ~125 hours
```

## 🔍 **Monitoring & Analysis**

### **Per-Epoch Logging:**
- CSV files saved every 5 epochs
- Location: `checkpoints_expert_iid/per_epoch_logs/`
- Format: `iid_{model_name}_expert_{expert_id}_epochs.csv`

### **Key Metrics to Track:**
1. **Training Loss**: Should decrease steadily
2. **Validation Accuracy**: Target 75%+ threshold
3. **Learning Rate**: Cosine decay schedule
4. **Early Stopping**: Only after 75% accuracy
5. **Convergence Time**: When accuracy plateaus

### **Success Criteria:**
- ✅ All models reach 75%+ accuracy
- ✅ Training completes 250 epochs (or early stops appropriately)
- ✅ CSV logs saved successfully
- ✅ No hardware errors or crashes

## 🚨 **Troubleshooting**

### **Common Issues:**
1. **Early Stopping Too Soon**: Increase patience or check 75% threshold
2. **Low Accuracy**: Verify data loading and augmentation
3. **Training Hangs**: Check GPU memory and timeout settings
4. **Missing CSVs**: Verify save frequency and file paths

### **Performance Optimization:**
- Use mixed precision (AMP) for faster training
- Adjust batch size based on GPU memory
- Monitor GPU utilization during training
- Use appropriate learning rate schedules

## 🎯 **Next Steps**

1. **Complete Individual Training**: Train all experts to 250 epochs
2. **Validate Performance**: Ensure 75%+ accuracy threshold met
3. **Prepare Fusion**: Ready `simple_mcn.py` for collaborative experiments
4. **Analyze Results**: Compare individual vs collaborative performance

**The 250-epoch standard ensures fair comparison and optimal performance for all models!** 🚀
