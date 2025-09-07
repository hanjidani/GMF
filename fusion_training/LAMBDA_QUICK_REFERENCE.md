# 🎯 **Lambda Tuning Quick Reference**

## ⚡ **Quick Setup**

### **Change Lambda Value**
```bash
# Edit the main script
sed -i 's/LAMBDA_LOSS=.*/LAMBDA_LOSS=0.1/' train_all_fusions_4gpu.sh

# Or manually edit
nano train_all_fusions_4gpu.sh
# Change: LAMBDA_LOSS=1.0 to LAMBDA_LOSS=0.1
```

### **Submit Training**
```bash
cd fusion_training/scripts
bash submit_fusion_training_4gpu.sh
```

## 📊 **Lambda Values & Effects**

| λ Value | Effect | Use Case |
|---------|--------|----------|
| **0.1** | Minimal individual training | Strong global fusion focus |
| **0.5** | Reduced individual emphasis | Balance favoring global fusion |
| **1.0** | Equal balance (default) | Standard balanced training |
| **2.0** | Enhanced individual training | Stronger expert training |
| **5.0** | Strong individual emphasis | Very strong expert training |
| **10.0** | Dominant individual training | Expert-focused training |

## 🔄 **Quick Ablation Workflow**

```bash
# 1. Test low lambda
sed -i 's/LAMBDA_LOSS=.*/LAMBDA_LOSS=0.1/' train_all_fusions_4gpu.sh
bash submit_fusion_training_4gpu.sh

# 2. Test balanced lambda
sed -i 's/LAMBDA_LOSS=.*/LAMBDA_LOSS=1.0/' train_all_fusions_4gpu.sh
bash submit_fusion_training_4gpu.sh

# 3. Test high lambda
sed -i 's/LAMBDA_LOSS=.*/LAMBDA_LOSS=5.0/' train_all_fusions_4gpu.sh
bash submit_fusion_training_4gpu.sh
```

## 📁 **Output Naming**

### **Directory Structure**
```
fusion_checkpoints/
├── densenet_fusions_lambda_0.1/     # λ = 0.1
├── densenet_fusions_lambda_1.0/     # λ = 1.0 (default)
├── densenet_fusions_lambda_5.0/     # λ = 5.0
└── csv_logs/
    └── densenet_fusions/
        ├── densenet_multiplicative_lambda_0.1_training_log.csv
        ├── densenet_multiplicative_lambda_1.0_training_log.csv
        └── densenet_multiplicative_lambda_5.0_training_log.csv
```

### **File Naming Convention**
- **Models**: `{model}_{fusion_type}_lambda_{lambda_value}_best.pth`
- **CSV Logs**: `{model}_{fusion_type}_lambda_{lambda_value}_training_log.csv`

## 📈 **Expected Results**

### **Performance Patterns**
- **Low λ (0.1-0.5)**: Better global fusion, weaker individual experts
- **Balanced λ (1.0)**: Good balance, stable performance  
- **High λ (2.0-10.0)**: Stronger individual experts, potential overfitting

### **Convergence**
- **Low λ**: Faster global fusion convergence
- **High λ**: Slower overall convergence, stronger expert training
- **Balanced λ**: Moderate convergence speed

## 🔍 **Monitoring**

### **Check Current Lambda**
```bash
grep "LAMBDA_LOSS=" train_all_fusions_4gpu.sh
```

### **Monitor Training**
```bash
# Check job status
squeue -u $USER

# Monitor logs
tail -f logs/train_all_fusions_4gpu_<JOB_ID>.out

# Check specific lambda
grep "lambda_0.1" logs/train_all_fusions_4gpu_<JOB_ID>.out
```

### **Check Results**
```bash
# List completed lambda studies
ls fusion_checkpoints/ | grep "lambda_"

# Check CSV logs
ls fusion_checkpoints/csv_logs/densenet_fusions/ | grep "lambda_"
```

## 🚀 **Pro Tips**

1. **Start with λ = 1.0** (balanced) as baseline
2. **Test λ = 0.1** for strong global fusion
3. **Test λ = 5.0** for strong individual experts
4. **Compare CSV logs** for performance analysis
5. **Use different λ per use case** (deployment vs research)

---

**Quick Lambda Tuning: Change value → Submit job → Monitor → Analyze results! 🎯**
