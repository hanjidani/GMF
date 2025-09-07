# 🔍 **Expert Evaluation Quick Reference**

## ⚡ **Quick Start**

### **Run Expert Evaluation**
```bash
cd fusion_training/scripts
bash evaluate_experts_simple.sh
```

### **Or Run Directly**
```bash
cd fusion_training/scripts
python3 evaluate_experts.py
```

## 📊 **What Gets Evaluated**

- ✅ **4 Model Architectures**: DenseNet, ResNet, Improved WideResNet, ResNeXt
- ✅ **4 Experts per Model**: expert_0_best.pth through expert_3_best.pth
- ✅ **CIFAR-100 Test Set**: 10,000 test images
- ✅ **Performance Metrics**: Accuracy, Loss, Correct/Total predictions

## 📁 **Expected Checkpoints**

```
expert_training/checkpoints/
├── densenet/expert_0_best.pth, expert_1_best.pth, expert_2_best.pth, expert_3_best.pth
├── resnet/expert_0_best.pth, expert_1_best.pth, expert_2_best.pth, expert_3_best.pth
├── improved_wide_resnet/expert_0_best.pth, expert_1_best.pth, expert_2_best.pth, expert_3_best.pth
└── resnext/expert_0_best.pth, expert_1_best.pth, expert_2_best.pth, expert_3_best.pth
```

## 📈 **Output Files**

### **1. Detailed Results** (`expert_evaluation_results.csv`)
```csv
model,expert_id,accuracy,loss,correct,total,timestamp
densenet,0,78.45,0.89,7845,10000,2024-08-17 00:51:23
densenet,1,79.12,0.87,7912,10000,2024-08-17 00:51:23
resnet,0,75.67,0.95,7567,10000,2024-08-17 00:51:23
...
```

### **2. Model Summary** (`expert_evaluation_summary.csv`)
```csv
model,num_experts,avg_accuracy,std_accuracy,avg_loss,min_accuracy,max_accuracy
densenet,4,78.45,2.31,0.89,76.12,81.23
resnet,4,75.67,1.89,0.95,73.45,77.89
...
```

## 🎯 **Quality Thresholds**

| Performance | Accuracy | Recommendation |
|-------------|----------|----------------|
| **Excellent** | ≥ 80% | ✅ Ready for fusion! |
| **Good** | 75-80% | ✅ Ready for fusion |
| **Acceptable** | 70-75% | ⚠️ Consider retraining |
| **Poor** | < 70% | ❌ Retrain experts |

## 🔍 **Troubleshooting**

### **Missing Checkpoints**
```bash
# Check if checkpoints exist
ls -la ../expert_training/checkpoints/densenet/expert_*_best.pth

# Expected: 4 checkpoint files
```

### **CUDA Memory Issues**
```bash
# Reduce batch size
python3 evaluate_experts.py --batch_size 64
```

### **Model Import Errors**
```bash
# Check model files exist
ls -la ../expert_training/models/
```

## 🚀 **After Evaluation**

### **If Experts Pass Quality Check**
```bash
# Proceed with fusion training
bash submit_fusion_training_4gpu.sh
```

### **If Issues Found**
```bash
# Review evaluation results
cat ../fusion_checkpoints/expert_evaluation/expert_evaluation_summary.csv

# Fix issues and re-evaluate
bash evaluate_experts_simple.sh
```

## 📋 **Evaluation Checklist**

- [ ] All 16 expert checkpoints exist (4 models × 4 experts)
- [ ] All experts load without errors
- [ ] All experts achieve >70% accuracy
- [ ] Low variance across experts (<6% std dev)
- [ ] CSV results saved successfully
- [ ] Ready to proceed with fusion training

---

**Quick Evaluation: Run `bash evaluate_experts_simple.sh` → Check CSV results → Start fusion training! 🚀**
