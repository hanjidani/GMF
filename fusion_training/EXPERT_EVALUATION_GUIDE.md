# 🔍 **Expert Model Evaluation Guide**

## 📋 **Overview**

Before starting fusion training, it's crucial to **evaluate all expert models** to ensure they're working properly and establish baseline performance. This guide explains how to use the expert evaluation system to verify your experts are ready for fusion.

## 🎯 **Why Evaluate Experts First?**

### **1. Quality Assurance**
- **Verify Checkpoints**: Ensure all expert checkpoints load correctly
- **Performance Baseline**: Establish baseline accuracy for comparison
- **Consistency Check**: Verify experts have similar performance levels
- **Error Detection**: Catch issues before starting expensive fusion training

### **2. Fusion Training Prerequisites**
- **Expert Quality**: Poor experts lead to poor fusion results
- **Performance Gap**: Understand the improvement potential from fusion
- **Resource Planning**: Estimate training time and resource requirements
- **Troubleshooting**: Identify and fix issues early

## 🚀 **Quick Start Evaluation**

### **Option 1: Run the Convenience Script**
```bash
cd fusion_training/scripts
bash run_expert_evaluation.sh
```

### **Option 2: Run Python Script Directly**
```bash
cd fusion_training/scripts
python3 evaluate_experts.py \
    --data_dir "../data" \
    --output_dir "../fusion_checkpoints" \
    --batch_size 128 \
    --num_workers 4 \
    --seed 42
```

### **Option 3: Custom Evaluation**
```bash
cd fusion_training/scripts
python3 evaluate_experts.py \
    --models densenet resnet \
    --expert_dirs "../expert_training/checkpoints/densenet" "../expert_training/checkpoints/resnet" \
    --batch_size 64 \
    --output_dir "../fusion_checkpoints"
```

## 📊 **What Gets Evaluated**

### **1. Model Loading**
- ✅ **Checkpoint Loading**: Verify all expert checkpoints load without errors
- ✅ **Model Creation**: Ensure models are created with correct architecture
- ✅ **Device Placement**: Models are properly moved to GPU/CPU
- ✅ **State Dict**: Checkpoint weights are correctly applied

### **2. Performance Metrics**
- 📈 **Accuracy**: Top-1 accuracy on CIFAR-100 test set
- 📉 **Loss**: Cross-entropy loss on test set
- 🎯 **Per-Class Accuracy**: Performance breakdown by class
- 📊 **Consistency**: Variance across experts of same architecture

### **3. Model Behavior**
- 🔍 **Forward Pass**: Verify models produce valid outputs
- 📏 **Output Shape**: Confirm logits have correct dimensions
- 🧠 **Feature Extraction**: Check if models return features + logits
- ⚡ **Inference Speed**: Basic performance benchmarking

## 📁 **Expected Directory Structure**

### **Before Evaluation**
```
expert_training/checkpoints/
├── densenet/
│   ├── expert_0_best.pth
│   ├── expert_1_best.pth
│   ├── expert_2_best.pth
│   └── expert_3_best.pth
├── resnet/
│   ├── expert_0_best.pth
│   ├── expert_1_best.pth
│   ├── expert_2_best.pth
│   └── expert_3_best.pth
├── improved_wide_resnet/
│   └── ...
└── resnext/
    └── ...
```

### **After Evaluation**
```
fusion_checkpoints/expert_evaluation/
├── overall_summary.csv              # Summary of all models
├── densenet_expert_summary.csv      # DenseNet expert results
├── densenet_per_class_accuracy.csv  # Per-class performance
├── densenet_expert_evaluation.json  # Detailed results
├── resnet_expert_summary.csv        # ResNet expert results
├── resnet_per_class_accuracy.csv    # Per-class performance
├── resnet_expert_evaluation.json    # Detailed results
└── ... (similar for other models)
```

## 📈 **Understanding Evaluation Results**

### **1. Overall Summary (`overall_summary.csv`)**
```csv
model,num_experts,avg_accuracy,std_accuracy,avg_loss,min_accuracy,max_accuracy
densenet,4,78.45,2.31,0.89,76.12,81.23
resnet,4,75.67,1.89,0.95,73.45,77.89
improved_wide_resnet,4,82.34,1.45,0.76,80.89,84.12
resnext,4,80.12,2.78,0.83,77.34,83.45
```

### **2. Per-Model Summary (`{model}_expert_summary.csv`)**
```csv
expert_id,accuracy,loss,correct,total
expert_0,78.45,0.89,7845,10000
expert_1,79.12,0.87,7912,10000
expert_2,77.89,0.91,7789,10000
expert_3,78.34,0.88,7834,10000
```

### **3. Per-Class Accuracy (`{model}_per_class_accuracy.csv`)**
```csv
class_id,expert_0,expert_1,expert_2,expert_3
0,85.2,84.7,86.1,85.8
1,72.3,73.1,71.9,72.8
2,91.5,90.8,92.1,91.3
...
```

## 🎯 **Quality Thresholds & Recommendations**

### **Accuracy Thresholds**
| Performance Level | Accuracy Range | Recommendation |
|------------------|----------------|----------------|
| **Excellent** | ≥ 80% | ✅ Ready for fusion training |
| **Good** | 75-80% | ✅ Ready for fusion training |
| **Acceptable** | 70-75% | ⚠️ Consider retraining experts |
| **Poor** | < 70% | ❌ Retrain experts before fusion |

### **Consistency Thresholds**
| Variance Level | Std Dev Range | Recommendation |
|----------------|---------------|----------------|
| **Excellent** | ≤ 2% | ✅ Very consistent experts |
| **Good** | 2-4% | ✅ Consistent enough for fusion |
| **Acceptable** | 4-6% | ⚠️ Some inconsistency, monitor closely |
| **Poor** | > 6% | ❌ High variance, investigate issues |

### **Example Analysis**
```python
# Good performance example
densenet: 78.45% ± 2.31%  # ✅ Good accuracy, low variance
resnet: 75.67% ± 1.89%    # ✅ Good accuracy, low variance

# Problematic performance example  
improved_wide_resnet: 82.34% ± 8.45%  # ⚠️ Good accuracy, high variance
resnext: 65.23% ± 2.34%               # ❌ Poor accuracy, investigate
```

## 🔧 **Troubleshooting Common Issues**

### **1. Checkpoint Loading Errors**
```bash
# Check if checkpoint files exist
ls -la ../expert_training/checkpoints/densenet/

# Verify checkpoint format
python3 -c "
import torch
checkpoint = torch.load('../expert_training/checkpoints/densenet/expert_0_best.pth')
print('Keys:', checkpoint.keys() if isinstance(checkpoint, dict) else 'Direct state dict')
"
```

### **2. Model Import Errors**
```bash
# Check model file paths
ls -la ../expert_training/models/

# Test model creation
python3 -c "
import sys
sys.path.append('../expert_training/models')
from densenet_cifar import densenet121
model = densenet121(num_classes=100)
print('Model created successfully')
"
```

### **3. CUDA Memory Issues**
```bash
# Reduce batch size
python3 evaluate_experts.py --batch_size 64

# Check GPU memory
nvidia-smi

# Use CPU if needed
CUDA_VISIBLE_DEVICES="" python3 evaluate_experts.py
```

### **4. Data Loading Issues**
```bash
# Check data directory
ls -la ../data/

# Verify CIFAR-100 download
python3 -c "
import torchvision
dataset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True)
print(f'Dataset size: {len(dataset)}')
"
```

## 📊 **Advanced Analysis**

### **1. Performance Comparison**
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load overall summary
summary_df = pd.read_csv('../fusion_checkpoints/expert_evaluation/overall_summary.csv')

# Plot accuracy comparison
plt.figure(figsize=(10, 6))
plt.bar(summary_df['model'], summary_df['avg_accuracy'])
plt.errorbar(summary_df['model'], summary_df['avg_accuracy'], 
             yerr=summary_df['std_accuracy'], fmt='none', capsize=5)
plt.ylabel('Accuracy (%)')
plt.title('Expert Model Performance Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### **2. Expert Consistency Analysis**
```python
# Load per-model results
densenet_df = pd.read_csv('../fusion_checkpoints/expert_evaluation/densenet_expert_summary.csv')

# Calculate consistency metrics
accuracies = densenet_df['accuracy']
print(f"Mean: {accuracies.mean():.2f}%")
print(f"Std: {accuracies.std():.2f}%")
print(f"Range: {accuracies.max() - accuracies.min():.2f}%")
print(f"Coefficient of Variation: {accuracies.std()/accuracies.mean()*100:.1f}%")
```

### **3. Class Performance Analysis**
```python
# Load per-class results
class_df = pd.read_csv('../fusion_checkpoints/expert_evaluation/densenet_per_class_accuracy.csv')

# Find best and worst performing classes
class_df['avg_accuracy'] = class_df[['expert_0', 'expert_1', 'expert_2', 'expert_3']].mean(axis=1)

best_classes = class_df.nlargest(10, 'avg_accuracy')
worst_classes = class_df.nsmallest(10, 'avg_accuracy')

print("Best performing classes:")
print(best_classes[['class_id', 'avg_accuracy']])

print("\nWorst performing classes:")
print(worst_classes[['class_id', 'avg_accuracy']])
```

## 🚀 **Next Steps After Evaluation**

### **1. If All Experts Pass Quality Checks**
```bash
# Proceed with fusion training
cd fusion_training/scripts
bash submit_fusion_training_4gpu.sh
```

### **2. If Some Experts Need Improvement**
```bash
# Retrain specific experts
cd ../expert_training
# Follow expert training procedures for problematic models
```

### **3. If Major Issues Found**
```bash
# Investigate and fix root causes
# Check training logs, data quality, model architecture
# Consider different model configurations
```

## 📋 **Evaluation Checklist**

- [ ] **Checkpoint Verification**: All expert checkpoints load without errors
- [ ] **Performance Baseline**: All experts achieve >70% accuracy
- [ ] **Consistency Check**: Standard deviation <6% across experts
- [ ] **Model Behavior**: Forward pass works correctly
- [ ] **Output Quality**: Logits have correct shape and values
- [ ] **Resource Check**: Sufficient GPU memory for evaluation
- [ ] **Data Access**: CIFAR-100 test set loads correctly
- [ ] **Results Analysis**: Review all evaluation outputs
- [ ] **Quality Decision**: Determine if experts are ready for fusion

## 🎉 **Benefits of Expert Evaluation**

✅ **Quality Assurance**: Ensure only good experts proceed to fusion  
✅ **Performance Baseline**: Understand improvement potential  
✅ **Early Detection**: Catch issues before expensive training  
✅ **Resource Planning**: Estimate fusion training requirements  
✅ **Confidence Building**: Verify system is working correctly  

---

**Ready to evaluate your experts? Run `bash run_expert_evaluation.sh` and ensure your fusion training starts with high-quality expert models! 🔍✨**
