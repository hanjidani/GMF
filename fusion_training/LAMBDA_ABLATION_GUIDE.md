# 🎯 **Lambda Ablation Study Guide**

## 📋 **Overview**

This guide explains how to conduct **lambda ablation studies** with the 4-GPU parallel fusion training system. Lambda (λ) controls the balance between global fusion and individual expert objectives in the dual-path architecture.

## 🧠 **Understanding Lambda in Dual-Path Loss**

### **Dual-Path Loss Function**
```python
# Combined loss for dual-path architecture
loss_total = loss_global + λ * loss_individual

# Where:
# - loss_global: Cross-entropy on fused features (collaborative path)
# - loss_individual: Sum of individual expert losses (individual path)
# - λ: Balance parameter controlling the trade-off
```

### **Lambda Effects on Training**

| Lambda Value | Effect on Training | Expected Outcome |
|--------------|-------------------|------------------|
| **λ = 0.1** | Minimal individual expert training | Strong global fusion, weak individual experts |
| **λ = 0.5** | Reduced individual emphasis | Balanced but favor global fusion |
| **λ = 1.0** | Equal balance (default) | Good balance between global and individual |
| **λ = 2.0** | Enhanced individual training | Stronger individual experts, good fusion |
| **λ = 5.0** | Strong individual emphasis | Very strong individual experts |
| **λ = 10.0** | Dominant individual training | Individual experts dominate, potential overfitting |

## 🚀 **Running Lambda Ablation Studies**

### **Step 1: Configure Lambda Values**

Edit `train_all_fusions_4gpu.sh` and change the `LAMBDA_LOSS` variable:

```bash
# For λ = 0.1 (minimal individual training)
LAMBDA_LOSS=0.1

# For λ = 0.5 (reduced individual emphasis)
LAMBDA_LOSS=0.5

# For λ = 1.0 (balanced - default)
LAMBDA_LOSS=1.0

# For λ = 2.0 (enhanced individual training)
LAMBDA_LOSS=2.0

# For λ = 5.0 (strong individual emphasis)
LAMBDA_LOSS=5.0

# For λ = 10.0 (dominant individual training)
LAMBDA_LOSS=10.0
```

### **Step 2: Submit Training Jobs**

```bash
cd fusion_training/scripts

# Submit job with current lambda value
bash submit_fusion_training_4gpu.sh
```

### **Step 3: Repeat for Different Lambda Values**

```bash
# Edit lambda value
sed -i 's/LAMBDA_LOSS=1.0/LAMBDA_LOSS=0.1/' train_all_fusions_4gpu.sh

# Submit new job
bash submit_fusion_training_4gpu.sh

# Repeat for other values
sed -i 's/LAMBDA_LOSS=0.1/LAMBDA_LOSS=2.0/' train_all_fusions_4gpu.sh
bash submit_fusion_training_4gpu.sh
```

## 📊 **Expected Directory Structure**

### **Lambda-Specific Output Organization**
```
fusion_checkpoints/
├── densenet_fusions_lambda_0.1/
│   ├── multiplicative/
│   │   ├── experts/densenet_expert_0_lambda_0.1_best.pth
│   │   ├── fusion/densenet_multiplicative_lambda_0.1_fusion_best.pth
│   │   └── global/densenet_multiplicative_lambda_0.1_global_head_best.pth
│   └── ...
├── densenet_fusions_lambda_1.0/
│   ├── multiplicative/
│   │   ├── experts/densenet_expert_0_lambda_1.0_best.pth
│   │   ├── fusion/densenet_multiplicative_lambda_1.0_fusion_best.pth
│   │   └── global/densenet_multiplicative_lambda_1.0_global_head_best.pth
│   └── ...
├── densenet_fusions_lambda_5.0/
│   └── ...
└── csv_logs/
    ├── densenet_fusions/
    │   ├── densenet_multiplicative_lambda_0.1_training_log.csv
    │   ├── densenet_multiplicative_lambda_1.0_training_log.csv
    │   └── densenet_multiplicative_lambda_5.0_training_log.csv
    └── ...
```

### **CSV Log Naming Convention**
- **Filename**: `{model}_{fusion_type}_lambda_{lambda_value}_training_log.csv`
- **Examples**:
  - `densenet_multiplicative_lambda_0.1_training_log.csv`
  - `resnet_multiplicativeAddition_lambda_2.0_training_log.csv`
  - `improved_wide_resnet_TransformerBase_lambda_5.0_training_log.csv`

## 📈 **Analyzing Lambda Ablation Results**

### **1. Performance Comparison**

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load training logs for different lambda values
lambda_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
results = {}

for lambda_val in lambda_values:
    csv_path = f'fusion_checkpoints/csv_logs/densenet_fusions/densenet_multiplicative_lambda_{lambda_val}_training_log.csv'
    try:
        df = pd.read_csv(csv_path)
        results[lambda_val] = {
            'best_accuracy': df['val_accuracy'].max(),
            'final_accuracy': df['val_accuracy'].iloc[-1],
            'convergence_epoch': df.loc[df['val_accuracy'] == df['val_accuracy'].max(), 'epoch'].iloc[0]
        }
    except FileNotFoundError:
        print(f"Log file not found for λ = {lambda_val}")

# Create comparison DataFrame
comparison_df = pd.DataFrame(results).T
print("Lambda Ablation Results:")
print(comparison_df)
```

### **2. Visualization**

```python
# Plot accuracy vs lambda
plt.figure(figsize=(10, 6))
plt.plot(comparison_df.index, comparison_df['best_accuracy'], 'bo-', linewidth=2, markersize=8)
plt.xlabel('Lambda (λ)')
plt.ylabel('Best Validation Accuracy (%)')
plt.title('Lambda Ablation Study: Accuracy vs Lambda')
plt.grid(True, alpha=0.3)
plt.xticks(comparison_df.index)
plt.show()

# Plot convergence speed vs lambda
plt.figure(figsize=(10, 6))
plt.plot(comparison_df.index, comparison_df['convergence_epoch'], 'ro-', linewidth=2, markersize=8)
plt.xlabel('Lambda (λ)')
plt.ylabel('Convergence Epoch')
plt.title('Lambda Ablation Study: Convergence Speed vs Lambda')
plt.grid(True, alpha=0.3)
plt.xticks(comparison_df.index)
plt.show()
```

### **3. Loss Analysis**

```python
# Analyze loss components for different lambda values
def analyze_loss_components(lambda_val):
    csv_path = f'fusion_checkpoints/csv_logs/densenet_fusions/densenet_multiplicative_lambda_{lambda_val}_training_log.csv'
    df = pd.read_csv(csv_path)
    
    # Calculate average loss components
    avg_global_loss = df['loss_global'].mean()
    avg_individual_loss = df['loss_individual'].mean()
    avg_total_loss = df['loss_total'].mean()
    
    return {
        'lambda': lambda_val,
        'avg_global_loss': avg_global_loss,
        'avg_individual_loss': avg_individual_loss,
        'avg_total_loss': avg_total_loss,
        'global_individual_ratio': avg_global_loss / avg_individual_loss
    }

# Analyze all lambda values
loss_analysis = [analyze_loss_components(lambda_val) for lambda_val in lambda_values]
loss_df = pd.DataFrame(loss_analysis)
print("Loss Component Analysis:")
print(loss_df)
```

## 🎯 **Recommended Lambda Values for Ablation**

### **Primary Values to Test**
1. **λ = 0.1** - Minimal individual training
2. **λ = 0.5** - Reduced individual emphasis
3. **λ = 1.0** - Balanced (baseline)
4. **λ = 2.0** - Enhanced individual training
5. **λ = 5.0** - Strong individual emphasis

### **Extended Range (if needed)**
6. **λ = 0.2** - Very low individual emphasis
7. **λ = 0.8** - Slightly reduced individual emphasis
8. **λ = 3.0** - Moderate individual emphasis
9. **λ = 10.0** - Dominant individual training

## 📋 **Ablation Study Workflow**

### **Complete Ablation Study**
```bash
# 1. Start with balanced lambda
sed -i 's/LAMBDA_LOSS=.*/LAMBDA_LOSS=1.0/' train_all_fusions_4gpu.sh
bash submit_fusion_training_4gpu.sh

# 2. Test low lambda values
sed -i 's/LAMBDA_LOSS=1.0/LAMBDA_LOSS=0.1/' train_all_fusions_4gpu.sh
bash submit_fusion_training_4gpu.sh

sed -i 's/LAMBDA_LOSS=0.1/LAMBDA_LOSS=0.5/' train_all_fusions_4gpu.sh
bash submit_fusion_training_4gpu.sh

# 3. Test high lambda values
sed -i 's/LAMBDA_LOSS=0.5/LAMBDA_LOSS=2.0/' train_all_fusions_4gpu.sh
bash submit_fusion_training_4gpu.sh

sed -i 's/LAMBDA_LOSS=2.0/LAMBDA_LOSS=5.0/' train_all_fusions_4gpu.sh
bash submit_fusion_training_4gpu.sh

# 4. Return to baseline
sed -i 's/LAMBDA_LOSS=5.0/LAMBDA_LOSS=1.0/' train_all_fusions_4gpu.sh
```

### **Automated Ablation Script**
```bash
#!/bin/bash
# automated_lambda_ablation.sh

lambda_values=(0.1 0.5 1.0 2.0 5.0)

for lambda in "${lambda_values[@]}"; do
    echo "Running ablation study with λ = $lambda"
    
    # Update lambda value
    sed -i "s/LAMBDA_LOSS=.*/LAMBDA_LOSS=$lambda/" train_all_fusions_4gpu.sh
    
    # Submit job
    JOB_ID=$(sbatch train_all_fusions_4gpu.sh | awk '{print $4}')
    echo "Job submitted with ID: $JOB_ID"
    
    # Wait for completion (optional)
    echo "Waiting for job completion..."
    while squeue -u $USER | grep -q $JOB_ID; do
        sleep 60
    done
    
    echo "Job $JOB_ID completed for λ = $lambda"
    echo "----------------------------------------"
done

echo "All lambda ablation studies completed!"
```

## 🔍 **Monitoring Ablation Studies**

### **Real-time Monitoring**
```bash
# Monitor all lambda studies
watch -n 10 'squeue -u $USER && echo "---" && ls -la fusion_checkpoints/'

# Check specific lambda study
grep "lambda_0.1" logs/train_all_fusions_4gpu_<JOB_ID>.out | tail -10
```

### **Progress Tracking**
```bash
# Check which lambda values have been completed
ls fusion_checkpoints/ | grep "lambda_"

# Check CSV log completion
ls fusion_checkpoints/csv_logs/densenet_fusions/ | grep "lambda_"
```

## 📊 **Expected Results Analysis**

### **Performance Patterns**
- **Low λ (0.1-0.5)**: Better global fusion, weaker individual experts
- **Balanced λ (1.0)**: Good balance, stable performance
- **High λ (2.0-10.0)**: Stronger individual experts, potential overfitting

### **Convergence Analysis**
- **Low λ**: Faster global fusion convergence, slower individual expert convergence
- **High λ**: Slower overall convergence, stronger individual expert training
- **Balanced λ**: Moderate convergence speed, good balance

### **Loss Component Analysis**
- **Global Loss**: Should decrease with training
- **Individual Loss**: Should decrease with training
- **Total Loss**: Should decrease and stabilize
- **Loss Ratio**: Global/Individual ratio should be reasonable

## 🎉 **Benefits of Lambda Ablation**

### **1. Understanding Model Behavior**
- **Trade-off Analysis**: Balance between global and individual objectives
- **Performance Optimization**: Find optimal lambda for your specific use case
- **Overfitting Prevention**: Identify lambda values that cause overfitting

### **2. Research Insights**
- **Dual-Path Architecture**: Understand how different paths contribute to learning
- **Expert-Fusion Interaction**: Analyze how experts and fusion modules interact
- **Loss Function Design**: Optimize the combined loss function

### **3. Practical Applications**
- **Deployment Flexibility**: Choose lambda based on deployment requirements
- **Resource Optimization**: Balance training time vs performance
- **Model Selection**: Select best lambda for production use

---

**Ready to start your lambda ablation study? Follow the workflow above and discover the optimal balance between global fusion and individual expert training! 🚀**
