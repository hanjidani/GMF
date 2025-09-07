# DenseNet Fusion Training & Evaluation Framework (Alpha Ablation)

A specialized training and evaluation script for DenseNet-based Multiple Choice Networks (MCNs). This script is designed for conducting ablation studies on the `alpha` parameter of the dual-path loss function. It features a comprehensive, multi-phase evaluation pipeline and utilizes **fixed learning rates** to ensure a controlled experimental environment.

While the main framework focuses on adaptive learning rates for maximum performance, this script prioritizes experimental control to analyze the impact of the knowledge transfer parameter, `alpha`.

## 🔬 Core Concepts

### **1. Fixed Learning Rates for Controlled Ablations**

To isolate the impact of the `alpha` parameter, this script intentionally **omits learning rate schedulers**. The learning rates for the expert backbones, fusion module, and global head are kept constant throughout training. This ensures that any observed differences in performance can be attributed to the change in `alpha` rather than to complex learning rate dynamics.

- **Experts LR**: `base_lr * experts_lr_scale`
- **Fusion & Global Head LR**: `head_lr`

### **2. Dual-Path Loss Function**

The core of the training process is the dual-path loss, which balances learning between the collaborative (fused) path and the individual expert paths:

```python
# Combined loss for dual-path architecture
loss_total = loss_global + α * loss_individual
```
- `loss_global`: Cross-entropy loss from the fused output.
- `loss_individual`: Sum of cross-entropy losses from each expert's output.
- `α` (alpha): A hyperparameter that controls the contribution of the individual expert losses. This script is designed to run experiments by systematically varying this value.

### **3. Comprehensive Multi-Phase Evaluation**

The script integrates a rigorous, multi-phase evaluation protocol to provide a deep analysis of the model's capabilities before, during, and after training.

#### **Phase 1: Pre-Training Evaluation**
- **Purpose**: Establishes a performance baseline for the pre-trained experts and a benchmark model.
- **Evaluations**:
    - **Gaussian Noise Robustness**: Measures performance under varying levels of Gaussian noise.
    - **CIFAR-100-C Corruption Robustness**: Tests resilience against 19 common image corruptions at 5 severity levels.
    - **Out-of-Distribution (OOD) Detection**: Assesses the ability to distinguish in-distribution (CIFAR-100) from OOD data (e.g., SVHN, TinyImageNet).
- **Logging**: Results are saved to `..._pre_training_evaluation.csv`.

#### **Phase 2: Fusion Training & Component Checkpointing**
- **Purpose**: Trains the fusion model while continuously monitoring performance on a validation set.
- **Key Feature**: The script independently saves the best-performing state of each individual expert, the fusion module, and the global head whenever a new peak validation accuracy is achieved. This ensures that the post-training evaluation uses the optimal version of each component.
- **Logging**: Epoch-level training metrics are saved to `..._training_log.csv`.

#### **Phase 3: Post-Training Evaluation**
- **Purpose**: Measures the impact of fusion training on the individual experts and evaluates the final fused model.
- **Process**:
    1. The best-performing expert checkpoints saved during Phase 2 are loaded.
    2. These "best" experts and the final fusion model are re-evaluated on the same suite of Gaussian noise, corruption, and OOD benchmarks from Phase 1.
- **Logging**: Results are saved to `..._post_training_evaluation.csv`.

#### **Phase 4: Comprehensive Robustness Evaluation**
- **Purpose**: Runs a final, consolidated robustness analysis on the trained experts and fusion model.
- **Logging**: A summary of robustness metrics is saved to `..._robustness_evaluation.csv`.


## 🚀 How to Run

Execute the script from the `fusion_training/scripts/` directory. You must specify a `fusion_type` and can provide an `alpha` value to test.

```bash
python train_densenet_fusions_alpha_ablation_full_eval.py \
    --fusion_type TransformerBase \
    --alpha 0.5 \
    --epochs 100 \
    --batch_size 128 \
    --output_dir ../fusion_checkpoints_alpha_ablation_full_eval \
    --checkpoint_dir ../../expert_training/scripts/checkpoints_expert_iid \
    --base_lr 1e-4 \
    --head_lr 1e-3 \
    --use_train_val_split
```

### Key Arguments
- `--fusion_type`: The fusion mechanism to use (e.g., `multiplicative`, `TransformerBase`).
- `--alpha`: The weight for the individual expert loss component.
- `--output_dir`: Directory to save checkpoints and logs.
- `--checkpoint_dir`: Path to the pre-trained DenseNet expert models.
- `--base_lr`, `--head_lr`, `--experts_lr_scale`: Manually set the fixed learning rates.
- `--skip_pre_eval`: (Optional) Skip the lengthy pre-training evaluation phase.
- `--use_train_val_split`: Recommended. Creates a validation set from the training data for reliable checkpointing.

## 📁 Expected Output Structure

The script generates a structured output directory, making it easy to analyze results across different runs.

```
<output_dir>/
├── densenet_fusions_alpha_<alpha>/
│   └── <fusion_type>/                  # Checkpoints for a specific run
│       ├── experts_best/
│       │   ├── expert_0_best.pth
│       │   └── ...
│       ├── global_best/
│       │   └── global_head_best.pth
│       ├── fusion_best/
│       │   └── fusion_module_best.pth
│       └── best_model.pth              # Complete best model state
│
└── csv_logs/
    └── densenet_fusions/               # Aggregated CSV logs
        ├── <fusion_type>_alpha_<alpha>_pre_training_evaluation.csv
        ├── <fusion_type>_alpha_<alpha>_training_log.csv
        ├── <fusion_type>_alpha_<alpha>_post_training_evaluation.csv
        └── <fusion_type>_alpha_<alpha>_robustness_evaluation.csv
```
