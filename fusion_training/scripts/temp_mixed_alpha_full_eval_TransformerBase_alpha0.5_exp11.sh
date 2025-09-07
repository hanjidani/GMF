#!/usr/bin/env bash

# Temporary Slurm script for MIXED-EXPERTS ALPHA ABLATION (FULL EVAL) experiment 11
# Fusion: TransformerBase, Alpha: 0.5, Idxs: DN=1 RN=0 WRN=0 RX=0

#SBATCH -J MIXEDTRANSFORMERBASE0.5ABLFE
#SBATCH -p volta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o /home/ali.rasekh/orm/hos/geom/Fianl_MCN/fusion_training/scripts/logs_mixed_alpha_ablation_full_eval/mixed_TransformerBase_alpha0.5_1_0_0_0_exp11_%j.log
#SBATCH -e /home/ali.rasekh/orm/hos/geom/Fianl_MCN/fusion_training/scripts/logs_mixed_alpha_ablation_full_eval/mixed_TransformerBase_alpha0.5_1_0_0_0_exp11_%j.log

set -eo pipefail
exec 2>&1

# Conda env activation
source /home/ali.rasekh/miniconda3/envs/newenv/bin/activate /home/ali.rasekh/miniconda3/envs/newenv/envs/orm
conda activate orm

echo "=========================================="
echo "Mixed-Experts Alpha Ablation (Full Eval) Experiment 11: TransformerBase fusion, alpha=0.5"
echo "SLURM_JOB_ID = "
echo "SLURM_NODELIST = "
echo "Alpha: 0.5"
echo "Augmentation: CutMix (α=1.0) + Label Smoothing (0.1) + Grad Clip (1.0)"
echo "No LR schedulers (fixed LR); Full evaluation enabled"
echo "=========================================="

# Change to the correct directory
cd /home/ali.rasekh/orm/hos/geom/Fianl_MCN/fusion_training/scripts

# Run fusion training (alpha ablation full-eval, no schedulers)
python3 -u train_mixed_experts_fusions_alpha_ablation_full_eval.py     --fusion_type "TransformerBase"     --alpha 0.5     --checkpoint_dir ../../expert_training/scripts/checkpoints_expert_iid     --output_dir ../fusion_checkpoints_mixed_alpha_ablation_full_eval     --data_dir ../data     --epochs 40     --batch_size 128     --seed 42     --augmentation_mode cutmix     --mixup_alpha 0.2     --cutmix_alpha 1.0     --label_smoothing 0.1     --gradient_clip_norm 1.0     --base_lr 1e-4     --head_lr 1e-3     --experts_lr_scale 0.1     --densenet_idx 1     --resnet_idx 0     --wideresnet_idx 0     --resnext_idx 0

echo "✅ Experiment 11 completed: MIXED TransformerBase fusion, alpha=0.5 (Alpha Ablation Full Eval)"
