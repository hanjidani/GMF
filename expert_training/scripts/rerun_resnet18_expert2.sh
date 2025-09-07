#!/usr/bin/env bash

# Rerun ResNet-18 Expert 2 independently (Non-IID) due to CUDA ECC error
#SBATCH -J ResNet18_E2
#SBATCH -p volta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH -o logs/resnet18_expert2_rerun_%j.out

set -eo pipefail
exec 2>&1
mkdir -p logs

echo "=========================================="
echo "Rerunning ResNet-18 Expert 2 independently (Non-IID)"
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
echo "SLURM_NODELIST = ${SLURM_NODELIST}"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Conda env activation
source /home/ali.rasekh/miniconda3/envs/newenv/bin/activate /home/ali.rasekh/miniconda3/envs/newenv/envs/orm
conda activate orm

# Let Slurm control GPU visibility per task
unset CUDA_VISIBLE_DEVICES || true

# Non-IID training (no shared/unique ratios needed)
echo "Non-IID training: Expert 2 will see classes 50-74 (25 classes)"

echo "Python path: $(which python3)"
echo "Starting ResNet-18 Expert 2 training (Non-IID)..."

cd /home/ali.rasekh/orm/hos/geom/Fianl_MCN/expert_training/scripts

# Add CUDA debugging environment variables
export CUDA_LAUNCH_BLOCKING=1

python3 -u train_noniid_experts.py \
    --model resnet18 \
    --expert_id 2 \
    --no_wandb

echo "🏁 ResNet-18 Expert 2 training (Non-IID) completed!"
