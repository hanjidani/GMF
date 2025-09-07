#!/usr/bin/env bash

# 4-GPU Distributed Training for resnext29_8x64d - Single model across 4 GPUs
# Based on train_full_dataset_4gpu.sh structure for pascal-node10
#SBATCH -J ResNeXtDist4
#SBATCH -p volta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH -o logs/train_resnext_distributed_4gpu_%j.out

set -eo pipefail
exec 2>&1
mkdir -p logs

# Conda env activation
source /home/ali.rasekh/miniconda3/envs/newenv/bin/activate /home/ali.rasekh/miniconda3/envs/newenv/envs/orm
conda activate orm

# Let Slurm control GPU visibility per task
unset CUDA_VISIBLE_DEVICES || true

echo "=========================================="
echo "4-GPU Distributed Training for resnext29_8x64d"
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
echo "SLURM_NODELIST = ${SLURM_NODELIST}"
echo "Node: pascal-node10 (4x V100)"
echo "=========================================="

# Launch distributed training with single task (the script handles multi-GPU internally)
srun --ntasks=1 bash -c '
    MODEL="resnext29_8x64d"
    
    echo "=========================================="
    echo "Distributed training for $MODEL across 4 GPUs"
    echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
    echo "SLURM_NODELIST = ${SLURM_NODELIST}"
    echo "=========================================="
    
    which python3 || true
    nvidia-smi || true
    
    # Change to the correct directory and run distributed training
    cd /home/ali.rasekh/orm/hos/geom/Fianl_MCN/expert_training/scripts
    
    echo "Starting distributed training for $MODEL across 4 GPUs"
    
    # Use the built-in distributed training functionality
    python3 -u train_full_dataset_benchmark.py \
        --model "$MODEL" \
        --distributed \
        --world_size 4 \
        --no_wandb
'

echo "🏁 Distributed training for resnext29_8x64d across 4 GPUs completed!"
