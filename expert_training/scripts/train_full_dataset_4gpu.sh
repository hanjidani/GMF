#!/usr/bin/env bash

# 3-GPU Parallel Full Dataset Benchmark Training - Non-WideResNeXt models
# Based on train_iid_16gpu.sh structure for pascal-node10
#SBATCH -J FullBench3
#SBATCH -p volta
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH -o logs/train_full_dataset_3gpu_%j.out

set -eo pipefail
exec 2>&1
mkdir -p logs

# Conda env activation
source /home/ali.rasekh/miniconda3/envs/newenv/bin/activate /home/ali.rasekh/miniconda3/envs/newenv/envs/orm
conda activate orm

# Let Slurm control GPU visibility per task
unset CUDA_VISIBLE_DEVICES || true

echo "=========================================="
echo "4-GPU Full Dataset Benchmark Training"
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
echo "SLURM_NODELIST = ${SLURM_NODELIST}"
echo "Node: pascal-node10 (4x V100)"
echo "=========================================="

# Launch 3 tasks with model assignment logic
srun --ntasks=3 bash -c '
    # Determine model based on task ID
    TASK_ID=${SLURM_PROCID}
    
    if [ $TASK_ID -eq 0 ]; then
        MODEL="resnet18"
    elif [ $TASK_ID -eq 1 ]; then
        MODEL="densenet121"  
    else
        MODEL="wideresnet28_10"
    fi
    
    echo "=========================================="
    echo "Task $TASK_ID: $MODEL Full Dataset Benchmark"
    echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
    echo "SLURM_NODELIST = ${SLURM_NODELIST}"
    echo "GPU: $CUDA_VISIBLE_DEVICES"
    echo "=========================================="
    
    which python3 || true
    nvidia-smi || true
    
    # Change to the correct directory and run training
    cd /home/ali.rasekh/orm/hos/geom/Fianl_MCN/expert_training/scripts
    
    echo "Starting full dataset benchmark training: $MODEL"
    echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
    
    python3 -u train_full_dataset_benchmark.py \
        --model "$MODEL" \
        --no_wandb \
        2>&1 | sed -e "s/^/[task-$TASK_ID-$MODEL-full-benchmark] /"
'

echo "🏁 All 3 full dataset benchmark training completed!"
