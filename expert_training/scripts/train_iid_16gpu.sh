#!/usr/bin/env bash

# 16-GPU Parallel IID Expert Training - All models simultaneously
# Based on train_iid.sh structure for pascal-node10
#SBATCH -J IID16
#SBATCH -p volta
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH -o logs/train_iid_16gpu_%j.out

set -eo pipefail
exec 2>&1
mkdir -p logs

# Conda env activation
source /home/ali.rasekh/miniconda3/envs/newenv/bin/activate /home/ali.rasekh/miniconda3/envs/newenv/envs/orm
conda activate orm

# Let Slurm control GPU visibility per task
unset CUDA_VISIBLE_DEVICES || true

# IID ratios for shared/unique splits
export SHARED_RATIO="${SHARED_RATIO:-0.40}"
export UNIQUE_RATIO="${UNIQUE_RATIO:-0.15}"

# Define model assignment for 16 tasks
# Tasks 0-3: ResNet-18 (Experts 0-3)
# Tasks 4-7: DenseNet-121 (Experts 0-3)  
# Tasks 8-11: WideResNet-28-10 (Experts 0-3)
# Tasks 12-15: PreAct-ResNeXt-29 (Experts 0-3)

echo "=========================================="
echo "16-GPU IID Expert Training"
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
echo "SLURM_NODELIST = ${SLURM_NODELIST}"
echo "Node: pascal-node10 (16x V100)"
echo "=========================================="

# Launch 16 tasks with model assignment logic
srun --ntasks=16 bash -c '
    # Determine model and expert based on task ID
    TASK_ID=${SLURM_PROCID}
    
    if [ $TASK_ID -lt 4 ]; then
        MODEL="resnet18"
        EXPERT_ID=$TASK_ID
    elif [ $TASK_ID -lt 8 ]; then
        MODEL="densenet121"  
        EXPERT_ID=$((TASK_ID - 4))
    elif [ $TASK_ID -lt 12 ]; then
        MODEL="wideresnet28_10"
        EXPERT_ID=$((TASK_ID - 8))
    else
        MODEL="preact_resnext29_8x64d"
        EXPERT_ID=$((TASK_ID - 12))
    fi
    
    echo "=========================================="
    echo "Task $TASK_ID: $MODEL Expert $EXPERT_ID"
    echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
    echo "SLURM_NODELIST = ${SLURM_NODELIST}"
    echo "GPU: $CUDA_VISIBLE_DEVICES"
    echo "=========================================="
    
    which python3 || true
    nvidia-smi || true
    
    # Change to the correct directory and run training
    cd /home/ali.rasekh/orm/hos/geom/Fianl_MCN/expert_training/scripts
    
    echo "Starting training: $MODEL Expert $EXPERT_ID"
    echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
    
    python3 -u train_iid_experts.py \
        --model "$MODEL" \
        --expert_id $EXPERT_ID \
        --no_wandb \
        2>&1 | sed -e "s/^/[task-$TASK_ID-$MODEL-expert$EXPERT_ID] /"
'

echo "🏁 All 16 IID experts training completed!"
