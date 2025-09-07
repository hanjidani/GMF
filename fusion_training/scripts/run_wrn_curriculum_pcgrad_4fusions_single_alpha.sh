#!/usr/bin/env bash

# WRN Curriculum+PCGrad Runner: run 4 fusion types for a single alpha
# Loads cluster/partition/env style from existing comprehensive runner

set -eo pipefail

# Configuration (load style from the DenseNet comprehensive runner)
FUSION_TYPES=("multiplicative" "multiplicativeAddition" "TransformerBase" "concatenation")
ALPHA=${1:-1.0}
MODEL="improved_wide_resnet"
SLURM_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="$SLURM_SCRIPTS_DIR/logs_wrn_curriculum_pcgrad_alpha_${ALPHA}"
mkdir -p "$LOGS_DIR"

print_status() {
    local color='\033[0;36m'
    local nc='\033[0m'
    echo -e "${color}[$(date '+%Y-%m-%d %H:%M:%S')] $1${nc}"
}

create_slurm_script() {
    local fusion_type=$1
    local alpha=$2
    local script_path="$SLURM_SCRIPTS_DIR/temp_wrn_curriculum_pcgrad_${fusion_type}_alpha${alpha}.sh"

    cat > "$script_path" << EOF
#!/usr/bin/env bash
#SBATCH -J WRN${fusion_type^^}${alpha}CPC
#SBATCH -p volta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o ${LOGS_DIR}/${MODEL}_${fusion_type}_alpha${alpha}_%j.log
#SBATCH -e ${LOGS_DIR}/${MODEL}_${fusion_type}_alpha${alpha}_%j.log

set -eo pipefail
exec 2>&1

source /home/ali.rasekh/miniconda3/envs/newenv/bin/activate /home/ali.rasekh/miniconda3/envs/newenv/envs/orm
conda activate orm

echo "=========================================="
echo "WRN Curriculum+PCGrad: ${MODEL} ${fusion_type}, alpha=${alpha}"
echo "SLURM_JOB_ID = \\${SLURM_JOB_ID}"
echo "SLURM_NODELIST = \\${SLURM_NODELIST}"
echo "=========================================="

cd /home/ali.rasekh/orm/hos/geom/Fianl_MCN/fusion_training/scripts

/home/ali.rasekh/miniconda3/envs/newenv/envs/orm/bin/python3 -u train_improved_wide_resnet_fusions_curriculum_pcgrad.py \
  --fusion_type ${fusion_type} \
  --alpha ${alpha} \
  --checkpoint_dir ../../expert_training/scripts/checkpoints_expert_iid \
  --output_dir ../fusion_checkpoints_curriculum_pcgrad \
  --data_dir ../data \
  --epochs 40 \
  --warmup_epochs 5 \
  --batch_size 128 \
  --seed 42 \
  --base_lr 1e-4 \
  --head_lr 1e-3 \
  --experts_lr_scale 0.05 \
  --gradient_clip_norm 1.0

echo "✅ Completed: ${MODEL} ${fusion_type}, alpha=${alpha} (Curriculum+PCGrad)"
EOF

    chmod +x "$script_path"
    echo "$script_path"
}

print_status "Submitting 4 fusion types for alpha=${ALPHA} (WRN Curriculum+PCGrad)"

JOB_IDS=()
for fusion in "${FUSION_TYPES[@]}"; do
    script_path=$(create_slurm_script "$fusion" "$ALPHA")
    print_status "Submitting ${MODEL} ${fusion}, alpha=${ALPHA}"
    out=$(sbatch "$script_path" 2>&1 || true)
    job_id=$(echo "$out" | grep -o '[0-9]\+' | tail -1)
    if [ -n "$job_id" ]; then
        JOB_IDS+=("$job_id")
        print_status "Submitted (Job ID: $job_id)"
    else
        print_status "Submission failed: $out"
    fi
    rm -f "$script_path"
done

print_status "Launched Jobs: ${JOB_IDS[*]}"


