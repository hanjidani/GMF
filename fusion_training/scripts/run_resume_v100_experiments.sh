#!/usr/bin/env bash

# Resume remaining experiments on V100 (continues after failures or partial runs)
# Submits only experiments that are not marked SUCCESS in the master log.
# Usage:
#   bash run_resume_v100_experiments.sh           # uses master log SUCCESS detection
#   bash run_resume_v100_experiments.sh 17        # submits experiments with exp_id >= 17

set -o pipefail

# Optional start index (exp_id)
START_ID="$1"

# Config (align with run_comprehensive_v100_partition.sh)
ALPHAS=(0.5 1.0 5.0 10.0 15.0)
MODELS=("densenet")
FUSION_TYPES=("multiplicative" "multiplicativeAddition" "TransformerBase" "concatenation" "simpleAddition")
SLURM_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="$SLURM_SCRIPTS_DIR/logs"
RESULTS_DIR="$SLURM_SCRIPTS_DIR/densenet_v100_partition_results"
EXPERIMENT_LOG="$RESULTS_DIR/experiment_master_log.txt"

# Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
  local color=$1; shift
  local message=$*
  echo -e "${color}[$(date '+%Y-%m-%d %H:%M:%S')] ${message}${NC}"
}

create_experiment_list() {
  local experiments=()
  local exp_id=1
  for model in "${MODELS[@]}"; do
    for fusion_type in "${FUSION_TYPES[@]}"; do
      for alpha in "${ALPHAS[@]}"; do
        experiments+=("$exp_id|$model|$fusion_type|$alpha")
        ((exp_id++))
      done
    done
  done
  echo "${experiments[@]}"
}

create_slurm_script() {
  local model=$1
  local fusion_type=$2
  local alpha=$3
  local exp_id=$4

  local script_name="temp_resume_${model}_${fusion_type}_alpha${alpha}_exp${exp_id}.sh"
  local script_path="$SLURM_SCRIPTS_DIR/$script_name"

  cat > "$script_path" << EOF
#!/usr/bin/env bash
#SBATCH -J RESUME${model^^}${fusion_type^^}${alpha}
#SBATCH -p volta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o logs/${model}_${fusion_type}_alpha${alpha}_exp${exp_id}_%j.log
#SBATCH -e logs/${model}_${fusion_type}_alpha${alpha}_exp${exp_id}_%j.log

set -eo pipefail
exec 2>&1

# Conda env
source /home/ali.rasekh/miniconda3/envs/newenv/bin/activate /home/ali.rasekh/miniconda3/envs/newenv/envs/orm
conda activate orm

cd $SLURM_SCRIPTS_DIR

python3 -u train_${model}_fusions.py \
  --fusion_type "$fusion_type" \
  --alpha $alpha \
  --checkpoint_dir ../../expert_training/scripts/checkpoints_expert_iid \
  --output_dir ../fusion_checkpoints \
  --data_dir ../data \
  --epochs 100 \
  --batch_size 128 \
  --seed 42 \
  --augmentation_mode cutmix \
  --mixup_alpha 0.2 \
  --cutmix_alpha 1.0 \
  --label_smoothing 0.1 \
  --gradient_clip_norm 1.0 \
  --skip_pre_eval 
EOF

  chmod +x "$script_path"
  echo "$script_path"
}

# Build success set from master log (if START_ID not provided)
declare -A success_map
if [ -z "$START_ID" ]; then
  if [ -f "$EXPERIMENT_LOG" ]; then
    while IFS= read -r line; do
      if [[ "$line" == *"SUCCESS: Exp"* ]]; then
        exp_id=$(echo "$line" | sed -n 's/.*SUCCESS: Exp \([0-9]\+\).*/\1/p')
        if [ -n "$exp_id" ]; then success_map[$exp_id]=1; fi
      fi
    done < "$EXPERIMENT_LOG"
  else
    print_status "$YELLOW" "Master log not found at $EXPERIMENT_LOG, proceeding to submit all experiments."
  fi
fi

# Create remaining list
experiments=($(create_experiment_list))
remaining=()
for exp_data in "${experiments[@]}"; do
  exp_id=$(echo "$exp_data" | cut -d'|' -f1)
  if [ -n "$START_ID" ]; then
    # Use index-based resume
    if [ "$exp_id" -ge "$START_ID" ]; then
      remaining+=("$exp_data")
    fi
  else
    # Use SUCCESS-based resume
    if [ -z "${success_map[$exp_id]}" ]; then
      remaining+=("$exp_data")
    fi
  fi
done

print_status "$CYAN" "Total experiments: ${#experiments[@]}"
if [ -n "$START_ID" ]; then
  print_status "$CYAN" "Resuming from Exp ID >= $START_ID"
else
  print_status "$CYAN" "Already successful: ${#success_map[@]} (from master log)"
fi
print_status "$CYAN" "Remaining to submit: ${#remaining[@]}"

mkdir -p "$LOGS_DIR" "$RESULTS_DIR"

# Submit remaining experiments
submitted=0
for exp_data in "${remaining[@]}"; do
  exp_id=$(echo "$exp_data" | cut -d'|' -f1)
  model=$(echo "$exp_data" | cut -d'|' -f2)
  fusion_type=$(echo "$exp_data" | cut -d'|' -f3)
  alpha=$(echo "$exp_data" | cut -d'|' -f4)

  script_path=$(create_slurm_script "$model" "$fusion_type" "$alpha" "$exp_id")
  print_status "$CYAN" "Submitting Exp $exp_id: $model $fusion_type fusion, alpha=$alpha (resume)"
  job_output=$(sbatch "$script_path" 2>&1 || true)
  job_id=$(echo "$job_output" | grep -o '[0-9]\+' | tail -1)
  if [ -n "$job_id" ]; then
    print_status "$GREEN" "Submitted Exp $exp_id (Job ID: $job_id)"
    ((submitted++))
  else
    print_status "$RED" "Failed to submit Exp $exp_id: $job_output"
  fi
  rm -f "$script_path"
  sleep 0.2
done

print_status "$GREEN" "Resume submission complete. Submitted: $submitted of ${#remaining[@]} remaining."
