#!/usr/bin/env bash

# Non-IID Fusion Alpha Ablation Runner (Full Evaluation)
# Runs all combinations of fusion types × alpha values on V100 partition

set -eo pipefail

ALPHAS=(0.5 1.0 5.0 10.0 15.0)
FUSION_TYPES=("multiplicative" "multiplicativeAddition" "TransformerBase" "concatenation" "simpleAddition")

SLURM_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="$SLURM_SCRIPTS_DIR/logs_alpha_ablation_non_iid"
mkdir -p "$LOGS_DIR"

print_status() {
    local color=$1
    local message=$2
    echo -e "${color}[$(date '+%Y-%m-%d %H:%M:%S')] ${message}\033[0m"
}

create_slurm_script() {
    local fusion_type=$1
    local alpha=$2
    local exp_id=$3
    local script_name="temp_non_iid_${fusion_type}_alpha${alpha}_exp${exp_id}.sh"
    local script_path="$SLURM_SCRIPTS_DIR/$script_name"

    cat > "$script_path" << EOF
#!/usr/bin/env bash
#SBATCH -J NONIID_${fusion_type^^}_A${alpha}
#SBATCH -p volta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o ${LOGS_DIR}/${fusion_type}_alpha${alpha}_exp${exp_id}_%j.log
#SBATCH -e ${LOGS_DIR}/${fusion_type}_alpha${alpha}_exp${exp_id}_%j.log

set -eo pipefail
exec 2>&1

source /home/ali.rasekh/miniconda3/envs/newenv/bin/activate /home/ali.rasekh/miniconda3/envs/newenv/envs/orm
conda activate orm

cd /home/ali.rasekh/orm/hos/geom/Fianl_MCN/fusion_non_iid/scripts

python3 -u train_fusion_full_eval.py \
  --fusion_type "$fusion_type" \
  --alpha $alpha \
  --checkpoint_dir ../../expert_training/scripts/checkpoints_expert_noniid \
  --output_dir ./non_iid_full_eval_checkpoints \
  --data_dir ../data \
  --tinyimagenet_dir ./tiny-imagenet-200 \
  --epochs 100 \
  --batch_size 128 \
  --seed 42 \
  --lr_backbone 1e-5 \
  --lr_heads 1e-4
EOF

    chmod +x "$script_path"
    echo "$script_path"
}

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'

exp_id=1
BATCH_SIZE=15

# Create array of all experiment combinations
experiments=()
for fusion_type in "${FUSION_TYPES[@]}"; do
  for alpha in "${ALPHAS[@]}"; do
    experiments+=("$fusion_type:$alpha")
  done
done

total_experiments=${#experiments[@]}
print_status "$BLUE" "Total experiments: $total_experiments (${#FUSION_TYPES[@]} fusion types × ${#ALPHAS[@]} alpha values)"
print_status "$BLUE" "Submitting experiments in batches of $BATCH_SIZE..."

# Process experiments in batches
for ((i=0; i<total_experiments; i+=BATCH_SIZE)); do
  batch_num=$((i/BATCH_SIZE + 1))
  batch_end=$((i + BATCH_SIZE))
  if [ $batch_end -gt $total_experiments ]; then
    batch_end=$total_experiments
  fi
  
  print_status "$YELLOW" "=== BATCH $batch_num: Experiments $((i+1))-$batch_end ==="
  
  # Submit all experiments in current batch
  batch_jobs=()
  for ((j=i; j<batch_end; j++)); do
    IFS=':' read -r fusion_type alpha <<< "${experiments[j]}"
    print_status "$YELLOW" "Submitting ${fusion_type} alpha=${alpha} (exp $((j+1)))..."
    
    script_path=$(create_slurm_script "$fusion_type" "$alpha" "$((j+1))")
    out=$(sbatch "$script_path")
    jid=$(echo "$out" | grep -o '[0-9]\+')
    batch_jobs+=("$jid")
    print_status "$GREEN" "Submitted ${fusion_type} alpha=${alpha} (exp $((j+1))) as Job $jid"
    rm -f "$script_path"
  done
  
  print_status "$PURPLE" "Batch $batch_num submitted: ${#batch_jobs[@]} jobs (${batch_jobs[*]})"
  print_status "$PURPLE" "Waiting for all jobs in batch $batch_num to complete..."
  
  # Wait for all jobs in current batch to complete
  for jid in "${batch_jobs[@]}"; do
    while squeue -j "$jid" 2>/dev/null | grep -q "$jid"; do
      echo -n "."; sleep 30
    done
  done
  echo ""
  print_status "$YELLOW" "Batch $batch_num completed! All jobs finished. Check logs under $LOGS_DIR."
  
  # Small delay between batches
  if [ $batch_end -lt $total_experiments ]; then
    print_status "$BLUE" "Starting next batch in 10 seconds..."
    sleep 10
  fi
done

print_status "$GREEN" "All Non-IID fusion alpha ablation jobs have completed!"
print_status "$GREEN" "Total experiments completed: $total_experiments"


