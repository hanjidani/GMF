#!/usr/bin/env bash

# Mixed-Experts V100 Partition Alpha Ablation Experiment Runner (FULL EVALUATION, NO LR SCHEDULERS)
# Runs all 25 experiments: 5 fusion types × 5 alpha values
# Uses V100 partition with 16 GPUs for parallel execution
# Includes advanced augmentation: CutMix, Label Smoothing, Gradient Clipping
# Calls train_mixed_experts_fusions_alpha_ablation_full_eval.py (no schedulers, with full eval phases)

set -eo pipefail

# Configuration
ALPHAS=(0.5 1.0 5.0 10.0 15.0)
FUSION_TYPES=("multiplicative" "multiplicativeAddition" "TransformerBase" "concatenation" "simpleAddition")
DN_IDX=1
RN_IDX=0
WRN_IDX=0
RX_IDX=0
SLURM_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="$SLURM_SCRIPTS_DIR/logs_mixed_alpha_ablation_full_eval"
RESULTS_DIR="$SLURM_SCRIPTS_DIR/mixed_v100_partition_alpha_ablation_full_eval_results"
EXPERIMENT_LOG="$RESULTS_DIR/experiment_master_log.txt"
FAILED_JOBS_LOG="$RESULTS_DIR/failed_jobs_log.txt"
RESUBMISSION_LOG="$RESULTS_DIR/resubmission_log.txt"

# Create directories
mkdir -p "$LOGS_DIR"
mkdir -p "$RESULTS_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
ORANGE='\033[0;33m'
NC='\033[0m' # No Color

# Helper to print colored status and append to log
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}[$(date '+%Y-%m-%d %H:%M:%S')] ${message}${NC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${message}" >> "$EXPERIMENT_LOG"
}

log_failed_job() {
    local job_id=$1
    local exp_info=$2
    local error_type=$3
    local exit_code=$4
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: $exp_info (Job ID: $job_id, Error: $error_type, Exit: $exit_code)" >> "$FAILED_JOBS_LOG"
}

log_resubmission() {
    local original_job_id=$1
    local new_job_id=$2
    local exp_info=$3
    local attempt=$4
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] RESUBMITTED: $exp_info (Original: $original_job_id, New: $new_job_id, Attempt: $attempt)" >> "$RESUBMISSION_LOG"
}

# Create experiment combinations
create_experiment_list() {
    local experiments=()
    local exp_id=1
    for fusion_type in "${FUSION_TYPES[@]}"; do
        for alpha in "${ALPHAS[@]}"; do
            experiments+=("$exp_id|$fusion_type|$alpha|$DN_IDX|$RN_IDX|$WRN_IDX|$RX_IDX")
            ((exp_id++))
        done
    done
    echo "${experiments[@]}"
}

# Create Slurm script for a specific experiment
create_slurm_script() {
    local fusion_type=$1
    local alpha=$2
    local exp_id=$3
    local dn_idx=$4
    local rn_idx=$5
    local wrn_idx=$6
    local rx_idx=$7

    local script_name="temp_mixed_alpha_full_eval_${fusion_type}_alpha${alpha}_exp${exp_id}.sh"
    local script_path="$SLURM_SCRIPTS_DIR/$script_name"

    cat > "$script_path" << EOF
#!/usr/bin/env bash

# Temporary Slurm script for MIXED-EXPERTS ALPHA ABLATION (FULL EVAL) experiment $exp_id
# Fusion: $fusion_type, Alpha: $alpha, Idxs: DN=$dn_idx RN=$rn_idx WRN=$wrn_idx RX=$rx_idx

#SBATCH -J MIXED${fusion_type^^}${alpha}ABLFE
#SBATCH -p volta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o ${LOGS_DIR}/mixed_${fusion_type}_alpha${alpha}_${dn_idx}_${rn_idx}_${wrn_idx}_${rx_idx}_exp${exp_id}_%j.log
#SBATCH -e ${LOGS_DIR}/mixed_${fusion_type}_alpha${alpha}_${dn_idx}_${rn_idx}_${wrn_idx}_${rx_idx}_exp${exp_id}_%j.log

set -eo pipefail
exec 2>&1

# Conda env activation
source /home/ali.rasekh/miniconda3/envs/newenv/bin/activate /home/ali.rasekh/miniconda3/envs/newenv/envs/orm
conda activate orm

echo "=========================================="
echo "Mixed-Experts Alpha Ablation (Full Eval) Experiment $exp_id: $fusion_type fusion, alpha=$alpha"
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
echo "SLURM_NODELIST = ${SLURM_NODELIST}"
echo "Alpha: $alpha"
echo "Augmentation: CutMix (α=1.0) + Label Smoothing (0.1) + Grad Clip (1.0)"
echo "No LR schedulers (fixed LR); Full evaluation enabled"
echo "=========================================="

# Change to the correct directory
cd /home/ali.rasekh/orm/hos/geom/Fianl_MCN/fusion_training/scripts

# Run fusion training (alpha ablation full-eval, no schedulers)
python3 -u train_mixed_experts_fusions_alpha_ablation_full_eval.py \
    --fusion_type "$fusion_type" \
    --alpha $alpha \
    --checkpoint_dir ../../expert_training/scripts/checkpoints_expert_iid \
    --output_dir ../fusion_checkpoints_mixed_alpha_ablation_full_eval \
    --data_dir ../data \
    --epochs 40 \
    --batch_size 128 \
    --seed 42 \
    --augmentation_mode cutmix \
    --mixup_alpha 0.2 \
    --cutmix_alpha 1.0 \
    --label_smoothing 0.1 \
    --gradient_clip_norm 1.0 \
    --base_lr 1e-4 \
    --head_lr 1e-3 \
    --experts_lr_scale 0.1 \
    --densenet_idx $dn_idx \
    --resnet_idx $rn_idx \
    --wideresnet_idx $wrn_idx \
    --resnext_idx $rx_idx

echo "✅ Experiment $exp_id completed: MIXED $fusion_type fusion, alpha=$alpha (Alpha Ablation Full Eval)"
EOF

    chmod +x "$script_path"
    echo "$script_path"
}

is_job_running() {
    local job_id=$1
    squeue -j "$job_id" 2>/dev/null | grep -q "$job_id"
}

is_cuda_ecc_error() {
    local job_id=$1
    local log_file=$(find "$LOGS_DIR" -name "*_${job_id}.log" -type f | head -1)
    if [ -z "$log_file" ]; then
        log_file=$(grep -l "SLURM_JOB_ID = ${job_id}" "$LOGS_DIR"/*.log 2>/dev/null | head -1)
    fi
    if [ -n "$log_file" ] && [ -f "$log_file" ]; then
        print_status "$YELLOW" "🔍 Checking log file for CUDA ECC error: $log_file"
        if grep -q "uncorrectable ECC error" "$log_file" || \
           grep -q "CUDA error: uncorrectable ECC error" "$log_file"; then
            print_status "$ORANGE" "🔄 CUDA ECC error detected in log file"
            return 0
        fi
    else
        print_status "$YELLOW" "⚠️  Could not find log file for job ID: $job_id"
    fi
    return 1
}

wait_for_job() {
    local job_id=$1
    local exp_info=$2
    print_status "$BLUE" "Waiting for experiment: $exp_info (Job ID: $job_id)"
    while is_job_running "$job_id"; do
        echo -n "."
        sleep 30
    done
    echo ""
    local exit_code=$(sacct -j "$job_id" --format=ExitCode -n | tail -1 | cut -d: -f1 | tr -d '[:space:]')
    if [ "$exit_code" = "0" ]; then
        print_status "$GREEN" "✅ SUCCESS: $exp_info (Job ID: $job_id)"
        return 0
    else
        print_status "$RED" "❌ FAILED: $exp_info (Job ID: $job_id, Exit: $exit_code)"
        print_status "$YELLOW" "🔍 Checking for CUDA ECC error in job $job_id..."
        if is_cuda_ecc_error "$job_id"; then
            print_status "$ORANGE" "🔄 CUDA ECC error detected - will resubmit automatically"
            log_failed_job "$job_id" "$exp_info" "CUDA_ECC_ERROR" "$exit_code"
            return 2
        else
            print_status "$RED" "❌ Non-CUDA error - will not resubmit"
            log_failed_job "$job_id" "$exp_info" "OTHER_ERROR" "$exit_code"
            return 1
        fi
    fi
}

resubmit_failed_job() {
    local original_job_id=$1
    local fusion_type=$2
    local alpha=$3
    local exp_id=$4
    local expert_index=$5
    local attempt=$6
    local exp_info="Exp $exp_id: MIXED $fusion_type fusion, alpha=$alpha (Resubmission $attempt)"
    print_status "$ORANGE" "🔄 Resubmitting failed job: $exp_info"
    local script_path=$(create_slurm_script "$fusion_type" "$alpha" "$exp_id" "$expert_index")
    local job_output=$(sbatch "$script_path" 2>&1)
    local new_job_id=$(echo "$job_output" | grep -o '[0-9]\+')
    if [ -z "$new_job_id" ]; then
        print_status "$RED" "Failed to resubmit $exp_info: $job_output"
        rm -f "$script_path"
        return 1
    fi
    print_status "$GREEN" "🔄 Resubmitted $exp_info (New Job ID: $new_job_id)"
    log_resubmission "$original_job_id" "$new_job_id" "$exp_info" "$attempt"
    rm -f "$script_path"
    return 0
}

run_experiment_batch() {
    local experiments=("$@")
    local batch_size=16
    local total_experiments=${#experiments[@]}
    local current_batch=0
    local max_resubmission_attempts=3

    print_status "$CYAN" "Starting batch processing of $total_experiments experiments"

    while [ $current_batch -lt $total_experiments ]; do
        local batch_start=$current_batch
        local batch_end=$((current_batch + batch_size - 1))
        if [ $batch_end -ge $total_experiments ]; then
            batch_end=$(($total_experiments - 1))
        fi

        local batch_count=$((batch_end - batch_start + 1))
        print_status "$YELLOW" "Processing experiments $((current_batch + 1))-$((batch_end + 1)) of $total_experiments (Batch $((current_batch / batch_size + 1)))"

        local job_ids=()
        local exp_infos=()
        local exp_data_list=()

        for ((i=batch_start; i<=batch_end; i++)); do
            local exp_data="${experiments[$i]}"
            local exp_id=$(echo "$exp_data" | cut -d'|' -f1)
            local fusion_type=$(echo "$exp_data" | cut -d'|' -f2)
            local alpha=$(echo "$exp_data" | cut -d'|' -f3)
            local dn_idx=$(echo "$exp_data" | cut -d'|' -f4)
            local rn_idx=$(echo "$exp_data" | cut -d'|' -f5)
            local wrn_idx=$(echo "$exp_data" | cut -d'|' -f6)
            local rx_idx=$(echo "$exp_data" | cut -d'|' -f7)
            local exp_info="Exp $exp_id: MIXED $fusion_type fusion, alpha=$alpha"
            exp_infos+=("$exp_info")
            exp_data_list+=("$exp_data")
            local script_path=$(create_slurm_script "$fusion_type" "$alpha" "$exp_id" "$dn_idx" "$rn_idx" "$wrn_idx" "$rx_idx")
            print_status "$BLUE" "Submitting $exp_info"
            local job_output=$(sbatch "$script_path" 2>&1)
            local job_id=$(echo "$job_output" | grep -o '[0-9]\+')
            if [ -z "$job_id" ]; then
                print_status "$RED" "Failed to submit $exp_info: $job_output"
                rm -f "$script_path"
                continue
            fi
            job_ids+=("$job_id")
            print_status "$GREEN" "Submitted $exp_info (Job ID: $job_id)"
            rm -f "$script_path"
        done

        print_status "$PURPLE" "Waiting for experiment to complete..."

        local batch_success=0
        local batch_failures=()
        local batch_cuda_ecc_failures=()

        for ((i=0; i<${#job_ids[@]}; i++)); do
            local result
            if wait_for_job "${job_ids[$i]}" "${exp_infos[$i]}"; then
                result=0
            else
                result=$?
            fi
            if [ "$result" -eq 0 ]; then
                ((batch_success++))
            elif [ "$result" -eq 2 ]; then
                batch_cuda_ecc_failures+=("${exp_data_list[$i]}|${job_ids[$i]}")
            else
                batch_failures+=("${exp_data_list[$i]}|${job_ids[$i]}")
            fi
        done

        print_status "$CYAN" "Experiment completed: $batch_success/${#job_ids[@]} succeeded"

        if [ ${#batch_cuda_ecc_failures[@]} -gt 0 ]; then
            print_status "$ORANGE" "🔄 Processing ${#batch_cuda_ecc_failures[@]} CUDA ECC failures for resubmission..."
            for failure in "${batch_cuda_ecc_failures[@]}"; do
                local exp_data=$(echo "$failure" | cut -d'|' -f1-7)
                local failed_job_id=$(echo "$failure" | cut -d'|' -f8)
                local exp_id=$(echo "$exp_data" | cut -d'|' -f1)
                local fusion_type=$(echo "$exp_data" | cut -d'|' -f2)
                local alpha=$(echo "$exp_data" | cut -d'|' -f3)
                local dn_idx=$(echo "$exp_data" | cut -d'|' -f4)
                local rn_idx=$(echo "$exp_data" | cut -d'|' -f5)
                local wrn_idx=$(echo "$exp_data" | cut -d'|' -f6)
                local rx_idx=$(echo "$exp_data" | cut -d'|' -f7)
                print_status "$ORANGE" "🔄 Resubmitting experiment: $exp_data (Failed Job ID: $failed_job_id)"
                local resubmission_success=false
                for attempt in $(seq 1 $max_resubmission_attempts); do
                    print_status "$YELLOW" "🔄 Resubmission attempt $attempt of $max_resubmission_attempts"
                    if resubmit_failed_job "$failed_job_id" "$fusion_type" "$alpha" "$exp_id" "$dn_idx" "$rn_idx" "$wrn_idx" "$rx_idx" "$attempt"; then
                        resubmission_success=true
                        print_status "$GREEN" "✅ Resubmission attempt $attempt successful"
                        break
                    fi
                    if [ $attempt -lt $max_resubmission_attempts ]; then
                        print_status "$YELLOW" "🔄 Resubmission attempt $attempt failed, trying again in 30 seconds..."
                        sleep 30
                    fi
                done
                if [ "$resubmission_success" = false ]; then
                    print_status "$RED" "❌ Failed to resubmit after $max_resubmission_attempts attempts"
                    batch_failures+=("$exp_data|$failed_job_id")
                fi
            done
        fi

        local total_failures=$((${#batch_failures[@]} + ${#batch_cuda_ecc_failures[@]}))
        print_status "$CYAN" "Experiment final status: $batch_success succeeded, $total_failures failed"
        current_batch=$((batch_end + 1))
        if [ $current_batch -lt $total_experiments ]; then
            print_status "$BLUE" "Waiting 60 seconds before next batch..."
            sleep 60
        fi
    done
}

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -l, --list     List all experiment combinations"
    echo "  -d, --dry-run  Show what would be run without executing"
    echo ""
    echo "Fusion Types: ${FUSION_TYPES[*]}"
    echo "Alpha Values: ${ALPHAS[*]}"
    echo "Total Experiments: $((${#FUSION_TYPES[@]} * ${#ALPHAS[@]}))"
}

list_experiments() {
    local experiments=($(create_experiment_list))
    echo "All Experiment Combinations (MIXED-EXPERTS ALPHA ABLATION FULL EVAL MODE):"
    echo "============================================"
    for exp_data in "${experiments[@]}"; do
        local exp_id=$(echo "$exp_data" | cut -d'|' -f1)
        local fusion_type=$(echo "$exp_data" | cut -d'|' -f2)
        local alpha=$(echo "$exp_data" | cut -d'|' -f3)
        printf "Exp %3d: %-20s alpha=%-5s (Alpha Ablation Full Eval)\n" "$exp_id" "$fusion_type" "$alpha"
    done
    echo "============================================"
    echo "Total: ${#experiments[@]} experiments"
    echo "Batch size: 16 (V100 partition, 16 GPUs)"
}

dry_run() {
    local experiments=($(create_experiment_list))
    echo "DRY RUN - No experiments will be executed"
    echo "=========================================="
    echo "Total experiments: ${#experiments[@]}"
    echo "Batch size: 16"
    echo "Total batches: $(((${#experiments[@]} + 15) / 16))"
    echo ""
    local batch_num=1
    for ((i=0; i<${#experiments[@]}; i+=16)); do
        local batch_end=$((i + 15))
        if [ $batch_end -ge ${#experiments[@]} ]; then
            batch_end=$((${#experiments[@]} - 1))
        fi
        echo "Batch $batch_num: Experiments $((i+1))-$((batch_end+1))"
        for ((j=i; j<=batch_end; j++)); do
            if [ $j -lt ${#experiments[@]} ]; then
                local exp_data="${experiments[$j]}"
                local exp_id=$(echo "$exp_data" | cut -d'|' -f1)
                local fusion_type=$(echo "$exp_data" | cut -d'|' -f2)
                local alpha=$(echo "$exp_data" | cut -d'|' -f3)
                echo "  Exp $exp_id: MIXED $fusion_type fusion, alpha=$alpha (Alpha Ablation Full Eval)"
            fi
        done
        echo ""
        ((batch_num++))
    done
}

DRY_RUN=false
LIST_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -l|--list)
            LIST_ONLY=true
            shift
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -*)
            print_status "$RED" "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            print_status "$RED" "Unexpected argument: $1"
            show_usage
            exit 1
            ;;
    esac
done

if ! command -v sbatch &> /dev/null; then
    print_status "$RED" "Error: sbatch command not found. Are you on a Slurm cluster?"
    exit 1
fi

# Check required Python script exists
SCRIPT="$SLURM_SCRIPTS_DIR/train_mixed_experts_fusions_alpha_ablation_full_eval.py"
if [ ! -f "$SCRIPT" ]; then
    print_status "$RED" "Error: Required Python script not found: $SCRIPT"
    exit 1
fi

echo "Mixed-Experts V100 Partition Experiment Log (Alpha Ablation Full Eval)" > "$EXPERIMENT_LOG"
echo "Started: $(date)" >> "$EXPERIMENT_LOG"
echo "==========================================" >> "$EXPERIMENT_LOG"
echo "Fusion Types: ${FUSION_TYPES[*]}" >> "$EXPERIMENT_LOG"
echo "Alpha Values: ${ALPHAS[*]}" >> "$EXPERIMENT_LOG"
echo "Total Experiments: $((${#FUSION_TYPES[@]} * ${#ALPHAS[@]}))" >> "$EXPERIMENT_LOG"
echo "Partition: volta (V100)" >> "$EXPERIMENT_LOG"
echo "Batch Size: 16 experiments (16 GPUs for parallel execution)" >> "$EXPERIMENT_LOG"
echo "Mode: Alpha Ablation Full Eval (No LR schedulers)" >> "$EXPERIMENT_LOG"
echo "Augmentation: CutMix (α=1.0) + Label Smoothing (0.1) + Gradient Clipping (1.0)" >> "$EXPERIMENT_LOG"
echo "Auto-resubmission: CUDA ECC errors" >> "$EXPERIMENT_LOG"
echo "==========================================" >> "$EXPERIMENT_LOG"

echo "Failed Jobs Log - Mixed-Experts Experiments (Alpha Ablation Full Eval)" > "$FAILED_JOBS_LOG"
echo "Started: $(date)" >> "$FAILED_JOBS_LOG"
echo "==========================================" >> "$FAILED_JOBS_LOG"

echo "Resubmission Log - Mixed-Experts Experiments (Alpha Ablation Full Eval)" > "$RESUBMISSION_LOG"
echo "Started: $(date)" >> "$RESUBMISSION_LOG"
echo "==========================================" >> "$RESUBMISSION_LOG"

print_status "$CYAN" "=========================================="
print_status "$CYAN" "MIXED-EXPERTS V100 PARTITION EXPERIMENT RUNNER (ALPHA ABLATION FULL EVAL)"
print_status "$CYAN" "=========================================="
print_status "$CYAN" "Fusion Types: ${FUSION_TYPES[*]}"
print_status "$CYAN" "Alpha Values: ${ALPHAS[*]}"
print_status "$CYAN" "Total Experiments: $((${#FUSION_TYPES[@]} * ${#ALPHAS[@]}))"
print_status "$CYAN" "Partition: volta (V100)"
print_status "$CYAN" "Batch Size: 16 experiments (16 GPUs for parallel execution)"
print_status "$CYAN" "Mode: Alpha Ablation Full Eval (No LR schedulers)"
print_status "$CYAN" "Augmentation: CutMix + Label Smoothing + Gradient Clipping"
print_status "$CYAN" "Auto-resubmission: CUDA ECC errors"
print_status "$CYAN" "=========================================="

if [ "$LIST_ONLY" = true ]; then
    list_experiments
    exit 0
fi

if [ "$DRY_RUN" = true ]; then
    dry_run
    exit 0
fi

print_status "$BLUE" "Creating experiment combinations..."
experiments=($(create_experiment_list))
print_status "$GREEN" "Created ${#experiments[@]} experiment combinations"

print_status "$GREEN" "Starting comprehensive experiment execution on V100 partition (Mixed-Experts Alpha Ablation Full Eval)..."
run_experiment_batch "${experiments[@]}"

print_status "$GREEN" "=========================================="
print_status "$GREEN" "ALL EXPERIMENTS COMPLETED!"
print_status "$GREEN" "=========================================="
print_status "$GREEN" "Total experiments: ${#experiments[@]}"
print_status "$GREEN" "Results saved in: $RESULTS_DIR"
print_status "$GREEN" "Master log: $EXPERIMENT_LOG"
print_status "$GREEN" "Failed jobs log: $FAILED_JOBS_LOG"
print_status "$GREEN" "Resubmission log: $RESUBMISSION_LOG"
print_status "$GREEN" "=========================================="

echo "==========================================" >> "$EXPERIMENT_LOG"
echo "ALL EXPERIMENTS COMPLETED!" >> "$EXPERIMENT_LOG"
echo "Completed: $(date)" >> "$EXPERIMENT_LOG"
echo "Total experiments: ${#experiments[@]}" >> "$EXPERIMENT_LOG"
echo "==========================================" >> "$EXPERIMENT_LOG"


