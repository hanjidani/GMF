#!/usr/bin/env bash

# DenseNet V100 Partition Alpha Ablation Experiment Runner (FULL EVALUATION, NO LR SCHEDULERS)
# Runs all 25 experiments: 5 fusion types × 5 alpha values
# Uses V100 partition with 16 GPUs for parallel execution
# Includes advanced augmentation: CutMix, Label Smoothing, Gradient Clipping
# Calls train_densenet_fusions_alpha_ablation_full_eval.py (no schedulers, with full eval phases)

set -eo pipefail

# Configuration
ALPHAS=(0.5 1.0 5.0 10.0 15.0)
MODELS=("densenet")
FUSION_TYPES=("multiplicative" "multiplicativeAddition" "TransformerBase" "concatenation" "simpleAddition")
SLURM_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="$SLURM_SCRIPTS_DIR/logs_alpha_ablation_full_eval"
RESULTS_DIR="$SLURM_SCRIPTS_DIR/densenet_v100_partition_alpha_ablation_full_eval_results"
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

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}[$(date '+%Y-%m-%d %H:%M:%S')] ${message}${NC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${message}" >> "$EXPERIMENT_LOG"
}

# Function to log failed jobs
log_failed_job() {
    local job_id=$1
    local exp_info=$2
    local error_type=$3
    local exit_code=$4
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: $exp_info (Job ID: $job_id, Error: $error_type, Exit: $exit_code)" >> "$FAILED_JOBS_LOG"
}

# Function to log resubmissions
log_resubmission() {
    local original_job_id=$1
    local new_job_id=$2
    local exp_info=$3
    local attempt=$4
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] RESUBMITTED: $exp_info (Original: $original_job_id, New: $new_job_id, Attempt: $attempt)" >> "$RESUBMISSION_LOG"
}

# Function to create experiment combinations
create_experiment_list() {
    local experiments=()
    local exp_id=1
    
    # FULL TRAINING: Run all experiments (5 fusion types × 5 alpha values = 25 experiments)
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

# Function to create Slurm script for a specific experiment
create_slurm_script() {
    local model=$1
    local fusion_type=$2
    local alpha=$3
    local exp_id=$4
    
    local script_name="temp_alpha_full_eval_${model}_${fusion_type}_alpha${alpha}_exp${exp_id}.sh"
    local script_path="$SLURM_SCRIPTS_DIR/$script_name"
    
    cat > "$script_path" << EOF
#!/usr/bin/env bash

# Temporary Slurm script for ALPHA ABLATION (FULL EVAL) experiment $exp_id
# Model: $model, Fusion: $fusion_type, Alpha: $alpha

#SBATCH -J ${model^^}${fusion_type^^}${alpha}ABLFE
#SBATCH -p volta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o ${LOGS_DIR}/${model}_${fusion_type}_alpha${alpha}_exp${exp_id}_%j.log
#SBATCH -e ${LOGS_DIR}/${model}_${fusion_type}_alpha${alpha}_exp${exp_id}_%j.log

set -eo pipefail
exec 2>&1

# Conda env activation
source /home/ali.rasekh/miniconda3/envs/newenv/bin/activate /home/ali.rasekh/miniconda3/envs/newenv/envs/orm
conda activate orm

echo "=========================================="
echo "Alpha Ablation (Full Eval) Experiment $exp_id: $model $fusion_type fusion, alpha=$alpha"
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
echo "SLURM_NODELIST = ${SLURM_NODELIST}"
echo "Alpha: $alpha"
echo "Augmentation: CutMix (α=1.0) + Label Smoothing (0.1) + Grad Clip (1.0)"
echo "No LR schedulers (fixed LR); Full evaluation enabled"
echo "=========================================="

# Change to the correct directory
cd /home/ali.rasekh/orm/hos/geom/Fianl_MCN/fusion_training/scripts

# Run fusion training (alpha ablation full-eval, no schedulers)
python3 -u train_${model}_fusions_alpha_ablation_full_eval.py \
    --fusion_type "$fusion_type" \
    --alpha $alpha \
    --checkpoint_dir ../../expert_training/scripts/checkpoints_expert_iid \
    --output_dir ../fusion_checkpoints_alpha_ablation_full_eval \
    --data_dir ../data \
    --epochs 100 \
    --batch_size 128 \
    --seed 42 \
    --augmentation_mode cutmix \
    --mixup_alpha 0.2 \
    --cutmix_alpha 1.0 \
    --label_smoothing 0.1 \
    --gradient_clip_norm 1.0 \
    --base_lr 1e-4 \
    --head_lr 1e-3 \
    --experts_lr_scale 0.1

echo "✅ Experiment $exp_id completed: $model $fusion_type fusion, alpha=$alpha (Alpha Ablation Full Eval)"
EOF

    chmod +x "$script_path"
    echo "$script_path"
}

# Function to check if a job is still running
is_job_running() {
    local job_id=$1
    squeue -j "$job_id" 2>/dev/null | grep -q "$job_id"
}

# Function to check if job failed due to CUDA ECC error
is_cuda_ecc_error() {
    local job_id=$1
    
    # Find the log file for this job ID by searching in the logs directory
    local log_file=$(find "$LOGS_DIR" -name "*_${job_id}.log" -type f | head -1)
    
    # If no log file found with the job ID pattern, try alternative patterns
    if [ -z "$log_file" ]; then
        log_file=$(grep -l "SLURM_JOB_ID = ${job_id}" "$LOGS_DIR"/*.log 2>/dev/null | head -1)
    fi
    
    # Check if log file exists and contains CUDA ECC error
    if [ -n "$log_file" ] && [ -f "$log_file" ]; then
        print_status "$YELLOW" "🔍 Checking log file for CUDA ECC error: $log_file"
        if grep -q "uncorrectable ECC error" "$log_file" || \
           grep -q "CUDA error: uncorrectable ECC error" "$log_file"; then
            print_status "$ORANGE" "🔄 CUDA ECC error detected in log file"
            return 0  # True - CUDA ECC error detected
        fi
    else
        print_status "$YELLOW" "⚠️  Could not find log file for job ID: $job_id"
    fi
    
    return 1  # False - no CUDA ECC error
}

# Function to wait for a job to complete
wait_for_job() {
    local job_id=$1
    local exp_info=$2
    
    print_status "$BLUE" "Waiting for experiment: $exp_info (Job ID: $job_id)"
    
    while is_job_running "$job_id"; do
        echo -n "."
        sleep 30  # Check every 30 seconds
    done
    
    echo ""
    
    # Get final job status
    local exit_code=$(sacct -j "$job_id" --format=ExitCode -n | tail -1 | cut -d: -f1)
    if [ "$exit_code" = "0" ]; then
        print_status "$GREEN" "✅ SUCCESS: $exp_info (Job ID: $job_id)"
        return 0
    else
        print_status "$RED" "❌ FAILED: $exp_info (Job ID: $job_id, Exit: $exit_code)"
        
        # Check if it's a CUDA ECC error
        print_status "$YELLOW" "🔍 Checking for CUDA ECC error in job $job_id..."
        if is_cuda_ecc_error "$job_id"; then
            print_status "$ORANGE" "🔄 CUDA ECC error detected - will resubmit automatically"
            log_failed_job "$job_id" "$exp_info" "CUDA_ECC_ERROR" "$exit_code"
            return 2  # Special return code for CUDA ECC errors
        else
            print_status "$RED" "❌ Non-CUDA error - will not resubmit"
            log_failed_job "$job_id" "$exp_info" "OTHER_ERROR" "$exit_code"
            return 1
        fi
    fi
}

# Function to resubmit failed job
resubmit_failed_job() {
    local original_job_id=$1
    local exp_data=$2
    local attempt=$3
    
    local exp_id=$(echo "$exp_data" | cut -d'|' -f1)
    local model=$(echo "$exp_data" | cut -d'|' -f2)
    local fusion_type=$(echo "$exp_data" | cut -d'|' -f3)
    local alpha=$(echo "$exp_data" | cut -d'|' -f4)
    
    local exp_info="Exp $exp_id: $model $fusion_type fusion, alpha=$alpha (Resubmission $attempt)"
    
    print_status "$ORANGE" "🔄 Resubmitting failed job: $exp_info"
    
    # Create new Slurm script
    local script_path=$(create_slurm_script "$model" "$fusion_type" "$alpha" "$exp_id")
    
    # Submit new job
    local job_output=$(sbatch "$script_path" 2>&1)
    local new_job_id=$(echo "$job_output" | grep -o '[0-9]\+')
    
    if [ -z "$new_job_id" ]; then
        print_status "$RED" "Failed to resubmit $exp_info: $job_output"
        rm -f "$script_path"
        return 1
    fi
    
    print_status "$GREEN" "🔄 Resubmitted $exp_info (New Job ID: $new_job_id)"
    log_resubmission "$original_job_id" "$new_job_id" "$exp_info" "$attempt"
    
    # Clean up temporary script
    rm -f "$script_path"
    
    return 0
}

# Function to run experiments in batches of 16 (full training mode with 16 GPUs)
run_experiment_batch() {
    local experiments=("$@")
    local batch_size=16  # Use all 16 GPUs for parallel execution
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
        
        # Submit batch of experiments (full training mode with 16 GPUs)
        local job_ids=()
        local exp_infos=()
        local exp_data_list=()
        
        for ((i=batch_start; i<=batch_end; i++)); do
            local exp_data="${experiments[$i]}"
            local exp_id=$(echo "$exp_data" | cut -d'|' -f1)
            local model=$(echo "$exp_data" | cut -d'|' -f2)
            local fusion_type=$(echo "$exp_data" | cut -d'|' -f3)
            local alpha=$(echo "$exp_data" | cut -d'|' -f4)
            
            local exp_info="Exp $exp_id: $model $fusion_type fusion, alpha=$alpha"
            exp_infos+=("$exp_info")
            exp_data_list+=("$exp_data")
            
            # Create temporary Slurm script
            local script_path=$(create_slurm_script "$model" "$fusion_type" "$alpha" "$exp_id")
            
            # Submit job
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
            
            # Clean up temporary script
            rm -f "$script_path"
        done
        
        # Wait for all jobs in this batch to complete
        print_status "$PURPLE" "Waiting for experiment to complete..."
        
        local batch_success=0
        local batch_failures=()
        local batch_cuda_ecc_failures=()
        
        for ((i=0; i<${#job_ids[@]}; i++)); do
            wait_for_job "${job_ids[$i]}" "${exp_infos[$i]}"
            local result=$?
            
            if [ "$result" -eq 0 ]; then
                ((batch_success++))
            elif [ "$result" -eq 2 ]; then
                # CUDA ECC error - add to resubmission list
                batch_cuda_ecc_failures+=("${exp_data_list[$i]}|${job_ids[$i]}")
            else
                # Other error - add to failure list
                batch_failures+=("${exp_data_list[$i]}|${job_ids[$i]}")
            fi
        done
        
        print_status "$CYAN" "Experiment completed: $batch_success/${#job_ids[@]} succeeded"
        
        # Handle CUDA ECC failures with automatic resubmission
        if [ ${#batch_cuda_ecc_failures[@]} -gt 0 ]; then
            print_status "$ORANGE" "🔄 Processing ${#batch_cuda_ecc_failures[@]} CUDA ECC failures for resubmission..."
            
            for failure in "${batch_cuda_ecc_failures[@]}"; do
                local exp_data=$(echo "$failure" | cut -d'|' -f1-4)
                local failed_job_id=$(echo "$failure" | cut -d'|' -f5)
                
                print_status "$ORANGE" "🔄 Resubmitting experiment: $exp_data (Failed Job ID: $failed_job_id)"
                
                # Resubmit with attempt tracking
                local resubmission_success=false
                for attempt in $(seq 1 $max_resubmission_attempts); do
                    print_status "$YELLOW" "🔄 Resubmission attempt $attempt of $max_resubmission_attempts"
                    if resubmit_failed_job "$failed_job_id" "$exp_data" "$attempt"; then
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
        
        # Report final experiment status
        local total_failures=$((${#batch_failures[@]} + ${#batch_cuda_ecc_failures[@]}))
        print_status "$CYAN" "Experiment final status: $batch_success succeeded, $total_failures failed"
        
        current_batch=$((batch_end + 1))
        
        # Add delay between batches to avoid overwhelming the scheduler
        if [ $current_batch -lt $total_experiments ]; then
            print_status "$BLUE" "Waiting 60 seconds before next batch..."
            sleep 60
        fi
    done
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -l, --list     List all experiment combinations"
    echo "  -d, --dry-run  Show what would be run without executing"
    echo ""
    echo "This script runs all possible combinations:"
    echo "  Models: ${MODELS[*]}"
    echo "  Fusion Types: ${FUSION_TYPES[*]}"
    echo "  Alpha Values: ${ALPHAS[*]}"
    echo "  Total Experiments: $((${#MODELS[@]} * ${#FUSION_TYPES[@]} * ${#ALPHAS[@]}))"
    echo ""
    echo "Experiments run on V100 partition with 16 GPUs for parallel execution"
    echo "Alpha Ablation Full Eval: No LR schedulers (fixed LR); CutMix + Label Smoothing + Gradient Clipping"
    echo "Automatic resubmission for CUDA ECC errors"
}

# Function to list all experiments
list_experiments() {
    local experiments=($(create_experiment_list))
    
    echo "All Experiment Combinations (ALPHA ABLATION FULL EVAL MODE):"
    echo "============================================"
    
    # Show all experiments that will run
    for exp_data in "${experiments[@]}"; do
        local exp_id=$(echo "$exp_data" | cut -d'|' -f1)
        local model=$(echo "$exp_data" | cut -d'|' -f2)
        local fusion_type=$(echo "$exp_data" | cut -d'|' -f3)
        local alpha=$(echo "$exp_data" | cut -d'|' -f4)
        printf "Exp %3d: %-20s %-20s alpha=%-5s (Alpha Ablation Full Eval)\n" "$exp_id" "$model" "$fusion_type" "$alpha"
    done
    
    echo "============================================"
    echo "Total: ${#experiments[@]} experiments (alpha ablation full eval, no schedulers)"
    echo "Batch size: 16 (V100 partition, 16 GPUs for parallel execution)"
    echo "Augmentation: CutMix (α=1.0) + Label Smoothing (0.1) + Gradient Clipping (1.0)"
}

# Function to dry run
dry_run() {
    local experiments=($(create_experiment_list))
    
    echo "DRY RUN - No experiments will be executed"
    echo "=========================================="
    echo "Total experiments: ${#experiments[@]}"
    echo "Batch size: 16"
    echo "Total batches: $(((${#experiments[@]} + 15) / 16)) (alpha ablation full eval mode)"
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
                local model=$(echo "$exp_data" | cut -d'|' -f2)
                local fusion_type=$(echo "$exp_data" | cut -d'|' -f3)
                local alpha=$(echo "$exp_data" | cut -d'|' -f4)
                echo "  Exp $exp_id: $model $fusion_type fusion, alpha=$alpha (Alpha Ablation Full Eval)"
            fi
        done
        echo ""
        ((batch_num++))
    done
}

# Parse command line arguments
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

# Check if Slurm is available
if ! command -v sbatch &> /dev/null; then
    print_status "$RED" "Error: sbatch command not found. Are you on a Slurm cluster?"
    exit 1
fi

# Check if required Python scripts exist
for model in "${MODELS[@]}"; do
    script="$SLURM_SCRIPTS_DIR/train_${model}_fusions_alpha_ablation_full_eval.py"
    if [ ! -f "$script" ]; then
        print_status "$RED" "Error: Required Python script not found: $script"
        exit 1
    fi
done

# Initialize logs
echo "Comprehensive V100 Partition Experiment Log (Alpha Ablation Full Eval)" > "$EXPERIMENT_LOG"
echo "Started: $(date)" >> "$EXPERIMENT_LOG"
echo "==========================================" >> "$EXPERIMENT_LOG"
echo "Models: ${MODELS[*]}" >> "$EXPERIMENT_LOG"
echo "Fusion Types: ${FUSION_TYPES[*]}" >> "$EXPERIMENT_LOG"
echo "Alpha Values: ${ALPHAS[*]}" >> "$EXPERIMENT_LOG"
echo "Total Experiments: $((${#MODELS[@]} * ${#FUSION_TYPES[@]} * ${#ALPHAS[@]}))" >> "$EXPERIMENT_LOG"
echo "Partition: volta (V100)" >> "$EXPERIMENT_LOG"
echo "Batch Size: 16 experiments (16 GPUs for parallel execution)" >> "$EXPERIMENT_LOG"
echo "Mode: Alpha Ablation Full Eval (No LR schedulers)" >> "$EXPERIMENT_LOG"
echo "Augmentation: CutMix (α=1.0) + Label Smoothing (0.1) + Gradient Clipping (1.0)" >> "$EXPERIMENT_LOG"
echo "Auto-resubmission: CUDA ECC errors" >> "$EXPERIMENT_LOG"
echo "==========================================" >> "$EXPERIMENT_LOG"

# Initialize failed jobs log
echo "Failed Jobs Log - Comprehensive V100 Partition Experiments (Alpha Ablation Full Eval)" > "$FAILED_JOBS_LOG"
echo "Started: $(date)" >> "$FAILED_JOBS_LOG"
echo "==========================================" >> "$FAILED_JOBS_LOG"

# Initialize resubmission log
echo "Resubmission Log - Comprehensive V100 Partition Experiments (Alpha Ablation Full Eval)" > "$RESUBMISSION_LOG"
echo "Started: $(date)" >> "$RESUBMISSION_LOG"
echo "==========================================" >> "$RESUBMISSION_LOG"

# Show experiment summary
print_status "$CYAN" "=========================================="
print_status "$CYAN" "COMPREHENSIVE V100 PARTITION EXPERIMENT RUNNER (ALPHA ABLATION FULL EVAL)"
print_status "$CYAN" "=========================================="
print_status "$CYAN" "Models: ${MODELS[*]}"
print_status "$CYAN" "Fusion Types: ${FUSION_TYPES[*]}"
print_status "$CYAN" "Alpha Values: ${ALPHAS[*]}"
print_status "$CYAN" "Total Experiments: $((${#MODELS[@]} * ${#FUSION_TYPES[@]} * ${#ALPHAS[@]}))"
print_status "$CYAN" "Partition: volta (V100)"
print_status "$CYAN" "Batch Size: 16 experiments (16 GPUs for parallel execution)"
print_status "$CYAN" "Mode: Alpha Ablation Full Eval (No LR schedulers)"
print_status "$CYAN" "Augmentation: CutMix + Label Smoothing + Gradient Clipping"
print_status "$CYAN" "Auto-resubmission: CUDA ECC errors"
print_status "$CYAN" "=========================================="

# Handle different modes
if [ "$LIST_ONLY" = true ]; then
    list_experiments
    exit 0
fi

if [ "$DRY_RUN" = true ]; then
    dry_run
    exit 0
fi

# Create experiment list
print_status "$BLUE" "Creating experiment combinations..."
experiments=($(create_experiment_list))
print_status "$GREEN" "Created ${#experiments[@]} experiment combinations"

# Run experiments
print_status "$GREEN" "Starting comprehensive experiment execution on V100 partition (Alpha Ablation Full Eval)..."
run_experiment_batch "${experiments[@]}"

# Final summary
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




