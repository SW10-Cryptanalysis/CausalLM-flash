#!/bin/bash
#SBATCH --job-name=llama
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=l4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:4

set -e
set -o pipefail

# 1. THE PROJECT DIRECTORY
PROJECT_DIR="/ceph/project/SW10-CausalLM/Llama-xFormers"
cd $PROJECT_DIR || exit 1
mkdir -p logs

# 3. DYNAMIC PATH CHECK
IF_SPACES="normal"
for arg in "$@"; do
    if [ "$arg" == "--spaces" ]; then
        IF_SPACES="spaces"
    fi
done

# 4. COMPLETION CHECK
if [ -d "outputs/${IF_SPACES}/final_model" ]; then
    echo "Success: Training is finished. Breaking chain."
    exit 0
fi

SCRIPT_NAME="train.slurm" 
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Queuing continuation job..."
    sbatch --dependency=afterany:$SLURM_JOB_ID "$SCRIPT_NAME" "$@"
fi

# 6. GPU ENVIRONMENT
NUM_GPUS=4
export HF_HUB_ENABLE_HF_TRANSFER=1
export CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f1-$NUM_GPUS)

export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# 7. INSTALLATION
if [ ! -d ".venv" ]; then
    uv pip install --python 3.12 -e .
    uv pip install --python 3.12 https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
fi

MASTER_PORT=$((10000 + $RANDOM % 20000))

echo "Job started on $(hostname) at $(date)" | tee -a "logs/train_live_${SLURM_JOB_ID}.log"
nvidia-smi | tee -a "logs/train_live_${SLURM_JOB_ID}.log"

# 8. EXECUTION
# We use timeout to stop torchrun 5 minutes early so it can exit cleanly
timeout 11h 55m uv run torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    src/train.py "$@" 2>&1 | tee -a "logs/train_live_${SLURM_JOB_ID}.log" || true

echo "Job cycle finished at $(date)" | tee -a "logs/train_live_${SLURM_JOB_ID}.log"