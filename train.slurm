#!/bin/bash
#SBATCH --job-name=llama
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=l4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:l4:4

set -e

PROJECT_DIR=/ceph/project/SW10-CausalLM/Llama-xFormers
cd $PROJECT_DIR

# ==============================================================================
# DYNAMIC AUTO-CHAINING LOGIC
# ==============================================================================
# 1. Determine which subfolder to check based on the --spaces flag
IF_SPACES="normal"
for arg in "$@"; do
    if [ "$arg" == "--spaces" ]; then
        IF_SPACES="spaces"
    fi
done

# 2. Break the chain if the specific final model is found
if [ -d "outputs/${IF_SPACES}/final_model" ]; then
    echo "Success: 'outputs/${IF_SPACES}/final_model' found. Training is completely finished."
    echo "Breaking the chain."
    exit 0
fi

echo "Dynamically queuing the next 12-hour continuation job with arguments: $@"
sbatch --dependency=afterany:$SLURM_JOB_ID $0 "$@"
# ==============================================================================

mkdir -p logs

export HF_HUB_ENABLE_HF_TRANSFER=1

export CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES)

nvidia-smi | tee -a logs/train_live_${SLURM_JOB_ID}.log

export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=8

# NCCL Optimizations for FSDP on L4s
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Ensure environment is synced
uv pip install --python 3.12 -e .

# Mistral relies heavily on FA2 for sliding window attention.
uv pip install --python 3.12 https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# Ensure master_port is randomized slightly to prevent port collision on shared nodes
MASTER_PORT=$((10000 + $RANDOM % 20000))

# Launch with torchrun. Hugging Face Trainer automatically detects the distributed 
# environment spawned by torchrun and applies the FSDP arguments defined in train.py
echo "Launching torchrun with arguments: $@" | tee -a logs/train_live_${SLURM_JOB_ID}.log

uv run torchrun \
    --nproc_per_node=4 \
    --master_port=$MASTER_PORT \
    src/train.py "$@" 2>&1 | tee -a logs/train_live_${SLURM_JOB_ID}.log