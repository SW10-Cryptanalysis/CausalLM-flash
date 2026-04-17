#!/bin/bash
set -eo pipefail

# Navigate to your mounted workspace
cd /work

# Clone the repository and specific branch if it doesn't exist yet
if [ ! -d "Llama-xFormers" ]; then
    echo "Cloning repository..."
    git clone -b UCloud https://github.com/SW10-Cryptanalysis/Llama-xFormers.git
    cd Llama-xFormers
else
    echo "Git pulling newest changes..."
    cd Llama-xFormers
    git pull
fi

mkdir -p logs
export HF_HUB_ENABLE_HF_TRANSFER=1
export UV_CACHE_DIR="/work/.uv_cache"

# Dynamically count available GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPU(s)"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export CUDA_DEVICE_MAX_CONNECTIONS=1

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
if [ "$NUM_GPUS" -gt 1 ]; then
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
    export NCCL_P2P_LEVEL=SYS
    export NCCL_NET_GDR_LEVEL=SYS
fi

# ── Virtual environment ────────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "Creating new uv virtual environment..."
    uv venv
fi

uv pip install -e .
uv pip install hf_transfer

echo "Installing Flash Attention 2..."
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu130torch2.10-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl

MASTER_PORT=$((10000 + $RANDOM % 20000))

# ── Launch ─────────────────────────────────────────────────────────────────────
echo "Launching torchrun with $NUM_GPUS processes..."
uv run torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    -m src.train "$@"

echo "Training Job finished at $(date)"
