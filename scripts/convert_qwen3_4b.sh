#!/bin/bash

# 激活环境
export MAMBA_EXE="$HOME/.local/bin/micromamba"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate slime

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export MODEL_ARGS_ROTARY_BASE=5000000
source "${SCRIPT_DIR}/models/qwen3-4B.sh"

export PYTHONPATH=/home/brx/data/Megatron-LM

cd /home/brx/data/slime

export CUDA_VISIBLE_DEVICES=7

# 防止 torch.cuda.device_count() 初始化过早导致的 304 报错
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export WORLD_SIZE=1
export RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /home/brx/models/Qwen3-4B-Instruct-2507 \
    --save /home/brx/models/Qwen3-4B_torch_dist
