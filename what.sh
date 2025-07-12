#!/bin/bash

source ~/.bashrc
source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate modelmerge 

export TORCH_DISTRIBUTED_TIMEOUT=3600
export NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO
export NCL_P2P_DISABLE=1



CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B \
    --host 0.0.0.0 \
    --port 11111 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 4096 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 8192 \
    --served-model-name qwen3-4b \
    --disable-log-requests \
    --trust-remote-code \
    --enable-prefix-caching \
    --use-v2-block-manager \
    > vllm_server.log 2>&1 &

sleep 60


deepspeed --include localhost:0,1,2 --master_port 29501 /home/allanz/ModelMerging/src/modelmerge/train/temp.py --port 11111






