#!/bin/bash

export TORCH_DISTRIBUTED_TIMEOUT=3600
export NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO
export NCL_P2P_DISABLE=1



echo "Killing any existing GPU processes..."
nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9 > /dev/null 2>&1
sleep 15


get_random_port() {
  local port
  # IANA suggests using ports 49152-65535 for dynamic/private ports
  local min_port=49152
  local max_port=65535
  port=$((RANDOM % (max_port - min_port + 1) + min_port))
  echo $port
}

BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"  # Pre-trained on math data
MAX_COMPLETION_LENGTH=2048
MAX_PROMPT_LENGTH=1024
PORT=$(get_random_port)
MAX_MODEL_LEN=$(($MAX_PROMPT_LENGTH + $MAX_COMPLETION_LENGTH))
CUDA_VISIBLE_DEVICES=3 python $WORK/eff_grpo/src/eff_grpo/vllm/vllm_serve.py \
    --model $BASE_MODEL \
    --max_model_len $MAX_MODEL_LEN \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --host 0.0.0.0 \
    --port $PORT \
    --return_texts True \
    > $WORK/ModelMerging/logs/vllm_server.log 2>&1 &
sleep 30

echo "Using port" $PORT
#--master_port 29501 \
#CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc-per-node=4 --nnodes=1 --master_port 29501 \
#    /home/allanz/ModelMerging/src/modelmerge/train/temp.py
deepspeed --include localhost:0,1,2 --master_port 29501 /home/allanz/ModelMerging/src/modelmerge/train/temp.py --port $PORT





