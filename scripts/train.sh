#!/bin/bash

export MASTER_PORT=29500
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export NCCL_TIMEOUT=3600  # 1 hour
export TORCH_DISTRIBUTED_TIMEOUT=3600
export NCCL_BLOCKING_WAIT=1  # Make operations synchronous
# Additional stability settings
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO  # Enable debugging
# make sure nvlink isn't on
export NCCL_P2P_DISABLE=1


#Killing processes on all GPUs
echo "Killing any existing GPU processes..."
nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9 > /dev/null 2>&1
sleep 15
echo "Starting VLLM server..."

get_random_port() {
  local port
  # IANA suggests using ports 49152-65535 for dynamic/private ports
  local min_port=49152
  local max_port=65535
  port=$((RANDOM % (max_port - min_port + 1) + min_port))
  echo $port
}

PORT=$(get_random_port)
echo "Using port" $PORT
CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
    --tensor-parallel-size 1 \
    --model Qwen/Qwen3-0.6B \
    --port "$PORT" \
    --max_model_len 2048 \
    --dtype float16 \
    --enable-prefix-caching True \
    --gpu-memory-utilization 0.65 &
    #> $WORK/ModelMerging/logs/vllm_server.log 2>&1 &

while ! curl --silent --fail http://0.0.0.0:$PORT/health/; do
  echo "Waiting for vLLM server…"
  sleep 1
done

# 3 gpus, 2 completions per prompt. change this later
echo "Starting training"
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num_machines 1 --num_processes 3 /home/allanz/ModelMerging/src/modelmerge/train/grpo.py \
    --port $PORT \
    --num_generations 4

