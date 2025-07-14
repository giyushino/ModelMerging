#!/bin/bash

source ~/.bashrc
source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate modelmerge 

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export NCCL_TIMEOUT=3600  # 1 hour
export TORCH_DISTRIBUTED_TIMEOUT=3600
export NCCL_BLOCKING_WAIT=1  # Make operations synchronous
# Additional stability settings
export NCCL_ASYNC_ERROR_HANDLING=1
#export NCCL_DEBUG=INFO  # Enable debugging
# make sure nvlink isn't on
export NCCL_P2P_DISABLE=1
#source $WORK/ModelMerging/scripts/training_params.sh

#Killing processes on all GPUs
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

PORT=$(get_random_port)
MAX_COMPLETION_LENGTH=2600
MAX_PROMPT_LENGTH=512
MAX_MODEL_LENGTH=$(($MAX_COMPLETION_LENGTH + $MAX_PROMPT_LENGTH))
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"

echo "Using port" $PORT

CUDA_VISIBLE_DEVICES=3 trl vllm-serve \
    --tensor-parallel-size 1 \
    --model $MODEL_NAME \
    --port "$PORT" \
    --max_model_len $MAX_MODEL_LENGTH \
    --dtype float16 \
    --enable-prefix-caching True \
    --gpu-memory-utilization 0.4 \
    > $WORK/ModelMerging/logs/vllm_server.log 2>&1 &

echo "Waiting for vLLM server…"
while ! curl --silent --fail http://0.0.0.0:$PORT/health/; do
  sleep 1
done
echo "vLLM server started"

NUM_GPUS=3
PER_DEVICE_BATCH_SIZE=2
NUM_GENERATIONS=$(($NUM_GPUS * $PER_DEVICE_BATCH_SIZE))
WANDB_RUN_NAME="qwen1.5b_arithmetic_2048_batch_size6"
SAVE_STEPS=100
export WANDB_DIR=$WORK/ModelMerging


#sunyiyou/math_comp_polynomial_gcd
#sunyiyou/math_algebra_polynomial_roots_7B_train
#sunyiyou/math_arithmetic_gcd_7B_train

DATASET=sunyiyou/math_arithmetic_gcd_7B_train
echo "Starting training"
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_machines 1 --num_processes 3 /home/allanz/ModelMerging/src/modelmerge/train/grpo.py \
    --model_name $MODEL_NAME \
    --port $PORT \
    --num_generations $NUM_GENERATIONS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --dataset_path $DATASET \
    --local_dataset false \
    --save_path $WORK/ModelMerging/checkpoints/$WANDB_RUN_NAME \
    --wandb_run_name $WANDB_RUN_NAME \
    --save_steps $SAVE_STEPS \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --max_prompt_length $MAX_PROMPT_LENGTH


echo "Killing any existing GPU processes..."
nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9 > /dev/null 2>&1
