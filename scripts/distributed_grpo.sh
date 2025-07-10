#!/bin/bash

echo "Setting up distributed environment variables..."
export MASTER_PORT=29500
TOTAL_GPUS=3
export WORLD_SIZE=$TOTAL_GPUS
export MASTER_ADDR="localhost"

# Source shared parameters
source $WORK/ModelMerging/scripts/training_params.sh

NUM_EPOCHS=${1:-1}
LOG_PROB_MULTIPLIER=${7:-1}
DATASET=${8:-$DATASET}
BASE_MODEL=${9:-$BASE_MODEL}
PER_DEVICE_BATCH_SIZE=${10:-$PER_DEVICE_BATCH_SIZE}
NUM_GENERATIONS=${11:-$NUM_GENERATIONS}
RESUME_FROM_CHECKPOINT=${12:-""}

MAXIMIZE_THROUGHPUT=False


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
MAX_MODEL_LEN=$(($MAX_PROMPT_LENGTH + $MAX_COMPLETION_LENGTH))
CUDA_VISIBLE_DEVICES=3 python $WORK/ModelMerging/src/modelmerge/vllm/vllm_serve.py \
    --model $BASE_MODEL \
    --max_model_len $MAX_MODEL_LEN \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --host 0.0.0.0 \
    --port $PORT \
    --return_texts True \
    > $WORK/ModelMerging/vllm_server.log 2>&1 &
sleep 30

echo "Starting distributed training with DeepSpeed..."
# Run training with DeepSpeed

deepspeed --include localhost:0,1,2 \
    --master_port 29501 \
    $WORK/ModelMerging/src/modelmerge/train/grpo.py \
    --model_name_or_path $BASE_MODEL \
    --dataset_name $DATASET \
    --curriculum $CURRICULUM \
    --output_dir checkpoints/ \
    --resume_from_checkpoint $RESUME_FROM_CHECKPOINT \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --gradient_accumulation_steps $GRAD_ACCUMULATION_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --logging_steps $LOGGING_STEPS \
    --save_strategy steps \
    --save_steps $SAVE_STEPS \
    --local_dataset \
    --num_generations $NUM_GENERATIONS \
    --temperature $TEMPERATURE \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --scale_rewards $SCALE_REWARDS \
    --deepspeed $DS_CONFIG_PATH \
    --use_vllm True \
    --vllm_server_host 0.0.0.0 \
    --vllm_server_port $PORT \

echo "Job completed successfully!"
#--loss_type \


