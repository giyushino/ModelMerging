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

source $WORK/ModelMerging/scripts/training_params.sh
get_random_port() {
  local port
  # IANA suggests using ports 49152-65535 for dynamic/private ports
  local min_port=49152
  local max_port=65535
  port=$((RANDOM % (max_port - min_port + 1) + min_port))
  echo $port
}
PORT=$(get_random_port)

# Print configuration
echo "=============================================="
echo "Training Configuration:"
echo "Base Model            : $BASE_MODEL"
echo "Dataset Path          : $DATASET" 
echo "Local Dataset         : $LOCAL_DATASET"
echo "Number of Epochs      : $NUM_EPOCHS"
echo "Learning Rate         : $LEARNING_RATE"
echo "Weight Decay          : $WEIGHT_DECAY"
echo "Gradient Accum Steps  : $GRAD_ACCUMULATION_STEPS"
echo "Logging Steps         : $LOGGING_STEPS"
echo "Save Steps            : $SAVE_STEPS"
echo "Total GPUs            : $NUM_GPUS"
echo "Per Device Batch Size : $PER_DEVICE_BATCH_SIZE"
echo "Num Generations       : $NUM_GENERATIONS"
echo "Effective Batch Size  : $(($PER_DEVICE_BATCH_SIZE * $(((NUM_GPUS - 1))) * $GRAD_ACCUMULATION_STEPS))"
echo "WANDB Run Name        : $WANDB_RUN_NAME"
echo "Port for vLLM server  : $PORT" 
#echo "Resumed Checkpoint    : $RESUME_FROM_CHECKPOINT" haven't implemented this yet
#echo "Resume Checkpoint     : $RESUME_FROM_CHECKPOINT"
echo "=============================================="
echo "GRPO Specific Parameters:"
echo "Scale Rewards         : $SCALE_REWARDS"
echo "Loss Type             : $LOSS_TYPE"
echo "=============================================="


VLLM_GPU=$((NUM_GPUS - 1))
MAX_MODEL_LENGTH=$(($MAX_COMPLETION_LENGTH + $MAX_PROMPT_LENGTH))

CUDA_VISIBLE_DEVICES=$VLLM_GPU trl vllm-serve \
    --tensor-parallel-size 1 \
    --model $BASE_MODEL \
    --port "$PORT" \
    --max_model_len $MAX_MODEL_LENGTH \
    --dtype float16 \
    --enable-prefix-caching True \
    --gpu-memory-utilization 0.4 \
    > $WORK/ModelMerging/logs/vllm_server.log 2>&1 &

echo "Waiting for vLLM server to start on GPU $VLLM_GPU"
while ! curl --silent --fail http://0.0.0.0:$PORT/health/; do
  sleep 1
done
echo "vLLM server started on GPU $VLLM_GPU" 


export WANDB_DIR=$WORK/ModelMerging
echo "Starting training on GPUs $(seq -s, 0 $((NUM_GPUS - 2)))"

CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 2))) accelerate launch --num_machines 1 --num_processes $((NUM_GPUS - 1)) /home/allanz/ModelMerging/src/modelmerge/train/grpo.py \
    --model_name $BASE_MODEL \
    --port $PORT \
    --num_generations $NUM_GENERATIONS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --dataset_path $DATASET \
    --local_dataset false \
    --save_path $WORK/ModelMerging/checkpoints/$WANDB_RUN_NAME \
    --wandb_run_name $WANDB_RUN_NAME \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --temperature $TEMPERATURE \
    --num_train_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --loss_type $LOSS_TYPE \
    --gradient_accumulation_steps $GRAD_ACCUMULATION_STEPS


echo "Shutting down vLLM server on GPU $VLLM_GPU"
nvidia-smi | grep 'python' \
  | awk -v gpu="$VLLM_GPU" '$2 == gpu { print $5 }' \
  | xargs -r -n1 kill -9 > /dev/null 2>&1

echo "All finished"
