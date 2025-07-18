#!/bin/bash

source ~/.bashrc
source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate modelmerge

get_random_port() {
  local port
  # IANA suggests using ports 49152-65535 for dynamic/private ports
  local min_port=49152
  local max_port=65535
  port=$((RANDOM % (max_port - min_port + 1) + min_port))
  echo $port
}


#sunyiyou/math_comp_polynomial_gcd
#sunyiyou/math_algebra_polynomial_roots_7B_train
#sunyiyou/math_arithmetic_gcd_7B_train

MAX_COMPLETION_LENGTH=2048
MAX_PROMPT_LENGTH=512
MAX_MODEL_LENGTH=$(($MAX_COMPLETION_LENGTH + $MAX_PROMPT_LENGTH))
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
MODEL_1=/home/allanz/ModelMerging/checkpoints/qwen1.5b_algebra_1600_batch_size8/checkpoint-2700/
MODEL_2=/home/allanz/ModelMerging/checkpoints/qwen1.5b_arithmetic_1600_batch_size8/checkpoint-2700/
MODELS=($MODEL_1 $MODEL_2)
MODEL_1_STRENGTH=0.7
MODEL_2_STRENGTH=1.2
MODEL_STRENGTHS=($MODEL_1_STRENGTH $MODEL_2_STRENGTH)
DATASET="sunyiyou/math_comp_polynomial_gcd"
VLLM_GPU=0
BATCH_SIZE=12
MERGE=true
SAVE=true
#MODEL=$MODEL_2


if [ "$MERGE" = false ]; then
    PORT=$(get_random_port)
    echo "Using port" $PORT
    CUDA_VISIBLE_DEVICES=$VLLM_GPU python -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --host 0.0.0.0 \
        --port $PORT \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.5 \
        --max-model-len $MAX_MODEL_LENGTH \
        --max-num-seqs 256 \
        --max-num-batched-tokens 8192 \
        --served-model-name $MODEL \
        --disable-log-requests \
        --trust-remote-code \
        --enable-prefix-caching \
        --use-v2-block-manager \
        > $WORK/ModelMerging/logs/vllm_server_accuracy.log 2>&1 &

    echo "Waiting for vLLM server…"
    while ! curl --silent --fail http://0.0.0.0:$PORT/health/; do
      sleep 1
    done
    echo "vLLM server started"

    python $WORK/ModelMerging/src/modelmerge/eval/accuracy.py \
        --dataset_path $DATASET \
        --model $MODEL \
        --port $PORT \
        --max_completion_length $MAX_COMPLETION_LENGTH \

    echo "Shutting down vLLM server on GPU $VLLM_GPU"
    nvidia-smi | grep 'python' \
      | awk -v gpu="$VLLM_GPU" '$2 == gpu { print $5 }' \
      | xargs -r -n1 kill -9 > /dev/null 2>&1
else
    CUDA_VISIBLE_DEVICES=$VLLM_GPU python $WORK/ModelMerging/src/modelmerge/eval/accuracy_merge.py \
        --dataset_path $DATASET \
        --model $MODEL \
        --max_completion_length $MAX_COMPLETION_LENGTH \
        --models "${MODELS[@]}" \
        --strength "${MODEL_STRENGTHS[@]}" \
        --batch_size $BATCH_SIZE
fi

echo "All finished"
