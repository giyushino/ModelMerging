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

PORT=$(get_random_port)
echo "Using port" $PORT



CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.5 \
    --max-model-len 4096 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 8192 \
    --served-model-name Qwen/Qwen2.5-1.5B-Instruct \
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


#sunyiyou/math_comp_polynomial_gcd
#sunyiyou/math_algebra_polynomial_roots_7B_train
#sunyiyou/math_arithmetic_gcd_7B_train

DATASET="sunyiyou/math_comp_polynomial_gcd"
python $WORK/ModelMerging/src/modelmerge/eval/accuracy.py \
    --dataset_path $DATASET \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --port $PORT \
    --max_completion_length 2048

echo "Killing any existing GPU processes..."
nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9 > /dev/null 2>&1
