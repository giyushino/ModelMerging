#!/bin/bash

###################
# Training Parameters
###################

# other datasets
#sunyiyou/math_comp_polynomial_gcd
#sunyiyou/math_algebra_polynomial_roots_7B_train
#sunyiyou/math_arithmetic_gcd_7B_train


# Model and dataset
BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
DATASET="sunyiyou/math_algebra_polynomial_roots_7B_train"
LOCAL_DATASET="false"

# Training hyperparameters
NUM_EPOCHS=1 
LEARNING_RATE=3e-6
WEIGHT_DECAY=0.0
GRAD_ACCUMULATION_STEPS=1

# Generation parameters
TEMPERATURE=1.0
NUM_GPUS=4 #Number of GPUs available to us -> one will be used to host the vllm server, so akin to training with with NUM_GPUS - 1 
MAX_COMPLETION_LENGTH=2048
MAX_PROMPT_LENGTH=1024
PER_DEVICE_BATCH_SIZE=2
NUM_GENERATIONS=$(($(((NUM_GPUS - 1))) * $PER_DEVICE_BATCH_SIZE))

# Logging parameters
LOGGING_STEPS=1
SAVE_STEPS=2000
SAVE_TOTAL_LIMIT=5
WANDB_RUN_NAME="${BASE_MODEL//\//_}_${DATASET//\//_}_${NUM_GENERATIONS}"

# GRPO parameters
SCALE_REWARDS=False # haven't implemented this yet
LOSS_TYPE="grpo"
