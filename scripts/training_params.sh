#!/bin/bash

###################
# Training Parameters
###################

# Model and dataset
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"  # Pre-trained on math data
DATASET="sunyiyou/math_algebra_polynomial_roots_7B_train"

# Training hyperparameters
NUM_EPOCHS=1
LEARNING_RATE=3e-6
WEIGHT_DECAY=0.0
GRAD_ACCUMULATION_STEPS=1
WARMUP_STEPS=0

# Generation parameters
TEMPERATURE=1.0
NUM_GENERATIONS=3
MAX_COMPLETION_LENGTH=2048
MAX_PROMPT_LENGTH=1024
PER_DEVICE_BATCH_SIZE=1
# Logging parameters
LOGGING_STEPS=1
SAVE_STEPS=2000

# GRPO parameters
SCALE_REWARDS=False

#--beta 0.0
#--num_unique_prompts_rollout 128
#--num_samples_per_prompt_rollout 64
#--kl_estimator kl3
#--learning_rate 5e-7
#--max_token_length 8192
#--max_prompt_token_length 2048
#--response_length 6336
#--pack_length 8384
#--apply_r1_style_format_reward True
#--apply_verifiable_reward True
#--non_stop_penalty True
#--non_stop_penalty_value 0.0
#--chat_template_name r1_simple_chat_postpend_think
#--temperature 1.0
#--masked_mean_axis 1
#--total_episodes 20000000
#--deepspeed_stage 2
#--per_device_train_batch_size 1
#--num_mini_batches 1
#--num_learners_per_node 8 8
#--num_epochs 1
#--vllm_tensor_parallel_size 1
#--vllm_num_engines 16
#--lr_scheduler_type linear
#--seed 3
#--num_evals 200
