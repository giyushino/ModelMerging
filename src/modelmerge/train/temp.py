#conda_env: modelmerge
import os
import torch
import argparse

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer   

from modelmerge.data import reformat_dataset
from modelmerge.grader import compute_rewards

def parse_args():
    parser = argparse.ArgumentParser(description="Train with GRPO using distributed setup")
    parser.add_argument("--port", type=int, default=8000,
                      help="VLLM port")
    parser.add_argument("--local_rank", type=int, default=-1) 
    parser.add_argument("--deepspeed", action="store_true")
    args = parser.parse_args()
    return args 



def setup_distributed_training():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    rank = int(os.environ.get("RANK", -1))

    if torch.cuda.is_available() and local_rank != -1:
        torch.cuda.set_device(local_rank) 

    print(f"Local Rank: {local_rank}, World Size: {world_size}, Rank: {rank}")
    return local_rank, world_size, rank


local_rank, world_size, rank = setup_distributed_training()
if torch.cuda.is_available():
    print(f"[Rank {rank}] CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[Rank {rank}] torch.cuda.current_device() = {torch.cuda.current_device()}")
    print(f"[Rank {rank}] device name = {torch.cuda.get_device_name(torch.cuda.current_device())}")


dataset = load_dataset("sunyiyou/math_algebra_polynomial_roots_7B_train")
train_dataset = reformat_dataset(dataset, "train")

args = parse_args()
training_args = GRPOConfig(
    output_dir="test", 
    use_vllm=True, 
    vllm_mode="server",
    vllm_server_port=args.port
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-7B-Instruct",
    reward_funcs=compute_rewards,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
