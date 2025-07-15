#conda_env: modelmerge
import os
import wandb
import torch
import argparse

from datasets import load_dataset, load_from_disk
from trl import GRPOConfig, GRPOTrainer   

from modelmerge.data import reformat_dataset
from modelmerge.grader import compute_rewards

def parse_args():
    parser = argparse.ArgumentParser(description="Train with GRPO using distributed setup")
    parser.add_argument("--port", type=int, default=8000,
                      help="VLLM port")
    parser.add_argument("--num_generations", type=int, default=3, 
                        help="Number of generations (num gpus * per device batch size)")
    parser.add_argument("--dataset_path", type=str, default="sunyiyou/math_algebra_polynomial_roots_7B_train")
    parser.add_argument("--local_dataset", type=str, default="False", 
                        help="Whether or not dataset is local path")
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--save_total_limit", type=str, default=5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--max_completion_length", type=int, default=2048)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--wandb_run_name", type=str, default="test")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--loss_type", type=str, default="grpo")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    args = parser.parse_args()

    if args.local_dataset.lower() == "true":
        args.local_dataset = True
    else:
        args.local_dataset = False
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

if __name__ == "__main__":
    args = parse_args()
        
    if args.local_dataset:
        print("using local dataset")
        dataset = load_from_disk(args.dataset_path)
    else:
        dataset = load_dataset(args.dataset_path)

    train_dataset = reformat_dataset(dataset["train"])

    if rank in [-1, 0]:
        wandb.init(
            project="modelmerge",
            config=vars(args), 
            name=args.wandb_run_name,
        )

    training_args = GRPOConfig(
        output_dir=args.save_path, 
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        use_vllm=True, 
        vllm_mode="server",
        vllm_server_port=args.port,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_generations = args.num_generations,
        report_to="wandb" if rank in [-1, 0] else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        temperature=args.temperature,
        loss_type=args.loss_type,
           
        #log_completions=True
    )

    #model="Qwen/Qwen2.5-7B-Instruct",
    trainer = GRPOTrainer(
        model=args.model_name,
        reward_funcs=compute_rewards,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
