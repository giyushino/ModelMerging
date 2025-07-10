#conda_env: modelmerge
import os
import argparse

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer   

from modelmerge.data import reformat_dataset
from modelmerge.grader import compute_rewards

def parse_args():
    parser = argparse.ArgumentParser(description="Train with GRPO using distributed setup")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                      help="Path to pretrained model or model identifier from huggingface.co/models")

    # GRPO Parameters
    parser.add_argument("--scale_rewards", type=str, default="True",
                      help="Scale rewards")
    parser.add_argument("--loss_type", type=str, default="grpo",
                      help="Loss type")
    parser.add_argument("--use_vllm", type=str, default="True",
                      help="Use VLLM")
    parser.add_argument("--vllm_server_host", type=str, default="0.0.0.0",
                      help="VLLM host")
    parser.add_argument("--vllm_server_port", type=int, default=8000,
                      help="VLLM port")
    
    # Debug Parameters
    parser.add_argument("--debug_grpo", type=str, default="False",
                      help="Debug")
    args = parser.parse_args()
    if args.use_vlm == "True":
        args.use_vllm = True 
    return args 

def setup_distributed_training():
    # Set up distributed training environment variables
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    rank = int(os.environ.get("RANK", -1))
    
    print(f"Local Rank: {local_rank}, World Size: {world_size}, Rank: {rank}")
    return local_rank, world_size, rank

#model_name = "Qwen/Qwen2.5-7B-Instruct"

# note to self: i think they use dr.grpo? maybe not
def main():
    args = parse_args()
    
    local_rank, world_size, rank = setup_distributed_training()
    # Set up output directory
    output_dir = os.path.join(args.output_dir, args.wandb_run_name)
    
    reward_funcs = [compute_rewards]
    reward_weights = [1.0] 
    
    # Load dataset
    # Only load and process dataset on rank 0
    if rank in [-1, 0]:
        train_dataset = load_dataset(args.dataset_name)
        reformatted_dataset = reformat_dataset(train_dataset, split="train")

    # Broadcast dataset from rank 0 to all other processes
    if world_size > 1:
        import accelerate
        train_dataset = accelerate.utils.broadcast_object_list([reformatted_dataset if rank in [-1, 0] else None], from_process=0)[0]
    

    training_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size 
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        save_total_limit=5,
        local_rank=local_rank,
        #report_to="wandb" if rank in [-1, 0] else None,
        ddp_find_unused_parameters=False,
        num_generations=args.num_generations,
        log_completions=True,
        reward_weights=reward_weights,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        temperature=args.temperature,
        top_p=args.top_p,
        scale_rewards=args.scale_rewards,
        loss_type=args.loss_type,
        model_init_kwargs={"torch_dtype": "bfloat16"},
        use_vllm=args.use_vllm,
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
    )

    
    # Initialize trainer        
    trainer = GRPOTrainer(
        model=args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset, 
    )
    
    # Train the model with resume capability
    trainer.train()
    
    # Save final model if main process
    if rank in [-1, 0]:
        trainer.save_model()



if __name__ == "__main__":
    main()
