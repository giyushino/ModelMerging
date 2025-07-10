#conda_env: modelmerge
import os
import datasets 
import argparse

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer   


def parse_args():
    parser = argparse.ArgumentParser(description="Train with GRPO using distributed setup")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                      help="Path to pretrained model or model identifier from huggingface.co/models")
    
    # GRPO Parameters
    parser.add_argument("--scale_rewards", type=str, default="True",
                      help="Scale rewards")
    parser.add_argument("--loss_type", type=str, default="grpo",
                      help="Loss type")
    parser.add_argument("--use_vllm", type=str, default="False",
                      help="Use VLLM")
    parser.add_argument("--vllm_server_host", type=str, default="0.0.0.0",
                      help="VLLM host")
    parser.add_argument("--vllm_server_port", type=int, default=8000,
                      help="VLLM port")
    
    # Debug Parameters
    parser.add_argument("--debug_grpo", type=str, default="False",
                      help="Debug")
    
    return parser.parse_args()

def setup_distributed_training():
    # Set up distributed training environment variables
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    rank = int(os.environ.get("RANK", -1))
    
    print(f"Local Rank: {local_rank}, World Size: {world_size}, Rank: {rank}")
    return local_rank, world_size, rank

#model_name = "Qwen/Qwen2.5-7B-Instruct"

"""
sunyiyou/math_algebra_polynomial_roots_7B_train
Qwen/Qwen2.5-7B-Instruct
"""

def main():
    args = parse_args()
    
    local_rank, world_size, rank = setup_distributed_training()
    # Set up output directory
    output_dir = os.path.join(args.output_dir, args.wandb_run_name)
    
    # Validate checkpoint path if provided
    resume_checkpoint = None
    if args.resume_from_checkpoint:
        resume_checkpoint = validate_checkpoint_path(args.resume_from_checkpoint, output_dir, args.wandb_run_name)
        if resume_checkpoint:
            checkpoint_info = get_checkpoint_info(resume_checkpoint)
            if checkpoint_info:
                logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
                logger.info(f"Checkpoint info: global_step={checkpoint_info['global_step']}, epoch={checkpoint_info['epoch']}")
        else:
            logger.warning("Invalid checkpoint path provided, starting from scratch")
    
    # Initialize wandb if main process
    if rank in [-1, 0]:
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "eff_grpo"),
            config=vars(args),
            name=args.wandb_run_name,
            resume="allow" if resume_checkpoint else None
        )

    # Set reward functions and weights
    reward_funcs = [grpo_boxed_reward_fn]
    reward_weights = [1.0] 
    
    # Load dataset
    # Only load and process dataset on rank 0
    if rank in [-1, 0]:
        train_dataset = load_dataset(args.dataset_name, split="train") if not args.local_dataset else load_from_disk(args.dataset_name)["train"]
        
        # Format Data
        def apply_qwen_math_template(question: str):
            return (
                "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
                + question
                + "<|im_end|>\n<|im_start|>assistant\n"
            )

        def add_prompt(row):
            row["prompt"] = apply_qwen_math_template(row["problem"])
            return row
        train_dataset = train_dataset.map(add_prompt)

        # Subset dataset
        subset_indices = list(range(len(train_dataset)))
        if os.path.exists(args.subset_file_path):
            logger.info(f"Loading subset indices from {args.subset_file_path}")
            subset_indices = np.load(args.subset_file_path)
        else:
            logger.info(f"No subset indices file found at {args.subset_file_path}")
            subset_file_path = os.path.join(os.environ["WORK"], "eff_grpo", "data", "subset_indices", f"{args.wandb_run_name}_indices.npy")
            if os.path.exists(subset_file_path):
                logger.info(f"Loading subset indices from {subset_file_path}")
                subset_indices = np.load(subset_file_path)
            else:
                logger.info(f"No subset indices file found at {subset_file_path}, using full dataset")
        train_dataset = train_dataset.select(subset_indices)

    # Broadcast dataset from rank 0 to all other processes
    if world_size > 1:
        import accelerate
        train_dataset = accelerate.utils.broadcast_object_list([train_dataset if rank in [-1, 0] else None], from_process=0)[0]
     
    training_args = CustomGRPOConfig(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size * args.batch_size_multiplier * args.prompt_multiplier * args.log_prob_multiplier,
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
        report_to="wandb" if rank in [-1, 0] else None,
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
        batch_size_multiplier=args.batch_size_multiplier,
        use_vllm=args.use_vllm,
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
        use_cppo=args.use_cppo,
        cppo_multiplier=args.cppo_multiplier,
        debug_grpo=args.debug_grpo,
        prompt_multiplier=args.prompt_multiplier,
        only_positive_adv=args.only_positive_adv,
        log_prob_multiplier=args.log_prob_multiplier,
        maximize_throughput=args.maximize_throughput
    )

    if args.curriculum == True:
        # Get the starting step for completions when resuming
        last_completion = 0
        if resume_checkpoint:
            checkpoint_info = get_checkpoint_info(resume_checkpoint)
            if checkpoint_info:
                last_completion = checkpoint_info['global_step']
                logger.info(f"Setting last_completion to {last_completion} for curriculum callback")
        
        # Initialize trainer        
        trainer = CustomGRPOTrainer(
            model=args.model_name_or_path,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=train_dataset, 
            callbacks=[EndOfEpochCallback(
                                          args.data_selection_strategy, 
                                          args.data_selection_slice, 
                                          args.data_selection_ratio, 
                                          args.wandb_run_name,
                                          last_completion=last_completion
                                          )]
        )

    # normal training, default
    else:
        trainer = CustomGRPOTrainer(
            model=args.model_name_or_path,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=train_dataset, 
        )
    
    # Train the model with resume capability
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    # Save final model if main process
    if rank in [-1, 0]:
        trainer.save_model()
        wandb.finish()

"""
def install_signal_handlers(trainer):
    def handle_sigterm(signum, frame):
        print("SIGTERM received. Saving model...")
        trainer.save_model(output_dir=os.path.join(trainer.args.output_dir, "emergency_save"))
        sys.exit(0)

    def handle_sigint(signum, frame):
        print("SIGINT (Ctrl+C) received. Saving model...")
        trainer.save_model(output_dir=os.path.join(trainer.args.output_dir, "emergency_save"))
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigint)

# Then before training starts:
install_signal_handlers(trainer)
"""


if __name__ == "__main__":
    main()
