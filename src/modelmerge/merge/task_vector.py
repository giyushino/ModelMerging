#conda_env: modelmerge
import argparse
import torch.nn as nn

from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--max_completion_length", type=int, default=2048)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--wandb_run_name", type=str, default="test")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()

    return args 


class TaskVector:
    def __init__(self, pretrained: str, finetuned: List[str], strength: List[float], specific_layers: Optional[List[str]]) -> None:
        """
        Args:
            pretrained (str): Path to pretrained model on HuggingFace
            finetuned (list): List of path to finetuned checkpoints
            strength (list): List of floats indicating coefficient each task vector should be multiplied by 
            specific_layers (list): List of layers to merge. Leaving it as None merges all layers 
        Returns: 
            None
        """
        self.pretrained = AutoModelForCausalLM.from_pretrained(pretrained) 
        self.finetuned = [AutoModelForCausalLM.from_pretrained(checkpoint) for checkpoint in finetuned]
        self.strength = strength
        if specific_layers:
            self.specific_layers = set(specific_layers)
        else:
            self.specific_layers = None
    
    def merge(self):
        """
        Merges the pretrained model with the finetuned!
        """

        if len(self.finetuned) > len(self.strength):
            print("Number of finetuned models does not match number of strength coefficients, defaulting to 1.0")

        for _ in range(len(self.finetuned) - len(self.strength)):
            self.strength.append(1.0)

        #pretrained_state_dict = self.pretrained.state_dict()
        finetuned_state_dict = [model.state_dict() for model in self.finetuned]
        # go through each layer and add the task vectors
        for name, param in self.pretrained.named_parameters():
            if self.specific_layers is None or name in self.specific_layers:
                print(f"merging {name}")
                original_tensor = param.clone()
                for index, model in enumerate(finetuned_state_dict):
                    param.data += (model[name] - original_tensor) * self.strength[index] 
        
    
task = TaskVector("Qwen/Qwen2.5-1.5B-Instruct", ["/home/allanz/ModelMerging/checkpoints/qwen1.5b_arithmetic/checkpoint-1000/", "/home/allanz/ModelMerging/checkpoints/qwen1.5b_arithmetic/checkpoint-1400/"], [1., 0.5], None)
task.merge()
