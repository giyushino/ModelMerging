#conda_env: modelmerge
import time
import argparse

from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Train with GRPO using distributed setup")
    parser.add_argument("--model", type=str, required=False, help="path to pretrained model")
    parser.add_argument("--models", type=str, nargs="+", required=False, help="path to finetuned model")
    parser.add_argument("--strength", type=float, nargs="+", help="List of strength coefficients") 
    parser.add_argument("--save_path", type=str, help="path to save merged model")

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
        return self.pretrained
        

if __name__ == "__main__":
    args = parse_args()
    merge = TaskVector(args.model, args.models, args.strength, None)
    t0 = time.time()    
    model = merge.merge()
    t1 = time.time() 
    print(f"Saving model to {args.save_path}")
    model.save_pretrained(args.save_path)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.save_path)
    t2 = time.time()
    print(f"Took {(t1 - t0):.4f} seconds to merge models")
    print(f"Took {(t2 - t1):.4f} seconds to save models")
    
