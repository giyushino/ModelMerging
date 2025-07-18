import re
import time
import torch
import argparse

from datasets import load_dataset
from transformers.pipelines import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

from modelmerge.data import reformat_dataset
from modelmerge.merge import task_vector
from modelmerge.merge.task_vector import TaskVector


def parse_args():
    parser = argparse.ArgumentParser(description="Check completions for sparse errors")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to local dataset, or huggingface id")
    parser.add_argument("--model", type=str, required=True, help="model id")
    parser.add_argument("--max_completion_length", type=int, required=True, help="model length")
    parser.add_argument("--models", type=str, nargs="+", required=False, help="path to finetuned model")
    parser.add_argument("--strength", type=float, nargs="+", help="List of strength coefficients") 
    parser.add_argument("--batch_size", type=int, help="batch size") 

    args = parser.parse_args()
    return args


def extract_solution(output):
    """
    Use regex to extract boxed solution from model output
    Searching for \\{boxed}
    """ 
    # if multiple \\boxed within output, get the last occurance
    match = re.search(r'\\boxed\{([^{}]*)\}(?!.*\\boxed\{)', output)
    if match: 
        return match.group(1)
    return None


def compute_rewards(completion, ground_truth):
    """
    Extract \\boxed content from completion and compare to ground truth, return list of rewards
    """
    solution = extract_solution(completion)
    if solution: 
        print(f"model answer: {solution} ||ground truth: {ground_truth}")
        if str(ground_truth) == solution:
            print("correct")
            return 1
    print("formatted incorrectly")
    return 0

def load_model(pretrained, finetuned, strength):
    """
    Args:
        pretrained (str): path to default pretrained model
        finetuned (list(str)) : list containing paths of strengths
        strenth: 
    """
    merge = TaskVector(pretrained, finetuned, strength, None)
    merged_model = merge.merge()
    return merged_model


def score_old(dataset, pretrained, model, max_completion_length, device="cuda:0"):
    correct = 0
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    model.to(device)
    for index, element in enumerate(dataset): 

        t0 = time.time()
        model_inputs = tokenizer(element["prompt"], return_tensors="pt").to(device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_completion_length
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
        response = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        t1 = time.time()
        print(f"{((t1 - t0) / len(output_ids)):.4f} t/s")
        correct += compute_rewards(response, element["ground_truth"])
    print(f"{correct} out of {len(dataset)} correct")


def score(dataset, pretrained, model, max_completion_length, batch_size=8, device="cuda:0"):
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    model.to(device)
    model.eval()

    total_correct = 0
    total_tokens = 0
    total_time = 0.0

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:min(i + batch_size, len(dataset))]

        prompts = batch["prompt"] 
        ground_truths = batch["ground_truth"] 

        t0 = time.time()
        # Batch tokenization
        model_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        input_lengths = model_inputs.input_ids.shape[1]

        # Generate in batch
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_completion_length,
                do_sample=False,
            )

        t1 = time.time()

        # Remove input tokens to get only generated part
        gen = generated_ids[:, input_lengths:]

        # Decode outputs
        responses = tokenizer.batch_decode(gen, skip_special_tokens=True)

        # Compute rewards
        for response, gt in zip(responses, ground_truths):
            total_correct += compute_rewards(response.strip(), gt)

        # Track stats
        total_tokens += gen.numel()
        total_time += (t1 - t0)
        print(f"Token generation speed: {gen.numel() / (t1 - t0):.4f} tokens/sec")

    print(f"{total_correct} out of {len(dataset)} correct")
    print(f"Total time: {total_time}")
    print(f"Average speed: {total_tokens / total_time:.4f} tokens/sec")



if __name__ == "__main__":
    print("Parsing arguments...")
    args = parse_args()
    
    print("Loading Dataset...")
    dataset = load_dataset(args.dataset_path)["train"]
    #dataset = load_dataset(args.dataset_path)["train"].select(range(900,1000))
    reformatted = reformat_dataset(dataset)
        
    pipe = load_model(args.model, args.models, args.strength)
    print("Computing accuracy")
    score(reformatted, args.model, pipe, args.max_completion_length, args.batch_size)
