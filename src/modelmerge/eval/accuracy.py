#conda_env: mergemodel
import re
import asyncio
import argparse

from openai import AsyncOpenAI
from datasets import load_dataset

from modelmerge.data import reformat_dataset
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


def compute_rewards_old(completion, ground_truth):
    """
    Extract \\boxed content from completion and compare to ground truth, return list of rewards
    """
    solution = extract_solution(completion)
    if solution: 
        if str(ground_truth) == solution:
            print("correct")
            return 1
    print("incorrect")
    return 0


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

def parse_args():
    parser = argparse.ArgumentParser(description="Check completions for sparse errors")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to local dataset, or huggingface id")
    parser.add_argument("--port", type=int, required=True, help="port id")
    parser.add_argument("--model", type=str, required=True, help="model id")
    parser.add_argument("--max_completion_length", type=int, required=True, help="model length")
    args = parser.parse_args()
    return args


async def process_all_messages(ds, port, model, completion_length):
    client = AsyncOpenAI(base_url=f"http://localhost:{port}/v1",
                         api_key="dummy")

    semaphore = asyncio.Semaphore(100)

    async def process_one(prompt, answer):
        async with semaphore:
            try:
                r = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=completion_length
                )
            except Exception:
                r = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000
                )
            correct = compute_rewards(r.choices[0].message.content, ground_truth=str(answer))
            return correct 

    # ds must be an iterable of dicts with keys "prompt" and "ground_truth"
    tasks = [process_one(item["prompt"], item["ground_truth"]) for item in ds]
    results = await asyncio.gather(*tasks)
    await client.close()

    total_correct = sum(results)
    print(f"{total_correct} of {len(results)} correct")



if __name__ == "__main__":
    print("Parsing arguments...")
    args = parse_args()
    
    print("Loading Dataset...")
    dataset = load_dataset(args.dataset_path)["train"]
    #dataset = load_dataset(args.dataset_path)["train"].select(range(900,1000))
    reformatted = reformat_dataset(dataset)
        
    print("Computing accuracy")
    asyncio.run(process_all_messages(reformatted, args.port, args.model, args.max_completion_length))


