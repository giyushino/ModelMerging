#conda_env: modelmerge
import re

test_string = "Present the answer in LaTex format: \\boxed{Your answer}"  

def extract_solution(output):
    """
    Use regex to extract boxed solution from model output
    """
    match = re.search(r'\\boxed{(.*?)}', output)
    if match:
        return match.group(1)
    return None

def compute_rewards(prompts, completions, **kwargs):
    """
    Extract \\boxed content from completion and compare to ground truth, return list of rewards
    """
    rewards = []
    for completion, answer in zip(completions, kwargs["answer"]): # might be ground_truth and not anwer? 
        solution = extract_solution(completion)
        if int(solution) == answer: 
            rewards.append(1)
        else:
            rewards.append(0)

    return rewards


