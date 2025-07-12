#conda_env: modelmerge
import re


def extract_solution(output):
    """
    Use regex to extract boxed solution from model output
    """ 
    match = re.search(r'{(.*?)}', output.split("\\boxed")[-1])
    if match:
        return match.group()
    return None

def compute_rewards(prompts, completions, **kwargs):
    """
    Extract \\boxed content from completion and compare to ground truth, return list of rewards
    """
    rewards = []
    for completion, answer in zip(completions, kwargs["ground_truth"]): 
        print(completion)
        print(f"The last character in completions is {completion[-1]}")
        solution = extract_solution(completion)
        #if int(solution) == answer: 
        if str(answer) == solution: 
            rewards.append(1)
        else:
            rewards.append(0)

    return rewards

