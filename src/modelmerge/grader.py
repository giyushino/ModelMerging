#conda_env: modelmerge
import re


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


def compute_rewards(prompts, completions, completion_ids, **kwargs):
    """
    Extract \\boxed content from completion and compare to ground truth, return list of rewards
    """
    rewards = []
    for completion, completion_id, answer in zip(completions, completion_ids, kwargs["ground_truth"]): 
        score = 0
        solution = extract_solution(completion)
        # if the solution exists, check if it is correct and reward 
        # otherwise if formatted correctly but not correct, pass
        if solution:
            if str(answer) == solution: 
                score += 1
        else: 
            # this is the EOS token, if not found, determine that generation was not finished in time 
            if 151645 not in completion_id:
                score -= 1 
        rewards.append(score)
    print(f"rewards: {rewards}")
    return rewards


if __name__ == "__main__":
    print(compute_rewards("", ["what the heck \\boxed{20} \\boxed{10}", "\\boxed{something}", "brah"], "", ground_truth=[10, 20, 30]))
    # should output [1, 0, -1]
