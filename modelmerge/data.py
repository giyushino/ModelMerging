from datasets import Dataset

#sunyiyou/math_comp_polynomial_gcd
#sunyiyou/math_algebra_polynomial_roots_7B_train
#sunyiyou/math_arithmetic_gcd_7B_train

def reformat_dataset(dataset, split):
    """
    Reformat the oob dataset to work with GRPO Trainer class 
    """
    reformatted = []
    count = 0
    for element in dataset[split]: 
        try: 
            # extract question and remove their formatting instructions
            question = element["messages"][0]["content"].split("\n\nPresent the answer in LaTex format: \\boxed{Your answer}")[0]
            reformatted.append({ 
            "prompt": 
            "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
            + question
            + "<|im_end|>\n<|im_start|>assistant\n", 
            "ground_truth": element["ground_truth"]
            })
        except: count += 1

    print(f"{count} of {count + len(reformatted)} elements were skipped due to formatting issues")
    return Dataset.from_list(reformatted) 
