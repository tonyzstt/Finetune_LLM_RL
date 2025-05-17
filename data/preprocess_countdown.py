from datasets import load_dataset
import json
import os
import sys
from tqdm import tqdm
from transformers import pipeline

sys.path.append(os.path.join(os.path.dirname(__file__)))
import countdown


def preprocess_countdown_dataset(dataset, output_file):
    """
    Preprocess the warm start dataset to add a system message to each entry.
    """
    
    # TODO: change model if needed
    model = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B", device=0)
    
    # Open the JSON array
    with open(output_file, "w") as f:
        f.write("[")

    num = 100
    for item in tqdm(dataset):

        numbers = item["nums"]
        target = item["target"]
        ground_truth = {
            "target": target,
            "numbers": numbers
        }
        prompt = f"""<|im_start|>system
        A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant directly provides the user with the answer without any intermediate steps.
        <|im_end|>
        <|im_start|>user
        Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>
        <|im_end|>
        <|im_start|>assistant
        """
        output1 = model(prompt, max_new_tokens=50, do_sample=True, temperature=0.7, top_p=0.9, top_k=50)[0]["generated_text"][len(prompt):].strip()
        output2 = model(prompt, max_new_tokens=50, do_sample=True, temperature=0.9, top_p=1.0, top_k=30)[0]["generated_text"][len(prompt):].strip()
        # print(f"Prompt: {prompt}")
        # print(f"Output 1: {output1}")
        # print(f"Output 2: {output2}")
        solution1 = countdown.extract_solution(output1)
        solution2 = countdown.extract_solution(output2)
        # print(f"Solution 1: {solution1}")
        # print(f"Solution 2: {solution2}")
        score1 = countdown.compute_score(output1, ground_truth)
        score2 = countdown.compute_score(output2, ground_truth)
        if score1 > score2:
            chosen = solution1
            rejected = solution2
        else:
            chosen = solution2
            rejected = solution1

        message = {
            "input": prompt,
            "chosen": chosen,
            "rejected": rejected
        }
        
        message = json.dumps(message, indent=4, ensure_ascii=False)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(message + ",\n")

        num -= 1
        if num == 0:
            break
            
    # Remove the last comma and close the JSON array
    with open(output_file, "rb+") as f:
        f.seek(-2, os.SEEK_END)
        f.truncate()
    
    with open(output_file, "a") as f:
        f.write("]")
    
    print(f"Preprocessed dataset saved to {output_file}")


if __name__ == "__main__":

    output_dir = "../processed_dataset"
    os.makedirs(output_dir, exist_ok=True)
    ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    output_file = os.path.join(output_dir, "countdown_train_small.json")
    preprocess_countdown_dataset(ds, output_file)