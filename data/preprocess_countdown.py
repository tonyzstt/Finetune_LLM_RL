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
    Preprocess the countdown dataset by adding preferred and dispreferred responses with corresponding scores.
    """
    
    # TODO: change the model to SFT model
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
        system = "<|im_start|>system\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n"
        user = f"<|im_start|>user\nUsing the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer> <|im_end|>\n"
        assistant = "<|im_start|>assistant\n"
        prompt = user + assistant
        output1 = model(prompt, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9, top_k=50)[0]["generated_text"][len(prompt):].strip()
        output2 = model(prompt, max_new_tokens=512, do_sample=True, temperature=0.9, top_p=1.0, top_k=30)[0]["generated_text"][len(prompt):].strip()
        # print(f"Prompt: {prompt}\n\n")
        # print(f"Output 1: {output1}\n\n")
        # print(f"Output 2: {output2}\n\n")
        solution1 = countdown.extract_solution(output1)
        solution2 = countdown.extract_solution(output2)
        # print(f"Solution 1: {solution1}")
        # print(f"Solution 2: {solution2}")
        score1 = countdown.compute_score(output1, ground_truth)
        score2 = countdown.compute_score(output2, ground_truth)
        if score1 > score2:
            chosen = solution1
            chosen_score = score1
            rejected = solution2
            rejected_score = score2
        elif score1 < score2:
            chosen = solution2
            chosen_score = score2
            rejected = solution1
            rejected_score = score1
        # filtering out for ties
        else:
            continue

        message = {
            "input": prompt,
            "chosen": chosen,
            "chosen_score": chosen_score,
            "rejected": rejected,
            "rejected_score": rejected_score,
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