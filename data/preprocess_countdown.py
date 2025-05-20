from datasets import load_dataset
import json
import os
import sys
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.join(os.path.dirname(__file__)))
import countdown


def preprocess_countdown_dataset(dataset, output_file):
    """
    Preprocess the countdown dataset by adding preferred and dispreferred responses with corresponding scores.
    """
    
    # TODO: change the model to SFT model
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").cuda()
    
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
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        output_ids1 = model.generate(input_ids, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9, top_k=50, pad_token_id=tokenizer.eos_token_id)
        output_ids2 = model.generate(input_ids, max_new_tokens=512, do_sample=True, temperature=0.9, top_p=1.0, top_k=30, pad_token_id=tokenizer.eos_token_id)
        output1 = tokenizer.decode(output_ids1[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        output2 = tokenizer.decode(output_ids2[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        # print(f"Prompt: {prompt}\n\n")
        # print(f"Output 1: {output1}\n\n")
        # print(f"Output 2: {output2}\n\n")
        score1 = countdown.compute_score(output1, ground_truth)
        score2 = countdown.compute_score(output2, ground_truth)
        # print(f"Score 1: {score1}")
        # print(f"Score 2: {score2}")
        if score1 > score2:
            chosen = output1
            rejected = output2
        elif score1 < score2:
            chosen = output2
            rejected = output1
        # filtering out for ties
        else:
            continue

        message = {
            "input": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "ground_truth": ground_truth,
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