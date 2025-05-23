import json
import os
import openai
from datasets import load_dataset
from countdown import compute_score
from tqdm import tqdm
from generate_countdown_gt import get_answer

if __name__ == "__main__":
    output_dir = "../processed_dataset"
    os.makedirs(output_dir, exist_ok=True)
    ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    

    total_num = 1500
    curr_num = 0
    openai.api_key = "API_KEY here"
    with open(os.path.join(output_dir, "countdown_train_preference.json"), "w") as f:
        f.write('[')

        for item in tqdm(ds):

            if total_num == 0:
                break

            numbers = item["nums"]
            target = item["target"]

            answer = get_answer(item)

            current_score = 0
            curr_try = 0
            current_highest_score = 0
            current_highest_score_response = ""

            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Using the numbers {numbers}, create an equation that equals {target}, here is the answer: {answer}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your thinking process in <think> </think> tags. And return the final answer equation in <answer> </answer> tags with the format <answer> {answer} </answer>"}
                ],
                temperature=0.3,
                max_tokens=1024
            )

            response = response['choices'][0]['message']['content']


            current_score = compute_score(f"Assistant: {response}", {"target": target, "numbers": numbers})

            if current_score < 0.9:
                continue

            total_num -= 1

            f.write(json.dumps({
                "numbers": numbers,
                "target": target,
                "response": response,
                "score": current_score
            }) + ',\n')

        f.write(']')

