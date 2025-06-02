import json
import argparse
from tqdm import tqdm
from eval_countdown import compute_score
from eval_ultrafeedback import generate_response, get_model

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()


    llm = get_model(args.model_path)
    scores = []

    data = []
    with open("./dataset/data/countdown.json", "r") as f:
        for line in f:
            if line.strip():  # skip empty lines
                data.append(json.loads(line))

    print(f"Loaded {len(data)} entries.")
    save_path = "submission_countdown.json"

    for item in tqdm(data):

        target = item["target"]
        numbers = item["num"]

        prompt = f"<|im_start|>user\nUsing the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer> <|im_end|>\n"
        prompt += f"<|im_start|>assistant\n"
        response = generate_response(llm, prompt, temperature=0, top_p=1, max_tokens=2048, repetition_penalty=1)

        score = compute_score(
            solution_str=f"Assistant: {response}",
            ground_truth={"target": target, "numbers": numbers},
            method="strict",
            format_score=0.1,
            score=1.0
        )
        scores.append(score)
        # print("prompt:--------------------------------")
        # print(prompt)
        # print("response:--------------------------------")
        # print(response)
        # print("score:--------------------------------")
        # print(score)
        # exit()
        dict_to_save = {
            "num": numbers,
            "expression": f"Assistant: {response}",
            "target": target
        }
        with open(save_path, "a") as f:
            json.dump(dict_to_save, f)
            f.write("\n")

    print(f"Average score: {sum(scores) / len(scores)}")

