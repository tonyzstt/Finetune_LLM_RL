import json
import argparse
from tqdm import tqdm
import numpy as np
from eval_ultrafeedback import generate_response, get_model, get_client, get_reward

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    save_path = "submission_ultrafeedback.json"
    client = get_client()
    llm = get_model(args.model_path)
    rewards = []

    data = []
    with open("./dataset/data/ultrafeedback.json", "r") as f:
        for line in f:
            if line.strip():  # skip empty lines
                data.append(json.loads(line))

    print(f"Loaded {len(data)} entries.")

    for item in tqdm(data):

        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{item['prompt']}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n"
        response = generate_response(llm, prompt, temperature=0, top_p=1, max_tokens=512, repetition_penalty=1.1)
        # -22.712421875 
        # response = generate_response(llm, prompt, temperature=0, top_p=1, max_tokens=1280, repetition_penalty=1.1)
        print(response)
        dict_to_save = {
            "prompt": item['prompt'],
            "response": response
        }
        with open(save_path, "a") as f:
            json.dump(dict_to_save, f)
            f.write("\n")
        reward = get_reward(client, prompt, response)
        reward = float(reward.split(":")[-1])
        print(reward)
        rewards.append(reward)

    print(f"Average reward: {sum(rewards) / len(rewards)}")
    np.save("rewards_base.npy", rewards)


