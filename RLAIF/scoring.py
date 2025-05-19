import json
from openai import OpenAI
from config import API_KEY


def reward_labeling(
    data,
    client,
    model="nvidia/llama-3.1-nemotron-70b-reward"
):
    """
    Get the reward score for a given response to a prompt using the reward model.
    """
    for item in data:
        input = item["input"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        chosen_score = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": input},
                {"role": "assistant", "content": chosen}
            ],
        )
        
        rejected_score = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": input},
                {"role": "assistant", "content": rejected}
            ],
        )
        
        print(f"Chosen: {chosen_score}")
        print(f"Rejected: {rejected_score}")
        exit(0)
        
        item["chosen_score"] = chosen_score
        item["rejected_score"] = rejected_score
        
    return data
        
        
if __name__ == "__main__":
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=API_KEY
    )
    
    data_path = "../processed_dataset/ultrafeedback_binarized_train_prefs_small.json"
    data = json.load(open(data_path, "r"))
    
    data = reward_labeling(data, client)
    with open(f"{data_path.split('.')[0]}_rewarded.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)