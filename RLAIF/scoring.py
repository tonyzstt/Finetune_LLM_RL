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
    processed_data = []
    for item in data:
        input = item["input"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        try:
            chosen_reward = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": input},
                    {"role": "assistant", "content": chosen}
                ],
            )
            chosen_reward = chosen_reward.choices[0].message.content.split(':')[-1].strip()
            chosen_reward = float(chosen_reward)

            rejected_reward = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": input},
                    {"role": "assistant", "content": rejected}
                ],
            )
            rejected_reward = rejected_reward.choices[0].message.content.split(':')[-1].strip()
            rejected_reward = float(rejected_reward)
            
            item["chosen_reward"] = chosen_reward
            item["rejected_reward"] = rejected_reward
            
            processed_data.append(item)
            
        except Exception as e:
            print(f"Error processing item: {item}")
            print(f"Error message: {e}")
            continue
    
    print(f"Processed {len(processed_data)} items.")
    return processed_data
        
        
if __name__ == "__main__":
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=API_KEY
    )
    
    data_path = "../processed_dataset/ultrafeedback_binarized_train_prefs_small.json"
    data = json.load(open(data_path, "r"))
    
    data = reward_labeling(data, client)
    with open(f"{data_path.replace(".json", "_reward.json")}", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)