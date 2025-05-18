import json
import os
from datasets import load_dataset
from tqdm import tqdm


def preprocess_warm_start_dataset(dataset, output_file):
    """
    Preprocess the warm start dataset to add a system message to each entry.
    """
    
    # Open the JSON array
    with open(output_file, "w") as f:
        f.write('[')
    
    system_message = {
        "role": "system",
        "content": "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
                   "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer."
    }
    
    for item in tqdm(dataset):
        user_message = {
            "role": "user",
            "content": item["query"].split("\nAssistant:")[0].split("User:")[1].strip()
        }
        assistant_message = {
            "role": "assistant",
            "content": item["completion"]
        }
        
        message = {
            "messages": [
                system_message,
                user_message,
                assistant_message
            ],
        }
        
        message = json.dumps(message, indent=4)
        with open(output_file, "a") as f:
            f.write(message + ",\n")
            
    # Remove the last comma and close the JSON array
    with open(output_file, "rb+") as f:
        f.seek(-2, os.SEEK_END)
        f.truncate()
    
    with open(output_file, "a") as f:
        f.write(']')
    
    print(f"Preprocessed dataset saved to {output_file}")
    
    
def preprocess_smol_talk_dataset(dataset, output_file):
    """
    Preprocess the smol talk dataset to add a system message to each entry.
    """
    
    with open(output_file, "a") as f:
        f.write('[')
    
    system_message = {
        "role": "system",
        "content": ""   # TODO: Add a system message for the smol talk dataset
    }
    
    for item in tqdm(dataset):        
        message = {
            "messages": [system_message] + item["messages"]
        }
        # indent the JSON for better readability
        message = json.dumps(message, indent=4)
        with open(output_file, "a") as f:
            f.write(message + ",\n")
            
    # Remove the last comma and close the JSON array
    with open(output_file, "rb+") as f:
        f.seek(-2, os.SEEK_END)
        f.truncate()
        
    with open(output_file, "a") as f:
        f.write(']\n')
        
    print(f"Preprocessed dataset saved to {output_file}")
    
    
def preprocess_sft_datasets():
    output_dir = "../processed_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    warm_start_dataset = load_dataset("Asap7772/cog_behav_all_strategies", split="train")
    warm_start_output_file = os.path.join(output_dir, "processed_warm_start_dataset.json")
    preprocess_warm_start_dataset(warm_start_dataset, warm_start_output_file)
    
    # smol_talk_dataset = load_dataset("HuggingFaceTB/smol-smoltalk", split='train')
    # smol_talk_output_file = os.path.join(output_dir, "processed_smol_talk_dataset.json")
    # preprocess_smol_talk_dataset(smol_talk_dataset, smol_talk_output_file)
    

def remove_too_long_conversations(input_file, max_length=256):
    """
    Remove conversations that are too long from the dataset.
    """
    
    with open(input_file, "r") as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} entries from {input_file}")
    print(f"Filtering conversations longer than {max_length} characters...")
    
    filtered_data = []
    for item in data:
        conversation = item["messages"]
        max_len = max(len(msg["content"].split(" ")) for msg in conversation)
        if max_len <= max_length:
            filtered_data.append(item)
    
    # output file name is the same as input file except with "_filter_over_{max_length}" added
    output_file = input_file.replace(".json", f"_filter_over_{max_length}.json")
    with open(output_file, "w") as f:
        json.dump(filtered_data, f, indent=4)
    
    print(f"Filtered {len(data) - len(filtered_data)}, remaining {len(filtered_data)} entries")
    print(f"Filtered dataset saved to {output_file}")
    
    
if __name__ == "__main__":
    pass
    # remove_too_long_conversations("../processed_dataset/processed_warm_start_dataset.json", max_length=256)
    # remove_too_long_conversations("../processed_dataset/processed_smol_talk_dataset.json", max_length=256)
