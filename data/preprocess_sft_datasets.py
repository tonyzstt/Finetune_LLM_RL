import json
import os
from datasets import load_dataset
from tqdm import tqdm


def check_conversation_length(message, max_input_length=128, max_output_length=512):
    """
    Check if the conversation word count is within the specified limits.
    """
    input_length = sum(len(m["content"].split(" ")) for m in message if m["role"] in ["system", "user"])
    output_length = sum(len(m["content"].split(" ")) for m in message if m["role"] == "assistant")
    return input_length <= max_input_length and output_length <= max_output_length


def preprocess_warm_start_dataset(dataset, output_file, max_length, overwrite, use_system_message=False, ):
    """
    Preprocess the warm start dataset to add a system message to each entry.
    """
    
    if os.path.exists(output_file) and not overwrite:
        print(f"Output file {output_file} already exists. Use overwrite=True to overwrite.")
        return
    else:
        if os.path.exists(output_file):
            os.remove(output_file)
        print(f"Output file {output_file} removed to overwrite.")
    
    with open(output_file, "w") as f:
        f.write('[')
    
    system_message = {
        "role": "system",
        "content": "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
                   "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer."
    }
    
    logger = {
        'total_entries': len(dataset),
        'filtered_entries': 0,
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
        
        if use_system_message:
            message = {
                "messages": [
                    system_message,
                    user_message,
                    assistant_message
                ],
            }
        else:
            message = {
                "messages": [
                    user_message,
                    assistant_message
                ],
            }
        
        if not check_conversation_length(message["messages"], max_length[0], max_length[1]):
            logger['filtered_entries'] += 1
            continue
        
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
    print(f"Filtered {logger['filtered_entries']} entries from {logger['total_entries']}, remaining {logger['total_entries'] - logger['filtered_entries']} entries")
    
    
def preprocess_smol_talk_dataset(dataset, output_file, max_length, overwrite, use_system_message=False, ):
    """
    Preprocess the smol talk dataset to add a system message to each entry.
    """
    
    if os.path.exists(output_file) and not overwrite:
        print(f"Output file {output_file} already exists. Use overwrite=True to overwrite.")
        return
    else:
        # Remove the output file if it exists
        if os.path.exists(output_file):
            os.remove(output_file)
        print(f"Output file {output_file} removed to overwrite.")
    
    with open(output_file, "a") as f:
        f.write('[')
    
    system_message = {
        "role": "system",
        "content": ""   # TODO: Add a system message for the smol talk dataset
    }
    
    logger = {
        'total_entries': len(dataset),
        'filtered_entries': 0,
    }
    
    for item in tqdm(dataset):
        first_user_message = None
        for message in item["messages"]:
            if message["role"] == "user":
                first_user_message = message
                break
        first_assistant_message = None
        for message in item["messages"]:
            if message["role"] == "assistant":
                first_assistant_message = message
                break
            
        if first_user_message is None or first_assistant_message is None:
            print(f"Missing user or assistant message in item: {item}")
            logger['filtered_entries'] += 1
            continue
        
        if use_system_message:
            message = {
                "messages": [system_message, first_user_message, first_assistant_message]
            }
        else:
            message = {
                "messages": [first_user_message, first_assistant_message]
            }
            
        # Check if message contains one user and one assistant message
        if not (len(message["messages"]) == 2 and \
                (message["messages"][-1]["role"] == "user" and message["messages"][-2]["role"] == "assistant" or 
                 message["messages"][-1]["role"] == "assistant" and message["messages"][-2]["role"] == "user")):
            print(f"Invalid message format: {message}")
            logger['filtered_entries'] += 1
            continue
        
        if not check_conversation_length(message["messages"], max_length[0], max_length[1]):
            logger['filtered_entries'] += 1
            continue
        
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
    print(f"Filtered {logger['filtered_entries']} entries from {logger['total_entries']}, remaining {logger['total_entries'] - logger['filtered_entries']} entries")
    
    
def preprocess_sft_datasets(use_system_message=False, max_length=(128, 512), overwrite=False, split="train"):
    output_dir = "../processed_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    warm_start_dataset = load_dataset("Asap7772/cog_behav_all_strategies", split=split)
    warm_start_output_file = os.path.join(output_dir, f"processed_warm_start_dataset_{max_length[0]}_{max_length[1]}_{split}.json")
    preprocess_warm_start_dataset(warm_start_dataset, warm_start_output_file, max_length, overwrite, use_system_message)
    
    smol_talk_dataset = load_dataset("HuggingFaceTB/smol-smoltalk", split=split)
    smol_talk_output_file = os.path.join(output_dir, f"processed_smol_talk_dataset_{max_length[0]}_{max_length[1]}_{split}.json")
    preprocess_smol_talk_dataset(smol_talk_dataset, smol_talk_output_file, max_length, overwrite, use_system_message)
    
    
if __name__ == "__main__":
    preprocess_sft_datasets(split='train', overwrite=False)
    preprocess_sft_datasets(split='test', overwrite=False)
