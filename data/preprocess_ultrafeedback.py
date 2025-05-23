from datasets import load_dataset
import json
import os
from tqdm import tqdm


def preprocess_ultrafeedback_dataset(dataset, output_file):
    """
    Preprocess the warm start dataset to add a system message to each entry.
    """
    
    # Open the JSON array
    with open(output_file, "w") as f:
        f.write('[')

    num_samples = 0

    for item in tqdm(dataset):

        prompt = item["prompt"]
        chosen = item["chosen"][1]['content']
        rejected = item["rejected"][1]['content']
        chosen_score = item["score_chosen"]
        rejected_score = item["score_rejected"]

        prompt_length = len(prompt.split(" "))
        chosen_length = len(chosen.split(" "))
        rejected_length = len(rejected.split(" "))

        if prompt_length > 128 or chosen_length > 512 or rejected_length > 512 or chosen_score < 5 or chosen_score <= rejected_score:
            continue

        num_samples += 1

        message = {
            "input": prompt,
            "chosen": chosen,
            "rejected": rejected
        }
        
        message = json.dumps(message, indent=4, ensure_ascii=False)
        with open(output_file, "a", encoding='utf-8') as f:
            f.write(message + ",\n")
            
    # Remove the last comma and close the JSON array
    with open(output_file, "rb+") as f:
        f.seek(-2, os.SEEK_END)
        f.truncate()
    
    with open(output_file, "a") as f:
        f.write(']')
    
    print(f"Preprocessed dataset saved to {output_file}")
    print(f"Number of samples: {num_samples}")

if __name__ == "__main__":

    output_dir = "../processed_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    output_file = os.path.join(output_dir, "ultrafeedback_binarized_train_prefs.json")
    preprocess_ultrafeedback_dataset(ds, output_file)
    
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")
    output_file = os.path.join(output_dir, "ultrafeedback_binarized_test_prefs.json")
    preprocess_ultrafeedback_dataset(ds, output_file)