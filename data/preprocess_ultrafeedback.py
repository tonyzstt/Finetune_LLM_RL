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

    num = 100
    for item in tqdm(dataset):

        input = item["prompt"]
        chosen = item["chosen"][1]['content']
        rejected = item["rejected"][1]['content']
        message = {
            "input": input,
            "chosen": chosen,
            "rejected": rejected
        }
        
        message = json.dumps(message, indent=4, ensure_ascii=False)
        with open(output_file, "a", encoding='utf-8') as f:
            f.write(message + ",\n")

        num -= 1
        if num == 0:
            break
            
    # Remove the last comma and close the JSON array
    with open(output_file, "rb+") as f:
        f.seek(-2, os.SEEK_END)
        f.truncate()
    
    with open(output_file, "a") as f:
        f.write(']')
    
    print(f"Preprocessed dataset saved to {output_file}")


if __name__ == "__main__":

    output_dir = "../processed_dataset"
    os.makedirs(output_dir, exist_ok=True)
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    
    output_file = os.path.join(output_dir, "ultrafeedback_binarized_train_prefs_small.json")
    preprocess_ultrafeedback_dataset(ds, output_file)