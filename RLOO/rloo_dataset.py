from torch.utils.data import Dataset
from dataclasses import dataclass

class RLOODataset(Dataset):
    def __init__(self, data, tokenizer, task, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        # task: str, either "ultrafeedback" or "countdown"
        self.task = task
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        prompt = example["input"]
        enc_prompt = self.tokenizer(prompt, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

        if self.task == "ultrafeedback":
            return {
                "input_ids_prompt": enc_prompt["input_ids"].squeeze(0),
                "attention_mask_prompt": enc_prompt["attention_mask"].squeeze(0),
            }

        elif self.task == "countdown":
            return {
                "ground_truth": example["ground_truth"],
                "input_ids_prompt": enc_prompt["input_ids"].squeeze(0),
                "attention_mask_prompt": enc_prompt["attention_mask"].squeeze(0),
            }