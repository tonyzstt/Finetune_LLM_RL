from torch.utils.data import Dataset
from dataclasses import dataclass
import json

class CountdownDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "r") as f:
            self.data = [json.loads(line) for line in f][-2000:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        example = self.data[idx]
        numbers = example["nums"]
        target = example["target"]

        user = f"<|im_start|>user\nUsing the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer> <|im_end|>\n"
        assistant = "<|im_start|>assistant\n"
        prompt = user + assistant

        return {
            "prompt": prompt,
            "target": target,
            "numbers": numbers
        }

if __name__ == "__main__":

    dataset = CountdownDataset("../../processed_dataset/countdown_gt.json")
    print(len(dataset))
    print(dataset[0])


