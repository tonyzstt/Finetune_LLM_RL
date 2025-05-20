from torch.utils.data import Dataset
from dataclasses import dataclass
import json

class UltraFeedbackDataset(Dataset):
    def __init__(self, data_path):
        self.data = json.load(open(data_path, "r"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        example = self.data[idx]
        prompt = f"<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n"
        chosen = example['chosen']
        rejected = example['rejected']

        prompt_chosen = prompt + chosen + "<|im_end|>\n"
        prompt_rejected = prompt + rejected + "<|im_end|>\n"

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        }


if __name__ == "__main__":

    dataset = UltraFeedbackDataset("../../processed_dataset/ultrafeedback_binarized_test_prefs.json")
    print(len(dataset))
    print("prompt: ", dataset[0]["prompt"])
    print("chosen: ", dataset[0]["chosen"])
    print("rejected: ", dataset[0]["rejected"])
