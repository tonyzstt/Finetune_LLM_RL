from torch.utils.data import Dataset

class DPODataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        prompt = example['input']
        chosen = example['chosen']
        rejected = example['rejected']

        prompt_chosen = prompt + chosen
        prompt_rejected = prompt + rejected

        enc_chosen = self.tokenizer(prompt_chosen, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        enc_rejected = self.tokenizer(prompt_rejected, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        enc_prompt = self.tokenizer(prompt, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        return {
            'input_ids_prompt': enc_prompt['input_ids'].squeeze(0),
            'attention_mask_prompt': enc_prompt['attention_mask'].squeeze(0),
            'input_ids_chosen': enc_chosen['input_ids'].squeeze(0),
            'attention_mask_chosen': enc_chosen['attention_mask'].squeeze(0),
            'input_ids_rejected': enc_rejected['input_ids'].squeeze(0),
            'attention_mask_rejected': enc_rejected['attention_mask'].squeeze(0),
        }