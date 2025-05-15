from typing import List
import json

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class SFTDataset(Dataset):
    def __init__(
        self, 
        data_path: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int
    ):
        self.data = []
        for path in data_path:
            with open(path, 'r') as f:
                data = json.load(f)
                self.data.extend(data)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        messages = self.data[idx]['messages']
        
        conversation = ""
        assistant_spans = []
        
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'system':
                if content.strip():
                    conversation += f"<|system|>{content}\n"
            elif role == 'user':
                conversation += f"<|user|>{content}\n"
            elif role == 'assistant':
                start = len(conversation)
                conversation += f"<|assistant|>{content}\n"
                end = len(conversation)
                assistant_spans.append((start, end))
        
        encoding = self.tokenizer(
            conversation,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        labels = input_ids.clone().fill_(self.tokenizer.pad_token_id)
        
        for start, end in assistant_spans:
            token_start = self.tokenizer(conversation[:start], add_special_tokens=False, return_tensors="pt")["input_ids"].size(1)
            token_end = self.tokenizer(conversation[:end], add_special_tokens=False, return_tensors="pt")["input_ids"].size(1)
            # Clamp to max_length
            token_start = min(token_start, self.max_length)
            token_end = min(token_end, self.max_length)
            labels[token_start:token_end] = input_ids[token_start:token_end]

        # Check if input_ids and labels are correct by converting them back to text
        if False:
            decoded_input = self.tokenizer.decode(input_ids, skip_special_tokens=True)
            decoded_labels = self.tokenizer.decode(labels, skip_special_tokens=True)
            print(f"Decoded input: {decoded_input}")
            print(f"Decoded labels: {decoded_labels}")
            
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        
    def collate_fn(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
