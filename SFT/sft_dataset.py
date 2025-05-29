from typing import List
import json

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

IGNORE_INDEX = -100

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
        system_prompt = "You are a helpful assistant."
        assistant_spans = []
        
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'system':
                if content.strip():
                    conversation += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == 'user':
                conversation += f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n"
            elif role == 'assistant':
                conversation += "<|im_start|>assistant\n"
                start = len(conversation)
                conversation += f"{content}<|im_end|>\n"
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
        labels = input_ids.clone().fill_(IGNORE_INDEX)
        
        for start, end in assistant_spans:
            token_start = self.tokenizer(conversation[:start], add_special_tokens=False, return_tensors="pt")["input_ids"].size(1)
            token_end = self.tokenizer(conversation[:end], add_special_tokens=False, return_tensors="pt")["input_ids"].size(1)
            # Clamp to max_length
            token_start = min(token_start, self.max_length)
            token_end = min(token_end, self.max_length)
            labels[:token_start] = IGNORE_INDEX

            # The +1 is for the eos token
            if token_end == self.max_length:
                labels[token_start:token_end] = input_ids[token_start:token_end]
                attention_mask[-1] = 1
            else:
                labels[token_start:token_end+1] = input_ids[token_start:token_end+1]
                attention_mask[token_end] = 1

        # Check if input_ids and labels are correct by converting them back to text
        if False:
            print(labels[token_end-1], input_ids[token_end-1], attention_mask[token_end-1])
            print(labels[token_end], input_ids[token_end], attention_mask[token_end])
            print(labels[token_end+1], input_ids[token_end+1], attention_mask[token_end+1])
            print(labels[token_end+2], input_ids[token_end+2], attention_mask[token_end+2])
            print(input_ids)
            print(labels)
            labels[labels == IGNORE_INDEX] = self.tokenizer.pad_token_id
            decoded_input = self.tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            decoded_labels = self.tokenizer.decode(labels, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            print("-----------------Start-----------------")
            print(f"Decoded input: {decoded_input}")
            print("--------------------------------")
            print(f"Decoded labels: {decoded_labels}")
            print("-----------------End-----------------")
            print(attention_mask)
            exit()
            
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
