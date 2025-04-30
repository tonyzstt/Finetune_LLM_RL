from typing import Literal

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer


class SmolTalkDataset(Dataset):
    def __init__(self, split):
        self.dataset = load_dataset("HuggingFaceTB/smol-smoltalk", split=split)
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]['messages']
        # item['input_ids'] = torch.tensor(item['input_ids'])
        # item['attention_mask'] = torch.tensor(item['attention_mask'])
        return item
    
    def collate_fn(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]

        # Pad sequences to the same length
        input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(ids) for ids in input_ids], batch_first=True)
        attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(mask) for mask in attention_mask], batch_first=True)
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(lbls) for lbls in labels], batch_first=True)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        

class WarmStartDataset(Dataset):
    def __init__(self, split: Literal['train', 'test'], 
                 tokenizer: PreTrainedTokenizer,
                 dataset_path: str = "Asap7772/cog_behav_all_strategies"):
        """
        Args:
            split (str): The split of the dataset to load ('train' or 'test').
            tokenizer (PreTrainedTokenizer): The tokenizer to use for encoding.
        """
        assert split in ['train', 'test'], "split must be either 'train' or 'test'"
        self.tokenizer = tokenizer
        self.dataset = load_dataset(dataset_path, split=split)
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        source = self.dataset[idx]
        print(source)
    
    def collate_fn(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]

        # Pad sequences to the same length
        input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(ids) for ids in input_ids], batch_first=True)
        attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(mask) for mask in attention_mask], batch_first=True)
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(lbls) for lbls in labels], batch_first=True)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        
        
class SFTTrainer(Trainer):
    pass


def get_default_training_args():
    training_args = TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
    )
    return training_args


def get_qwen_model():
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


if __name__ == "__main__":
    train_dataset = WarmStartDataset(split='train')
    eval_dataset = WarmStartDataset(split='test')
    training_args = get_default_training_args()
    model, tokenizer = get_qwen_model()
    
    ############## SFT Trainer ##############
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=train_dataset.collate_fn,
    )
    
    trainer.train()
    trainer.evaluate()
    # trainer.save_model()
    
    ############## RLHF Trainer ##############
    pass