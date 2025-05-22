import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from tqdm import tqdm
import wandb

from sft_dataset import SFTDataset


def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2) 
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def train(model, dataloader, optimizer, device, scheduler, cfg, save_dir):
    
    num_epochs = cfg['num_epochs']
    grad_accum_steps = cfg['gradient_accumulation_steps']
    log_steps = cfg['log_steps']
    save_steps = cfg['save_steps']
    
    model.train()
    iter = 0
    running_loss = 0.0
    optimizer.zero_grad()
    total_steps = num_epochs * len(dataloader)
    with tqdm(total=total_steps) as pbar:
        for epoch in range(num_epochs):
            for batch in dataloader:
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                loss = loss / grad_accum_steps
                loss.backward()

                iter += 1
                pbar.update(1)
                
                if iter % grad_accum_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                running_loss += loss.item()
                
                if iter % log_steps == 0:
                    avg_loss = running_loss / log_steps
                    grad_norm = get_grad_norm(model)
                    tqdm.write(f"epoch: {epoch}, iter: {iter}, avg_loss: {avg_loss}, lr: {scheduler.get_last_lr()[0]}, grad_norm: {grad_norm}")
                    wandb.log({"avg_loss": avg_loss, "lr": scheduler.get_last_lr()[0], "grad_norm": grad_norm})
                    
                    pbar.set_postfix(avg_loss=avg_loss, lr=scheduler.get_last_lr()[0])
                    running_loss = 0.0
                    
                if iter % save_steps == 0:
                    model.save_pretrained(f"/viscam/u/tonyzst/Research/test/SFT/models/{save_dir}_{iter}")
                    tokenizer.save_pretrained(f"/viscam/u/tonyzst/Research/test/SFT/models/{save_dir}_{iter}")
                

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    model_name = config['model_name']
    data_path_list = config['data_path_list']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    max_length = config['max_length']
    save_dir = config['save_dir']
    
    wandb.init(project="sft", config=config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    dataset = SFTDataset(data_path=data_path_list, tokenizer=tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(dataloader) * num_epochs
    num_warmup_steps = int(num_training_steps * 0.1)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    train(model, dataloader, optimizer, device, scheduler, config, save_dir)
    model.save_pretrained(f"/viscam/u/tonyzst/Research/test/SFT/models/{save_dir}")
    tokenizer.save_pretrained(f"/viscam/u/tonyzst/Research/test/SFT/models/{save_dir}")
    