import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import get_cosine_schedule_with_warmup
import yaml
import wandb
from tqdm import tqdm
import json

from reward_model import RewardModel
from rloo_dataset import RLOODataset


def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2) 
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def train(model, dataloader, optimizer, device, scheduler, cfg):
    
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

                chosen_ids    = batch['input_ids_chosen'].to(device)
                rejected_ids  = batch['input_ids_rejected'].to(device)
                chosen_mask   = batch['attention_mask_chosen'].to(device)
                rejected_mask = batch['attention_mask_rejected'].to(device)

                chosen_reward   = model(chosen_ids,   attention_mask=chosen_mask)
                rejected_reward = model(rejected_ids, attention_mask=rejected_mask)

                loss = model.loss(chosen_reward, rejected_reward)

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
                    torch.save(model.state_dict(), f"reward_model_{iter}.pt")


if __name__ == "__main__":

    with open("config/reward.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    model_name = config['model_name']
    data_path = config['data_path']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    max_length = config['max_length']
    task = config['task']

    wandb.init(project="reward", config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model = RewardModel(base_model).to(device)

    data = json.load(open(data_path, "r"))
    dataset = RLOODataset(data=data, tokenizer=tokenizer, task=task, max_length=max_length)
    dataloader = DataLoader(dataset, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(dataloader) * num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)  

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    train(model, dataloader, optimizer, device, scheduler, config)
    torch.save(model.state_dict(), "reward_model.pt")
