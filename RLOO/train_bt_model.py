import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import get_cosine_schedule_with_warmup
import yaml
import wandb
from tqdm import tqdm
import json

from bt_model import BTModel
from rloo_dataset import RLOODataset


def train(model, dataloader, optimizer, device, scheduler, num_epochs):
    model.train()
    iter = 0

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

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                iter += 1
                pbar.update(1)
                
                if iter % 10 == 0:
                    tqdm.write(f"epoch: {epoch}, iter: {iter}, loss: {loss.item()}, lr: {scheduler.get_last_lr()[0]}")
                    wandb.log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})
                    
                pbar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])


if __name__ == "__main__":

    with open("config/bt.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    model_name = config['model_name']
    data_path = config['data_path']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    max_length = config['max_length']

    wandb.init(project="bt", config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model = BTModel(base_model).to(device)

    data = json.load(open(data_path, "r"))
    dataset = RLOODataset(data=data, tokenizer=tokenizer, task="ultrafeedback", max_length=max_length)
    dataloader = DataLoader(dataset, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(dataloader) * num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)  

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    train(model, dataloader, optimizer, device, scheduler, num_epochs)
    torch.save(model.state_dict(), "bt_model.pt")
