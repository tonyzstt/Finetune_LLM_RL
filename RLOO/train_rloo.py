import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import get_cosine_schedule_with_warmup
import yaml
import wandb
from tqdm import tqdm
import json

from bt_model import BTModel
from rloo_dataset import RLOODataset


def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2) 
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def get_log_prob(logits, labels, prompt_lengths):
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
    
    # mask out the prompt tokens
    _, seq_len = labels.shape
    response_mask = torch.arange(seq_len, device=labels.device).unsqueeze(0) >= prompt_lengths.unsqueeze(1)
    response_mask = response_mask.float()
    response_log_probs = (token_log_probs * response_mask).sum(dim=-1)
    response_lengths = response_mask.sum(dim=-1).clamp(min=1)

    return response_log_probs / response_lengths


def rloo_loss(chosen_lp, rejected_lp, chosen_reward, rejected_reward, k):
    # k = 2 since we only have two perference data per prompt
    assert k == 2
    assert chosen_lp.shape == rejected_lp.shape
    assert chosen_reward.shape == rejected_reward.shape
    RLOOloss = -(1/k * (chosen_reward - rejected_reward) * chosen_lp + 1/k * (rejected_reward - chosen_reward) * rejected_lp)
    
    return RLOOloss


def train(model, dataloader, optimizer, device, scheduler, num_epochs, task, k=2, reward_model=None):
    model.train()
    iter = 0

    total_steps = num_epochs * len(dataloader)
    with tqdm(total=total_steps) as pbar:
        for epoch in range(num_epochs):
            for batch in dataloader:

                prompt_len = batch['attention_mask_prompt'].sum(dim=1).to(device)

                chosen_ids    = batch['input_ids_chosen'].to(device)
                rejected_ids  = batch['input_ids_rejected'].to(device)
                chosen_mask   = batch['attention_mask_chosen'].to(device)
                rejected_mask = batch['attention_mask_rejected'].to(device)

                chosen_logits   = model(chosen_ids,   attention_mask=chosen_mask).logits
                rejected_logits = model(rejected_ids, attention_mask=rejected_mask).logits

                chosen_lp   = get_log_prob(chosen_logits,   chosen_ids,   prompt_len)
                rejected_lp = get_log_prob(rejected_logits, rejected_ids, prompt_len)

                if task == "ultrafeedback":

                    # get the reward from the reward model
                    chosen_reward   = reward_model(chosen_ids,   attention_mask=chosen_mask)
                    rejected_reward = reward_model(rejected_ids, attention_mask=rejected_mask)

                elif task == "countdown":

                    # directly get the reward from the dataset
                    chosen_reward    = batch['chosen_score']
                    rejected_reward  = batch['rejected_score']

                loss = rloo_loss(chosen_lp, rejected_lp, chosen_reward, rejected_reward, k)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                iter += 1
                pbar.update(1)
                
                if iter % 10 == 0:
                    grad_norm = get_grad_norm(model)
                    tqdm.write(f"epoch: {epoch}, iter: {iter}, loss: {loss.item()}, lr: {scheduler.get_last_lr()[0]}, grad_norm: {grad_norm}")
                    wandb.log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0], "grad_norm": grad_norm})
                    
                pbar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])


if __name__ == "__main__":

    with open("config/rloo.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    model_name = config['model_name']
    data_path = config['data_path']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    max_length = config['max_length']

    wandb.init(project="rloo", config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    data = json.load(open(data_path, "r"))
    # TODO: change to countdown if needed
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

    # load reward model, only needed for ultrafeedback
    base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
    reward_model = BTModel(base_model)
    reward_model.to(device)
    reward_model.load_state_dict(torch.load("bt_model.pt", map_location="cuda"))
    reward_model.eval()

    # TODO: change to countdown if needed   
    train(model, dataloader, optimizer, device, scheduler, num_epochs, "ultrafeedback", 2, reward_model)
    model.save_pretrained("models/rloo_model")