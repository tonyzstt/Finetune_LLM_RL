import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import get_cosine_schedule_with_warmup
import yaml
import wandb
from tqdm import tqdm
import json
import os
import sys

from bt_model import BTModel
from rloo_dataset import RLOODataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eval")))
import countdown


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


def rloo_loss(response_lp_list, response_reward_list, k, device):
    assert len(response_lp_list) == len(response_reward_list) == k
    response_lp = torch.stack(response_lp_list).to(device).squeeze(1)
    response_reward = torch.tensor(response_reward_list).to(device)

    total_reward = response_reward.sum().unsqueeze(0)
    reward_LOO = (total_reward - response_reward) * 1/(k-1)
    diff_reward = response_reward - reward_LOO

    assert diff_reward.shape == response_lp.shape
    RLOOloss = -(diff_reward * response_lp).mean()
    
    return RLOOloss


def train(model, tokenizer, dataloader, optimizer, device, scheduler, num_epochs, task, k, max_length=1024, reward_model=None):
    model.train()
    iter = 0

    total_steps = num_epochs * len(dataloader)
    with tqdm(total=total_steps) as pbar:
        for epoch in range(num_epochs):
            for batch in dataloader:

                prompt_len = batch["attention_mask_prompt"].sum(dim=1).to(device)

                input_ids_prompt = batch["input_ids_prompt"].to(device)
                attention_mask_prompt = batch["attention_mask_prompt"].to(device)

                # generate k responses for each prompt
                response_lp_list = []
                response_reward_list = []
                for _ in range(k):
                    response = model.generate(input_ids_prompt, attention_mask=attention_mask_prompt, max_new_tokens=max_length, do_sample=True, pad_token_id=tokenizer.eos_token_id)
                    response_text = tokenizer.decode(response[0], skip_special_tokens=True)
                    # print("response_text:", response_text)

                    enc_response = tokenizer(response_text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
                    response_ids = enc_response["input_ids"].to(device)
                    response_mask = enc_response["attention_mask"].to(device)
                    response_logits = model(response_ids, attention_mask=response_mask).logits
                    response_lp = get_log_prob(response_logits, response_ids, prompt_len)
                    response_lp_list.append(response_lp)

                    if task == "ultrafeedback":
                        # get the reward from the reward model
                        response_reward = reward_model(response_ids, attention_mask=response_mask)
                        # print("response_reward:", response_reward)

                    elif task == "countdown":
                        # get the reward from the rule-based reward function
                        ground_truth = batch["ground_truth"]
                        ground_truth = {
                            "target": ground_truth["target"].item(),
                            "numbers": [x.item() for x in ground_truth["numbers"]]
                        }
                        prompt = tokenizer.decode(input_ids_prompt[0], skip_special_tokens=True)
                        length = len(prompt)
                        response_no_prompt = response_text[length:]
                        response_reward = countdown.compute_score(response_no_prompt, ground_truth)

                    response_reward_list.append(response_reward)

                loss = rloo_loss(response_lp_list, response_reward_list, k, device)

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
        
    model_name = config["model_name"]
    data_path = config["data_path"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    max_length = config["max_length"]
    bt_model_name = config["bt_model_name"]
    bt_model_weights_path = config["bt_model_weights_path"]
    task = config["task"]
    k = config["k"]

    wandb.init(project="rloo", config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

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

    if task == "ultrafeedback":
        # load reward model, only needed for ultrafeedback
        base_model = AutoModelForCausalLM.from_pretrained(bt_model_name)
        reward_model = BTModel(base_model)
        reward_model.to(device)
        reward_model.load_state_dict(torch.load(bt_model_weights_path, map_location="cuda"))
        reward_model.eval()
    else:
        reward_model = None

    train(model, tokenizer, dataloader, optimizer, device, scheduler, num_epochs, task, k, max_length, reward_model)
    model.save_pretrained("models/rloo_model")
    tokenizer.save_pretrained("models/rloo_model")