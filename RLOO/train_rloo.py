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

from reward_model import RewardModel
from rloo_dataset import RLOODataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eval")))
import eval_countdown


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

    _, seq_len = labels.shape
    response_mask = torch.arange(seq_len, device=labels.device).unsqueeze(0) >= prompt_lengths.unsqueeze(1)
    response_mask = response_mask.float()
    response_log_probs = (token_log_probs * response_mask).sum(dim=-1)
    response_lengths = response_mask.sum(dim=-1).clamp(min=1)

    return response_log_probs / response_lengths


def rloo_loss(response_lp_list, response_reward_list, k, device):
    response_lp = torch.stack(response_lp_list).to(device)
    response_reward = torch.stack(response_reward_list).to(device)
    assert response_lp.shape == response_reward.shape

    total_reward = response_reward.sum().unsqueeze(0)
    reward_LOO = 1/(k-1) * (total_reward - response_reward)
    diff_reward = response_reward - reward_LOO

    RLOOloss = -(diff_reward * response_lp).mean()
    
    return RLOOloss


def train(model, tokenizer, dataloader, optimizer, device, scheduler, cfg, max_length=1024, reward_model=None):
    
    num_epochs = cfg['num_epochs']
    grad_accum_steps = cfg['gradient_accumulation_steps']
    log_steps = cfg['log_steps']
    save_steps = cfg['save_steps']
    k = cfg["k"]
    eval_method = config["eval_method"]

    if reward_model:
        reward_model.eval()
        reward_model.requires_grad_(False)
    
    model.train()
    iter = 0
    running_loss = 0.0
    optimizer.zero_grad()
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
                    response_ids = model.generate(input_ids_prompt, attention_mask=attention_mask_prompt, max_new_tokens=max_length, do_sample=True, pad_token_id=tokenizer.eos_token_id)
                    # response_text = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                    # for i, text in enumerate(response_text):
                    #     print(f"Response {i}:", text)
                    response_mask = (response_ids != tokenizer.pad_token_id).to(device)
                    response_logits = model(response_ids, attention_mask=response_mask).logits
                    response_lp = get_log_prob(response_logits, response_ids, prompt_len)
                    response_lp_list.append(response_lp)

                    if eval_method == "reward_model":
                        # get the reward from the reward model
                        with torch.no_grad():
                            response_reward = reward_model(response_ids, attention_mask=response_mask)
                            # print("response_reward:", response_reward)
                    
                    elif eval_method == "reward_function":
                        # get the reward from the reward function
                        batch_size = response_ids.shape[0]
                        response_reward = []

                        for i in range(batch_size):
                            ground_truth = batch["ground_truth"][i]
                            prompt_text = tokenizer.decode(input_ids_prompt[i], skip_special_tokens=True)
                            response_text = tokenizer.decode(response_ids[i], skip_special_tokens=True)
                            length = len(prompt_text)
                            response_no_prompt = response_text[length:]
                            # print("ground_truth:", ground_truth)
                            # print("response_text:", response_text)
                            # print("prompt_text:", prompt_text)
                            # print("response_no_prompt:", response_no_prompt)
                            reward = eval_countdown.compute_score(
                                solution_str=f"Assistant: {response_no_prompt}".strip(),
                                ground_truth=ground_truth,
                                method="strict",
                                format_score=0.1,
                                score=1.0
                            )
                            # print("reward:", reward)
                            response_reward.append(reward)

                        response_reward = torch.tensor(response_reward, dtype=torch.float32, device=device)
                        # print("response_reward:", response_reward)
                    
                    response_reward_list.append(response_reward)

                loss = rloo_loss(response_lp_list, response_reward_list, k, device)
                print("loss:", loss.item())

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
                    model.save_pretrained(f"models/rloo_model_{iter}")
                    tokenizer.save_pretrained(f"models/rloo_model_{iter}")


if __name__ == "__main__":

    with open("config/rloo.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    model_name = config["model_name"]
    data_path = config["data_path"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    max_length = config["max_length"]
    reward_model_name = config["reward_model_name"]
    reward_model_weights_path = config["reward_model_weights_path"]
    task = config["task"]
    eval_method = config["eval_method"]

    wandb.init(project="rloo", config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    data = json.load(open(data_path, "r"))
    dataset = RLOODataset(data=data, tokenizer=tokenizer, task=task, max_length=max_length)

    if task == "countdown" and eval_method == "reward_function":
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=RLOODataset.collate_fn)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(dataloader) * num_epochs // config['gradient_accumulation_steps']
    num_warmup_steps = int(0.05 * num_training_steps)  

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    if eval_method == "reward_model":
        # load reward model
        base_model = AutoModelForCausalLM.from_pretrained(reward_model_name)
        reward_model = RewardModel(base_model)
        reward_model.to(device)
        reward_model.load_state_dict(torch.load(reward_model_weights_path, map_location="cuda"))
    else:
        reward_model = None

    train(model, tokenizer, dataloader, optimizer, device, scheduler, config, max_length, reward_model)
    model.save_pretrained("models/rloo_model")
    tokenizer.save_pretrained("models/rloo_model")