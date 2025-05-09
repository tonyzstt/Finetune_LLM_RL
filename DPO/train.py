import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from dpo_dataset import DPODataset
import json
import yaml
from tqdm import tqdm
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
import wandb


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

def loss_dpo(chosen, rejected, chosen_ref, rejected_ref, beta):
    
    relative = chosen - rejected
    relative_ref = chosen_ref - rejected_ref

    return -torch.log(torch.sigmoid(beta * (relative - relative_ref))).mean()

def train(model, ref_model, dataloader, optimizer, device, beta, scheduler):

     # freeze the reference model, train the model
    ref_model.eval()
    ref_model.requires_grad_(False)
    model.train()
    iter = 0
    
    total_steps = num_epochs * len(dataloader)
    with tqdm(total=total_steps) as pbar:
        for epoch in range(num_epochs):
            for batch in dataloader:

                optimizer.zero_grad()

                prompt_length = batch['attention_mask_prompt'].sum(dim=1).to(device)    

                chosen_ids = batch['input_ids_chosen'].to(device)
                rejected_ids = batch['input_ids_rejected'].to(device)
                chosen_mask = batch['attention_mask_chosen'].to(device)
                rejected_mask = batch['attention_mask_rejected'].to(device)
                
                chosen_logits = model(chosen_ids, attention_mask=chosen_mask).logits
                rejected_logits = model(rejected_ids, attention_mask=rejected_mask).logits

                chosen_log_prob = get_log_prob(chosen_logits, chosen_ids, prompt_length)
                rejected_log_prob = get_log_prob(rejected_logits, rejected_ids, prompt_length)

                with torch.no_grad():
                    chosen_logits_ref = ref_model(chosen_ids, attention_mask=chosen_mask).logits
                    rejected_logits_ref = ref_model(rejected_ids, attention_mask=rejected_mask).logits

                    chosen_log_prob_ref = get_log_prob(chosen_logits_ref, chosen_ids, prompt_length)
                    rejected_log_prob_ref = get_log_prob(rejected_logits_ref, rejected_ids, prompt_length)

                loss = loss_dpo(chosen_log_prob, rejected_log_prob, chosen_log_prob_ref, rejected_log_prob_ref, beta)
                loss.backward()
                optimizer.step()
                scheduler.step()
                iter += 1
                pbar.update(1)

                # TODO: add validation here
                if iter % 10 == 0:
                    grad_norm = get_grad_norm(model)
                    tqdm.write(f"epoch {epoch}, step {iter}, loss {loss.item()}, lr {scheduler.get_last_lr()[0]}, grad_norm {grad_norm}")
                    wandb.log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0], "grad_norm": grad_norm})
            
            

if __name__ == "__main__":

    with open("dpo.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_name = config["model_name"]
    data_path = config["data_path"]
    batch_size = config["batch_size"]
    max_length = config["max_length"]
    learning_rate = config["learning_rate"]
    num_epochs = config["num_epochs"]
    beta = config["beta"]

    wandb.init(project="dpo", config=config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = json.load(open(config["data_path"], "r"))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    dataset = DPODataset(data, tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(dataloader) * num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)  

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    train(model, ref_model, dataloader, optimizer, device, beta, scheduler)
    model.save_pretrained("models/dpo_model")
