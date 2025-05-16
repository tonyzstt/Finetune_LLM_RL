import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from dpo_dataset import DPODataset, DPOConfig
import json
import yaml
from tqdm import tqdm
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
import wandb
from omegaconf import OmegaConf
import deepspeed


def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2) 
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def get_grad_norm_ds(model):
    total_norm = 0.0
    for _, param in model.named_parameters():
        grad = deepspeed.utils.safe_get_full_grad(param).norm(2) 
        if grad is not None:
            total_norm += grad.item() ** 2
    return total_norm ** 0.5

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

def train(model, ref_model, dataloader, device, cfg, scheduler):
    ref_model.eval()
    ref_model.requires_grad_(False)
    model.train()
      
    running_loss = 0.0           
    iter_idx = 0

    total_steps = cfg.num_epochs * len(dataloader)
    with tqdm(total=total_steps) as pbar:
        for epoch in range(cfg.num_epochs):
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

                with torch.no_grad():
                    chosen_lp_ref   = get_log_prob(
                        ref_model(chosen_ids,   attention_mask=chosen_mask).logits,
                        chosen_ids, prompt_len)
                    rejected_lp_ref = get_log_prob(
                        ref_model(rejected_ids, attention_mask=rejected_mask).logits,
                        rejected_ids, prompt_len)

                loss = loss_dpo(chosen_lp, rejected_lp,
                                chosen_lp_ref, rejected_lp_ref, cfg.beta)

                model.backward(loss)
                model.step()
                scheduler.step()

                running_loss += loss.item()
                if (iter_idx + 1) % cfg.log_steps == 0:
                    avg_loss = running_loss / cfg.log_steps
                    grad_norm = get_grad_norm_ds(model)

                    tqdm.write(
                        f"epoch {epoch}, step {iter_idx}, "
                        f"avg_loss({cfg.log_steps}) {avg_loss:.4f}, "
                        f"lr {scheduler.get_last_lr()[0]:.2e}, "
                        f"grad_norm {grad_norm:.2f}"
                    )
                    wandb.log({
                        "avg_loss": avg_loss,
                        "lr": scheduler.get_last_lr()[0],
                        "grad_norm": grad_norm
                    })
                    running_loss = 0.0  

                if (iter_idx + 1) % cfg.save_steps == 0:
                    model.save_checkpoint(f"ckpts/dpo_peft/step_{iter_idx}")
                    tokenizer.save_pretrained(f"ckpts/dpo_peft/step_{iter_idx}")

                iter_idx += 1
                pbar.update(1)

        if iter_idx % cfg.log_steps:
            avg_loss = running_loss / (iter_idx % cfg.log_steps)
            wandb.log({"avg_loss_tail": avg_loss})

                    
            
            

if __name__ == "__main__":

    raw_cfg = OmegaConf.load("config/dpo.yaml")
    cfg = OmegaConf.structured(DPOConfig(**raw_cfg))

    wandb.init(project="dpo", config=OmegaConf.to_container(raw_cfg, resolve=True))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = json.load(open(cfg.data_path, "r"))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(device)

    dataset = DPODataset(data, tokenizer, max_length=cfg.max_length)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    num_training_steps = len(dataloader) * cfg.num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)  

    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=cfg.deepspeed_config,
    )

    optimizer = optimizer.optimizer

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    train(model, ref_model, dataloader, device, cfg, scheduler)
    model.save_checkpoint("ckpts/dpo_peft/last")
    tokenizer.save_pretrained("ckpts/dpo_peft/last")
