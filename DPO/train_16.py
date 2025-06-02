import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from dpo_dataset import DPODataset, DPOConfig
import json
from tqdm import tqdm
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
import wandb
from omegaconf import OmegaConf
from torch.cuda.amp import autocast, GradScaler


def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2) 
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def get_log_prob(logits, labels, prompt_lengths, attention_mask):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = attention_mask[..., 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)

    batch_size, seq_len = shift_labels.shape
    shift_prompt_lengths = (prompt_lengths - 1).clamp(min=0)
    response_mask = torch.arange(seq_len, device=shift_labels.device).unsqueeze(0) >= shift_prompt_lengths.unsqueeze(1)
    response_mask = response_mask.float() # * shift_mask.float()

    response_log_probs = (token_log_probs * response_mask).sum(dim=-1)
    response_lengths = response_mask.sum(dim=-1).clamp(min=1)

    return response_log_probs / response_lengths


def loss_dpo(chosen, rejected, chosen_ref, rejected_ref, beta):
    chosen_ratio = chosen - rejected
    rejected_ratio = chosen_ref - rejected_ref
    
    logits = beta * (chosen_ratio - rejected_ratio)
    loss = -F.logsigmoid(logits).mean()
    
    reward_acc = (chosen_ratio > rejected_ratio).float().mean().detach()
    reward_diff = (chosen_ratio - rejected_ratio).mean().detach()

    return loss, reward_acc, reward_diff, chosen_ratio.mean().detach(), rejected_ratio.mean().detach()

def train(model, ref_model, dataloader, device, cfg, scheduler, optimizer, tokenizer):
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
        
    model.train()
    scaler = GradScaler()
    
    running_loss = 0.0    
    running_reward_acc = 0.0
    running_reward = 0.0
    running_chosen_ratio = 0.0
    running_rejected_ratio = 0.0
    iter_idx = 0
    grad_accum_steps = cfg.gradient_accumulation_steps
    optimizer.zero_grad()

    total_steps = cfg.num_epochs * len(dataloader)
    with tqdm(total=total_steps) as pbar:
        for epoch in range(cfg.num_epochs):
            for batch in dataloader:
                # Get prompt lengths (need to account for shifting)
                prompt_len = batch['attention_mask_prompt'].sum(dim=1).to(device) - 1  # Subtract 1 for shifting
                chosen_ids    = batch['input_ids_chosen'].to(device)
                rejected_ids  = batch['input_ids_rejected'].to(device)
                chosen_mask   = batch['attention_mask_chosen'].to(device)
                rejected_mask = batch['attention_mask_rejected'].to(device)

                with autocast():
                    chosen_logits   = model(chosen_ids,   attention_mask=chosen_mask).logits
                    rejected_logits = model(rejected_ids, attention_mask=rejected_mask).logits

                    chosen_lp   = get_log_prob(chosen_logits,   chosen_ids,   prompt_len, chosen_mask)
                    rejected_lp = get_log_prob(rejected_logits, rejected_ids, prompt_len, rejected_mask)

                with torch.no_grad(), autocast():
                    chosen_lp_ref   = get_log_prob(
                        ref_model(chosen_ids,   attention_mask=chosen_mask).logits,
                        chosen_ids, prompt_len, chosen_mask)
                    rejected_lp_ref = get_log_prob(
                        ref_model(rejected_ids, attention_mask=rejected_mask).logits,
                        rejected_ids, prompt_len, rejected_mask)

                with autocast():
                    loss, reward_acc, reward_diff, chosen_ratio, rejected_ratio = loss_dpo(chosen_lp, rejected_lp, chosen_lp_ref, rejected_lp_ref, cfg.beta)
                
                running_loss += loss.item()
                running_reward_acc += reward_acc.item()
                running_reward += reward_diff.item()
                running_chosen_ratio += chosen_ratio.item()
                running_rejected_ratio += rejected_ratio.item()

                loss = loss / grad_accum_steps
                scaler.scale(loss).backward()

                if (iter_idx + 1) % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                if (iter_idx + 1) % cfg.log_steps == 0:
                    grad_norm = get_grad_norm(model)

                    avg_loss = running_loss / cfg.log_steps
                    avg_reward_acc = running_reward_acc / cfg.log_steps
                    avg_reward = running_reward / cfg.log_steps
                    avg_chosen_ratio = running_chosen_ratio / cfg.log_steps
                    avg_rejected_ratio = running_rejected_ratio / cfg.log_steps

                    tqdm.write(
                        f"epoch {epoch}, step {iter_idx}, "
                        f"loss {avg_loss:.4f}, "
                        f"reward_acc {avg_reward_acc:.4f}, "
                        f"reward_diff {avg_reward:.4f}, "
                        f"chosen_ratio {avg_chosen_ratio:.4f}, "
                        f"rejected_ratio {avg_rejected_ratio:.4f}, "
                        f"lr {scheduler.get_last_lr()[0]:.2e}, "
                        f"grad_norm {grad_norm:.2f}"
                    )
                    wandb.log({
                        "loss": avg_loss,
                        "reward_acc": avg_reward_acc,
                        "reward_diff": avg_reward,
                        "chosen_ratio": avg_chosen_ratio,
                        "rejected_ratio": avg_rejected_ratio,
                        "lr": scheduler.get_last_lr()[0],
                        "grad_norm": grad_norm,
                        "step": iter_idx + 1
                    })

                    running_loss = 0.0
                    running_reward_acc = 0.0
                    running_reward = 0.0

                if (iter_idx + 1) % cfg.save_steps == 0:
                    model.save_pretrained(f"/viscam/u/tonyzst/Research/test/DPO/ckpts/step_{iter_idx + 1}")
                    tokenizer.save_pretrained(f"/viscam/u/tonyzst/Research/test/DPO/ckpts/step_{iter_idx + 1}")

                iter_idx += 1
                pbar.update(1)

        if iter_idx % cfg.log_steps != 0:
            steps_remaining = iter_idx % cfg.log_steps
            avg_loss = running_loss / steps_remaining
            wandb.log({"final_avg_loss": avg_loss})

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
    model.gradient_checkpointing_enable()

    dataset = DPODataset(data, tokenizer, max_length=cfg.max_length)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    num_training_steps = len(dataloader) * cfg.num_epochs // cfg.gradient_accumulation_steps
    num_warmup_steps = int(0.1 * num_training_steps) 

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    train(model, ref_model, dataloader, device, cfg, scheduler, optimizer, tokenizer)

    model.save_pretrained("/viscam/u/tonyzst/Research/test/DPO/ckpts/dpo_last")
    tokenizer.save_pretrained("/viscam/u/tonyzst/Research/test/DPO/ckpts/dpo_last")