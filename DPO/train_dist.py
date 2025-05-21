import os
import json
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from dpo_dataset import DPODataset, DPOConfig
from tqdm import tqdm
from omegaconf import OmegaConf
import wandb


def setup_distributed():
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

def cleanup_distributed():
    dist.destroy_process_group()

def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def get_log_prob(logits, labels, prompt_lengths):
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)

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

def train(model, ref_model, dataloader, device, cfg, scheduler, optimizer, tokenizer):
    ref_model.eval()
    ref_model.requires_grad_(False)
    model.train()

    running_loss = 0.0
    iter_idx = 0
    total_steps = cfg.num_epochs * len(dataloader)

    if dist.get_rank() == 0:
        pbar = tqdm(total=total_steps)
    else:
        pbar = None

    for epoch in range(cfg.num_epochs):
        dataloader.sampler.set_epoch(epoch)

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

            with torch.inference_mode():
                ref_logits_chosen = ref_model(chosen_ids, attention_mask=chosen_mask).logits
                ref_logits_rejected = ref_model(rejected_ids, attention_mask=rejected_mask).logits

            chosen_lp_ref   = get_log_prob(ref_logits_chosen, chosen_ids, prompt_len)
            rejected_lp_ref = get_log_prob(ref_logits_rejected, rejected_ids, prompt_len)

            loss = loss_dpo(chosen_lp, rejected_lp, chosen_lp_ref, rejected_lp_ref, cfg.beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            if (iter_idx + 1) % cfg.log_steps == 0 and dist.get_rank() == 0:
                avg_loss = running_loss / cfg.log_steps
                grad_norm = get_grad_norm(model)

                tqdm.write(
                    f"epoch {epoch}, step {iter_idx}, avg_loss({cfg.log_steps}) {avg_loss:.4f}, "
                    f"lr {scheduler.get_last_lr()[0]:.2e}, grad_norm {grad_norm:.2f}"
                )
                wandb.log({
                    "avg_loss": avg_loss,
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": grad_norm
                })
                running_loss = 0.0

            if (iter_idx + 1) % cfg.save_steps == 0 and dist.get_rank() == 0:
                model.module.save_pretrained(f"/viscam/u/tonyzst/Research/test/DPO/ckpts/step_{iter_idx}")
                tokenizer.save_pretrained(f"/viscam/u/tonyzst/Research/test/DPO/ckpts/step_{iter_idx}")

            iter_idx += 1
            if pbar is not None:
                pbar.update(1)

    if iter_idx % cfg.log_steps and dist.get_rank() == 0:
        avg_loss = running_loss / (iter_idx % cfg.log_steps)
        wandb.log({"avg_loss_tail": avg_loss})

if __name__ == "__main__":
    tqdm.disable = True
    setup_distributed()
    local_rank = dist.get_rank() % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_rank}")

    raw_cfg = OmegaConf.load("config/dpo.yaml")
    cfg = OmegaConf.structured(DPOConfig(**raw_cfg))

    if dist.get_rank() == 0:
        wandb.init(project="dpo", config=OmegaConf.to_container(raw_cfg, resolve=True))

    data = json.load(open(cfg.data_path, "r"))
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    ref_model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(device)

    dataset = DPODataset(data, tokenizer, max_length=cfg.max_length)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    num_training_steps = len(dataloader) * cfg.num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    train(model, ref_model, dataloader, device, cfg, scheduler, optimizer, tokenizer)

    if dist.get_rank() == 0:
        model.module.save_pretrained("/viscam/u/tonyzst/Research/test/DPO/ckpts/last")
        tokenizer.save_pretrained("/viscam/u/tonyzst/Research/test/DPO/ckpts/last")

    cleanup_distributed()
