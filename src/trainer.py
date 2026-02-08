import logging
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from src.config import ExperimentConfig
from src.model import OnlineConceptCompressionModel


def setup_logging(output_dir: str) -> str:
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_qwen_2_{ts}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )
    return log_file


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ParquetSentenceDataset(Dataset):
    def __init__(self, parquet_path: str, max_samples: int | None = None):
        df = pd.read_parquet(parquet_path, engine="pyarrow")
        assert "sentence" in df.columns, "Parquet must contain a 'sentence' column."
        sentences = df["sentence"].astype(str).tolist()
        if max_samples is not None:
            sentences = sentences[:max_samples]
        self.sentences = sentences
        logging.info(f"[INFO] Loaded {len(self.sentences)} sentences from parquet.")

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return {"sentence": self.sentences[idx]}


@dataclass
class Collator:
    tokenizer: AutoTokenizer
    max_len: int

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        texts = [x["sentence"] for x in batch]
        out = self.tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_len,
            padding=True,
            return_tensors="pt",
        )
        return {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"]}


def get_tau(global_step: int, total_steps: int, tau_init: float, tau_min: float) -> float:
    ratio = global_step / max(1, total_steps)
    return tau_init + (tau_min - tau_init) * ratio


def detect_lora_targets(model: nn.Module, candidates: List[str]) -> List[str]:
    found = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            leaf = name.split(".")[-1]
            if leaf in candidates:
                found.add(leaf)
    if not found:
        raise RuntimeError("No LoRA target modules found. Check model architecture names.")
    return sorted(found)


def save_checkpoint(model: OnlineConceptCompressionModel, tokenizer: AutoTokenizer, step: int, output_dir: str) -> None:
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    lora_dir = os.path.join(ckpt_dir, "lora")
    os.makedirs(lora_dir, exist_ok=True)
    model.peft_model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    torch.save(
        {
            "concept_head": model.concept_head.state_dict(),
            "concept_embeddings": model.concept_embeddings.state_dict(),
            "tail_mlp": model.tail_mlp.state_dict(),
            "concept_vocab_size": model.concept_vocab_size,
            "null_id": model.null_id,
            "shallow_end": model.shallow_end,
            "middle_end": model.middle_end,
        },
        os.path.join(ckpt_dir, "concept_modules.pt"),
    )
    logging.info(f"[SAVE] checkpoint saved to: {ckpt_dir}")


def train(cfg: ExperimentConfig) -> None:
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    log_file = setup_logging(cfg.paths.output_dir)
    set_seed(cfg.train.seed)
    logging.info(f"[INFO] Logging to: {log_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"[INFO] Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.paths.base_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.paths.base_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            attn_implementation="eager",
        )
    except TypeError:
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.paths.base_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    base_model.to(device)

    target_modules = detect_lora_targets(base_model, cfg.adapter.target_candidates)
    lora_cfg = PeftLoraConfig(
        r=cfg.adapter.r,
        lora_alpha=cfg.adapter.alpha,
        target_modules=target_modules,
        lora_dropout=cfg.adapter.dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, lora_cfg)
    peft_model.to(device)

    model = OnlineConceptCompressionModel(peft_model=peft_model, cfg=cfg.model).to(device)

    # Hard constraint: train only (new modules + LoRA params).
    for name, param in model.named_parameters():
        is_new_module = (
            name.startswith("concept_head.")
            or name.startswith("concept_embeddings.")
            or name.startswith("tail_mlp.")
        )
        is_lora = "lora_" in name
        param.requires_grad = bool(is_new_module or is_lora)

    dataset = ParquetSentenceDataset(cfg.paths.parquet_path, max_samples=cfg.train.max_samples)
    collate_fn = Collator(tokenizer=tokenizer, max_len=cfg.train.max_input_tokens)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=collate_fn,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_params)
    logging.info(f"[INFO] Trainable params: {total_trainable:,}")
    logging.info(f"[INFO] LoRA targets: {target_modules}")
    logging.info(
        f"[INFO] Layer split: shallow[0:{model.shallow_end}), "
        f"middle[{model.shallow_end}:{model.middle_end}), "
        f"deep[{model.middle_end}:{len(model.layers)})"
    )

    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    total_steps = math.ceil(len(dataloader) / cfg.train.grad_accum) * cfg.train.epochs
    warmup_steps = int(total_steps * cfg.train.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    global_step = 0
    micro_step = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(total=total_steps, desc="train_qwen_2", unit="step")
    running_loss = 0.0
    running_ratio = 0.0
    running_rec = 0.0
    running_commit = 0.0
    running_unif = 0.0
    running_len = 0.0

    for epoch in range(cfg.train.epochs):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            tau = get_tau(global_step, total_steps, cfg.train.tau_init, cfg.train.tau_min)

            loss, batch_stats = model.forward_batch(input_ids, attention_mask, tau=tau)
            if batch_stats["num_targets"] < 1:
                continue

            loss_scaled = loss / cfg.train.grad_accum
            loss_scaled.backward()
            micro_step += 1

            if micro_step % cfg.train.grad_accum != 0:
                continue

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            running_loss += float(loss.detach().cpu().item())
            running_ratio += float(batch_stats["compress_ratio"])
            running_rec += float(batch_stats["loss_rec"])
            running_commit += float(batch_stats["loss_commit"])
            running_unif += float(batch_stats["loss_unif"])
            running_len += float(batch_stats["loss_len"])
            pbar.update(1)

            if global_step % cfg.train.log_steps == 0:
                avg_loss = running_loss / cfg.train.log_steps
                avg_ratio = running_ratio / cfg.train.log_steps
                avg_rec = running_rec / cfg.train.log_steps
                avg_commit = running_commit / cfg.train.log_steps
                avg_unif = running_unif / cfg.train.log_steps
                avg_len = running_len / cfg.train.log_steps
                lr_now = scheduler.get_last_lr()[0]
                logging.info(
                    f"[Epoch {epoch+1}/{cfg.train.epochs}] Step {global_step}/{total_steps} | "
                    f"Loss {avg_loss:.4f} | Rec {avg_rec:.4f} | Commit {avg_commit:.4f} | "
                    f"Unif {avg_unif:.4f} | Len {avg_len:.4f} | CompressRatio {avg_ratio:.4f} | "
                    f"Tau {tau:.4f} | LR {lr_now:.2e}"
                )
                running_loss = 0.0
                running_ratio = 0.0
                running_rec = 0.0
                running_commit = 0.0
                running_unif = 0.0
                running_len = 0.0

            if global_step % cfg.train.save_steps == 0:
                save_checkpoint(model, tokenizer, global_step, cfg.paths.output_dir)

            if global_step >= total_steps:
                break

        if global_step >= total_steps:
            break

    pbar.close()
    save_checkpoint(model, tokenizer, global_step, cfg.paths.output_dir)
    logging.info("[DONE] Training finished.")

