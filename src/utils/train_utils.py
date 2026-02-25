# -*- coding: utf-8 -*-
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from src.config import *
from src.model import *
def setup_logging(output_dir: str) -> str:
    """configure file+stdout logging and return the log file path."""
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return log_file

class ParquetSentenceDataset(Dataset):
    def __init__(self, parquet_path: str, max_samples: int = None):
        """load `text` column from parquet and optionally truncate."""
        df = pd.read_parquet(parquet_path, engine="pyarrow")
        assert "text" in df.columns, "Parquet must include a 'text' column."
        self.sentences = df["text"].astype(str).tolist()
        if max_samples is not None:
            self.sentences = self.sentences[:max_samples]
        logging.info(f"[INFO] loaded samples: {len(self.sentences)}")

    def __len__(self) -> int:
        """return number of usable training examples."""
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        """return one raw text sample as {'sentence': str}."""
        return {"text": self.sentences[idx]}

@dataclass
class Collator:
    tokenizer: AutoTokenizer
    max_len: int

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """tokenize a mini-batch into padded `input_ids` and `attention_mask`."""
        texts = [ex["text"] for ex in batch]
        tok = self.tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            padding=True,
        )
        return {
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
        }

def get_adaptive_tau(global_step: int, total_steps: int, tau_init: float, tau_min: float) -> float:
    """anneal Gumbel-Softmax temperature from `tau_init` to `tau_min`."""
    ratio = global_step / max(1, total_steps)
    if ratio < 0.6:
        local = ratio / 0.6
        return tau_init - (tau_init - 0.4) * local
    local = (ratio - 0.6) / 0.4
    return 0.4 - (0.4 - tau_min) * local

class CUDAPrefetcher:
    def __init__(self, loader: DataLoader, device: str = "cuda"):
        """overlap host->device transfer with compute using a side CUDA stream."""
        self.loader_iter = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream(device=device) if torch.cuda.is_available() else None
        self.next_batch = None
        self._preload()

    def _preload(self):
        """asynchronously stage the next batch to target device."""
        try:
            batch = next(self.loader_iter)
        except StopIteration:
            self.next_batch = None
            return
        if self.stream is None:
            self.next_batch = batch
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = {
                "input_ids": batch["input_ids"].to(self.device, non_blocking=True),
                "attention_mask": batch["attention_mask"].to(self.device, non_blocking=True),
            }

    def next(self):
        """return current prepared batch and trigger preload of the following batch."""
        if self.next_batch is None:
            return None
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self._preload()
        return batch

def save_checkpoint(
    model: SharedBackboneUnifiedHead,
    tokenizer: AutoTokenizer,
    metas: List[ConceptTypeMeta],
    step: int,
    output_dir: str,
    optimizer=None,
    scheduler=None,
    scaler=None,
):
    """persist backbone/tokenizer and lightweight planner metadata."""
    save_name = f"checkpoint-{step}" if isinstance(step, int) else str(step)
    save_dir = os.path.join(output_dir, save_name)
    backbone_dir = os.path.join(save_dir, "backbone")
    os.makedirs(backbone_dir, exist_ok=True)

    model.base_model.save_pretrained(backbone_dir)
    tokenizer.save_pretrained(backbone_dir)

    meta_dump = [
        {
            "name": m.name,
            "type_id": m.type_id,
            "eos_id": m.eos_id,
            "concept_ids": m.concept_ids.tolist(),
            "max_steps": m.max_steps,
            "target_ratio": m.target_ratio,
        }
        for m in metas
    ]

    planner_quota_state = None
    if hasattr(model, "planner_quota") and model.planner_quota is not None:
        planner_quota_state = model.planner_quota.state_dict()

    token_embed_new = model.get_new_token_embed_weight().detach().cpu()

    torch.save(
        {
            "output_head_base": model.output_head_base.state_dict(),
            "output_head_new": model.output_head_new.state_dict() if model.output_head_new is not None else None,
            "type_embed": model.type_embed.state_dict(),
            "token_embed_new": token_embed_new,
            "concept_metas": meta_dump,
            "planner_quota": planner_quota_state,
            "step": step,
            "tokenizer_size": len(tokenizer),
        },
        os.path.join(save_dir, "two_stage_heads.pt"),
    )
    if optimizer is not None and scheduler is not None and isinstance(step, int):
        trainer_state = {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        if scaler is not None and scaler.is_enabled():
            trainer_state["scaler"] = scaler.state_dict()
        torch.save(trainer_state, os.path.join(save_dir, "trainer_state.pt"))
    logging.info(f"[SAVE] checkpoint saved to: {save_dir}")

