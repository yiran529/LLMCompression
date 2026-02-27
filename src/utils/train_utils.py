# -*- coding: utf-8 -*-
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from src.config.train_config import *
from src.model import *

# ============================================================
# General training utilities (dataset / logging / checkpoint)
# ============================================================

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


# ============================================================
# Weights & Biases (wandb) utilities
# ============================================================

def _to_mb(x: int) -> float:
    return float(x) / (1024.0 * 1024.0)


def to_mb(x: int) -> float:
    return _to_mb(x)


def _build_wandb_config(
    *,
    total_steps: int,
    warmup_steps: int,
    trainable_params: int,
    all_params: int,
    new_rows: int,
    device: str,
) -> Dict[str, Any]:
    """collect stable hyperparameters for this run."""
    return {
        "base_dir": BASE_DIR,
        "parquet_path": PARQUET_PATH,
        "output_dir": OUTPUT_DIR,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "epochs": EPOCHS,
        "lr": LR,
        "warmup_ratio": WARMUP_RATIO,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "save_steps": SAVE_STEPS,
        "log_steps": LOG_STEPS,
        "seed": SEED,
        "model_dtype": MODEL_DTYPE,
        "attention_impl": ATTENTION_IMPL,
        "use_compile": USE_COMPILE,
        "compile_mode": COMPILE_MODE,
        "fp32_trainable": FP32_TRAINABLE,
        "tau_init": TAU_INIT,
        "tau_min": TAU_MIN,
        "min_concept_steps": MIN_CONCEPT_STEPS,
        "lambda_rec": LAMBDA_REC,
        "lambda_commit": LAMBDA_COMMIT,
        "lambda_unif": LAMBDA_UNIF,
        "lambda_eos": LAMBDA_EOS,
        "lambda_len": LAMBDA_LEN,
        "quota_tau": QUOTA_TAU,
        "quota_eta": QUOTA_ETA,
        "quota_lambda_init": QUOTA_LAMBDA_INIT,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lora_target_modules": LORA_TARGET_MODULES,
        "lora_modules_to_save": LORA_MODULES_TO_SAVE,
        "resume_enabled": RESUME_ENABLED,
        "resume_checkpoint_dir": RESUME_CHECKPOINT_DIR,
        "concept_type_configs": [cfg.__dict__ for cfg in CONCEPT_TYPE_CONFIGS],
        "trainable_params": trainable_params,
        "all_params": all_params,
        "output_head_new_rows": new_rows,
        "device": device,
    }


def init_wandb_run(
    *,
    model: torch.nn.Module,
    total_steps: int,
    warmup_steps: int,
    trainable_params: int,
    all_params: int,
    new_rows: int,
    device: str,
):
    """initialize wandb run with strong defaults and minimal external switches."""
    try:
        import wandb  # type: ignore
    except Exception:
        logging.warning("[WARN] wandb is not installed; training metrics will only be logged to local logger")
        return None

    wandb_kwargs: Dict[str, Any] = {
        "project": WANDB_PROJECT,
        "config": _build_wandb_config(
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            trainable_params=trainable_params,
            all_params=all_params,
            new_rows=new_rows,
            device=device,
        ),
        "mode": "online", #! 暂时先写死online，后续可以通过环境变量WANDB_MODE切换到offline或disabled
        "save_code": True,
        "dir": OUTPUT_DIR,
        "settings": wandb.Settings(x_stats_sampling_interval=15),
    }
    if WANDB_ENTITY:
        wandb_kwargs["entity"] = WANDB_ENTITY
    if WANDB_RUN_NAME:
        wandb_kwargs["name"] = WANDB_RUN_NAME

    run = wandb.init(**wandb_kwargs)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    run.log_code(
        root=project_root,
        include_fn=lambda p: p.endswith((".py", ".md", ".yaml", ".yml", ".json")),
    )
    run.watch(model, log="gradients", log_freq=max(1, LOG_STEPS))
    logging.info(
        f"[INFO] wandb run started: project={WANDB_PROJECT}, mode={os.getenv('WANDB_MODE', 'online')}, "
        f"name={WANDB_RUN_NAME or '(auto)'}"
    )
    return run


def finish_wandb_run(wandb_run, *, total_seconds: int, final_global_step: int) -> None:
    """finalize wandb run and write end-of-run summary metrics."""
    if wandb_run is None:
        return
    wandb_run.summary["total_training_seconds"] = int(total_seconds)
    wandb_run.summary["final_global_step"] = int(final_global_step)
    wandb_run.finish()


def log_wandb_step_metrics(
    *,
    wandb_run,
    global_step: int,
    epoch: int,
    metas: List[ConceptTypeMeta],
    avg_lens: List[float],
    avg_loss: float,
    loss: torch.Tensor,
    loss_rec: torch.Tensor,
    loss_commit: torch.Tensor,
    loss_unif: torch.Tensor,
    loss_eos: torch.Tensor,
    loss_len: torch.Tensor,
    loss_quota: torch.Tensor,
    quota_bar: torch.Tensor,
    quota_lambda: torch.Tensor,
    grad_norm: torch.Tensor,
    tau: float,
    lr: float,
    step_wall_time: float,
    step_tokens: int,
    scaler,
    stage_metrics: Dict[str, float],
) -> None:
    """build and log one training-step metric packet to wandb."""
    if wandb_run is None:
        return

    metrics: Dict[str, float] = {
        "train/loss": float(loss.detach().cpu()),
        "train/loss_avg": float(avg_loss),
        "train/loss_rec": float(loss_rec.detach().cpu()),
        "train/loss_commit": float(loss_commit.detach().cpu()),
        "train/loss_unif": float(loss_unif.detach().cpu()),
        "train/loss_eos": float(loss_eos.detach().cpu()),
        "train/loss_len": float(loss_len.detach().cpu()),
        "train/loss_quota": float(loss_quota.detach().cpu()),
        "train/quota_bar": float(quota_bar.detach().cpu()),
        "train/quota_lambda": float(quota_lambda.detach().cpu()),
        "train/grad_norm": float(grad_norm.detach().cpu()),
        "train/tau": float(tau),
        "train/lr": float(lr),
        "train/step_time_s": float(step_wall_time),
        "train/tokens_per_s": float(step_tokens / max(step_wall_time, 1e-9)),
        "train/tokens_this_step": float(step_tokens),
        "train/epoch": float(epoch + 1),
    }
    for idx, meta in enumerate(metas):
        if idx < len(avg_lens):
            metrics[f"train/type_len/{meta.name}"] = float(avg_lens[idx])
    if scaler is not None and scaler.is_enabled():
        metrics["amp/grad_scale"] = float(scaler.get_scale())
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        free_mem, total_mem = torch.cuda.mem_get_info()
        metrics.update(
            {
                "gpu/memory_allocated_mb": to_mb(alloc),  # 当前被张量实际占用的显存（MB）
                "gpu/memory_reserved_mb": to_mb(reserved),  # CUDA 缓存分配器向驱动保留的显存（MB）
                "gpu/memory_free_mb": to_mb(free_mem),  # 设备当前可用显存（MB，来自 mem_get_info）
                "gpu/memory_total_mb": to_mb(total_mem),  # 设备总显存（MB，来自 mem_get_info）
                "gpu/memory_utilization": float((total_mem - free_mem) / max(1, total_mem)),  # 显存使用率（0~1）
            }
        )
        metrics.update(stage_metrics)
    wandb_run.log(metrics, step=global_step)


def cuda_stage_begin(enabled: bool) -> Optional[Dict[str, float]]:
    if not enabled:
        return None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return {
        "t0": time.perf_counter(),
        "alloc0": float(torch.cuda.memory_allocated()),
        "reserved0": float(torch.cuda.memory_reserved()),
    }


def cuda_stage_end(stage_name: str, state: Optional[Dict[str, float]]) -> Dict[str, float]:
    if state is None:
        return {}
    torch.cuda.synchronize()
    alloc1 = float(torch.cuda.memory_allocated())
    reserved1 = float(torch.cuda.memory_reserved())
    return {
        f"perf/{stage_name}_time_ms": (time.perf_counter() - state["t0"]) * 1000.0,
        f"gpu/{stage_name}_alloc_start_mb": _to_mb(int(state["alloc0"])),
        f"gpu/{stage_name}_alloc_end_mb": _to_mb(int(alloc1)),
        f"gpu/{stage_name}_alloc_peak_mb": _to_mb(torch.cuda.max_memory_allocated()),
        f"gpu/{stage_name}_alloc_delta_mb": _to_mb(int(alloc1 - state["alloc0"])),
        f"gpu/{stage_name}_reserved_start_mb": _to_mb(int(state["reserved0"])),
        f"gpu/{stage_name}_reserved_end_mb": _to_mb(int(reserved1)),
        f"gpu/{stage_name}_reserved_peak_mb": _to_mb(torch.cuda.max_memory_reserved()),
        f"gpu/{stage_name}_reserved_delta_mb": _to_mb(int(reserved1 - state["reserved0"])),
    }

