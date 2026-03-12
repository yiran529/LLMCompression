# -*- coding: utf-8 -*-
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

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


def get_planner_mix_greedy_ratio(
    *,
    global_step: int,
    total_steps: int,
    ratio_min: float,
    ratio_max: float,
) -> float:
    """linearly increase planner greedy-row ratio from `ratio_min` to `ratio_max`."""
    lo = min(float(ratio_min), float(ratio_max))
    hi = max(float(ratio_min), float(ratio_max))
    progress = min(1.0, max(0.0, float(global_step) / max(1.0, float(total_steps))))
    return lo + (hi - lo) * progress


def get_stage2_tf_mask_ratio(
    *,
    global_step: int,
    total_steps: int,
    ratio_max: float,
    ratio_min: float,
) -> float:
    """piecewise schedule for stage-2 TF corruption ratio."""
    lo = min(float(ratio_max), float(ratio_min))
    hi = max(float(ratio_max), float(ratio_min))
    progress = min(1.0, max(0.0, float(global_step) / max(1.0, float(total_steps))))

    if progress <= 0.30:
        return lo
    if progress >= 0.80:
        return hi
    local = (progress - 0.30) / 0.50
    return lo + (hi - lo) * local


def apply_stage2_tf_token_masking(
    *,
    decoder_in: torch.Tensor,
    decoder_mask: torch.Tensor,
    global_step: int,
    total_steps: int,
    enabled: bool,
    ratio_max: float,
    ratio_min: float,
    random_token_upper_bound: int,
) -> Tuple[torch.Tensor, float, float]:
    """apply stage-2 decoder corruption with random-token replacement."""
    corrupted_decoder_in = decoder_in
    if not enabled:
        return corrupted_decoder_in, 0.0, 0.0

    target_ratio = get_stage2_tf_mask_ratio(
        global_step=global_step,
        total_steps=total_steps,
        ratio_max=ratio_max,
        ratio_min=ratio_min,
    )
    assert 0.0 <= target_ratio <= 1.0, "target_ratio must be in [0, 1]"

    corrupted_decoder_in = decoder_in.clone()
    candidate_mask = decoder_mask.bool().clone()
    if candidate_mask.size(1) > 0:
        candidate_mask[:, 0] = False  # keep BOS unmasked

    sampled = torch.rand(decoder_in.shape, device=decoder_in.device) < target_ratio
    corrupt_positions = candidate_mask & sampled

    candidate_count = int(candidate_mask.sum().item())
    masked_count = int(corrupt_positions.sum().item())
    if masked_count <= 0:
        return corrupted_decoder_in, target_ratio, 0.0

    upper = max(1, int(random_token_upper_bound))
    random_tokens = torch.randint(
        low=0,
        high=upper,
        size=decoder_in.shape,
        device=decoder_in.device,
        dtype=decoder_in.dtype,
    )
    corrupted_decoder_in[corrupt_positions] = random_tokens[corrupt_positions]

    applied_ratio = float(masked_count) / float(candidate_count) if candidate_count > 0 else 0.0
    return corrupted_decoder_in, target_ratio, applied_ratio

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
    meta: TokenMeta,
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

    meta_dump = {
        "type_id": meta.type_id_concept,
        "eos_id": meta.concept_eos_id,
        "concept_ids": meta.concept_ids.tolist(),
        "max_steps": meta.max_concept_steps,
        "target_ratio": meta.target_concept_ratio,
        "original_vocab_size": meta.original_vocab_size,
    }

    token_embed_new = model.get_new_token_embed_weight().detach().cpu()

    torch.save(
        {
            "output_head_base": model.output_head_base.state_dict(),
            "output_head_new": model.output_head_new.state_dict() if model.output_head_new is not None else None,
            "type_embed": model.type_embed.state_dict(),
            "token_embed_new": token_embed_new,
            "concept_meta": meta_dump,
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
        "eval_steps": EVAL_STEPS,
        "eval_num_samples": EVAL_NUM_SAMPLES,
        "eval_max_new_tokens": EVAL_MAX_NEW_TOKENS,
        "eval_planner_tau": EVAL_PLANNER_TAU,
        "eval_planner_sampling_mode": EVAL_PLANNER_SAMPLING_MODE,
        "eval_planner_mix_greedy_ratio": EVAL_PLANNER_MIX_GREEDY_RATIO,
        "enable_stage2_tf_masking": ENABLE_STAGE2_TF_MASKING,
        "stage2_tf_masking_max_ratio": STAGE2_TF_MASKING_MAX_RATIO,
        "stage2_tf_masking_min_ratio": STAGE2_TF_MASKING_MIN_RATIO,
        "seed": SEED,
        "model_dtype": MODEL_DTYPE,
        "attention_impl": ATTENTION_IMPL,
        "use_compile": USE_COMPILE,
        "compile_mode": COMPILE_MODE,
        "fp32_trainable": FP32_TRAINABLE,
        "tau_init": TAU_INIT,
        "tau_min": TAU_MIN,
        "train_planner_sampling_mode": TRAIN_PLANNER_SAMPLING_MODE,
        "train_planner_mix_greedy_ratio_min": TRAIN_PLANNER_MIX_GREEDY_RATIO_MIN,
        "train_planner_mix_greedy_ratio_max": TRAIN_PLANNER_MIX_GREEDY_RATIO_MAX,
        "min_concept_steps": MIN_CONCEPT_STEPS,
        "allow_planner_base_tokens": ALLOW_PLANNER_BASE_TOKENS,
        "lambda_rec": LAMBDA_REC,
        "lambda_commit": LAMBDA_COMMIT,
        "lambda_unif": LAMBDA_UNIF,
        "lambda_eos": LAMBDA_EOS,
        "lambda_repeat": LAMBDA_REPEAT,
        "lambda_len": LAMBDA_LEN,
        "planner_repeat_last_k": PLANNER_REPEAT_LAST_K,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lora_target_modules": LORA_TARGET_MODULES,
        "lora_modules_to_save": LORA_MODULES_TO_SAVE,
        "resume_enabled": RESUME_ENABLED,
        "resume_checkpoint_dir": RESUME_CHECKPOINT_DIR,
        "concept_config": CONCEPT_CONFIG.__dict__,
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
        "mode": "online", #! 鏆傛椂鍏堝啓姝籵nline锛屽悗缁彲浠ラ€氳繃鐜鍙橀噺WANDB_MODE鍒囨崲鍒皁ffline鎴杁isabled
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


def compute_concept_diversity_metrics(
    *,
    planner_out: PlannerOutput,
    meta: TokenMeta,
) -> Dict[str, float]:
    """Compute compact concept-token diversity metrics (excluding EOS tokens)."""
    # Planner output includes padded/finished positions and EOS markers.
    # Only valid non-EOS concept tokens are counted for diversity statistics.
    concept_token_ids = planner_out.concept.token_ids
    concept_valid_mask = planner_out.concept.valid_mask.bool()
    concept_effective_mask = concept_valid_mask & concept_token_ids.ne(meta.concept_eos_id)

    concept_seq_diversities: List[float] = []
    concept_seq_unique_counts: List[float] = []
    concept_effective_lengths = concept_effective_mask.sum(dim=1)
    for idx in range(concept_token_ids.size(0)):
        seq_len = int(concept_effective_lengths[idx].item())
        if seq_len <= 0:
            concept_seq_diversities.append(0.0)
            concept_seq_unique_counts.append(0.0)
            continue
        seq_tokens = concept_token_ids[idx][concept_effective_mask[idx]]
        seq_unique = int(torch.unique(seq_tokens).numel())
        concept_seq_unique_counts.append(float(seq_unique))
        concept_seq_diversities.append(float(seq_unique) / float(seq_len))

    if concept_seq_diversities:
        seq_div_mean = sum(concept_seq_diversities) / float(len(concept_seq_diversities))
        seq_unique_mean = sum(concept_seq_unique_counts) / float(len(concept_seq_unique_counts))
    else:
        seq_div_mean = 0.0
        seq_unique_mean = 0.0

    batch_tokens = concept_token_ids[concept_effective_mask]
    batch_token_count = int(batch_tokens.numel())
    if batch_token_count > 0:
        batch_unique_count = int(torch.unique(batch_tokens).numel())
        batch_div_ratio = float(batch_unique_count) / float(batch_token_count)
    else:
        batch_unique_count = 0
        batch_div_ratio = 0.0

    return {
        "train/concept_diversity_seq_mean": seq_div_mean,
        "train/concept_seq_unique_count_mean": seq_unique_mean,
        "train/concept_diversity_batch_ratio": batch_div_ratio,
    }


def log_wandb_step_metrics(
    *,
    wandb_run,
    global_step: int,
    epoch: int,
    meta: TokenMeta,
    avg_len: float,
    avg_loss: float,
    loss: torch.Tensor,
    loss_rec: torch.Tensor,
    loss_commit: torch.Tensor,
    loss_unif: torch.Tensor,
    loss_eos: torch.Tensor,
    loss_len: torch.Tensor,
    grad_norm: torch.Tensor,
    tau: float,
    lr: float,
    step_wall_time: float,
    step_tokens: int,
    scaler,
    stage_metrics: Dict[str, float],
    extra_metrics: Optional[Dict[str, float]] = None,
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
        "train/grad_norm": float(grad_norm.detach().cpu()),
        "train/tau": float(tau),
        "train/lr": float(lr),
        "train/step_time_s": float(step_wall_time),
        "train/tokens_per_s": float(step_tokens / max(step_wall_time, 1e-9)),
        "train/tokens_this_step": float(step_tokens),
        "train/epoch": float(epoch + 1),
    }
    metrics["train/concept_len"] = float(avg_len)
    if extra_metrics:
        metrics.update({k: float(v) for k, v in extra_metrics.items()})
    if scaler is not None and scaler.is_enabled():
        metrics["amp/grad_scale"] = float(scaler.get_scale())
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        free_mem, total_mem = torch.cuda.mem_get_info()
        metrics.update(
            {
                "gpu/memory_allocated_mb": to_mb(alloc),
                "gpu/memory_reserved_mb": to_mb(reserved),
                "gpu/memory_free_mb": to_mb(free_mem),
                "gpu/memory_total_mb": to_mb(total_mem),
                "gpu/memory_utilization": float((total_mem - free_mem) / max(1, total_mem)),
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

# ============================================================
# Eval utilities
# ============================================================

def _trim_to_first_eos(token_ids: List[int], eos_id: int) -> List[int]:
    if eos_id in token_ids:
        return token_ids[: token_ids.index(eos_id)]
    return token_ids


def _format_concepts_for_sample(
    tokenizer: AutoTokenizer,
    meta: TokenMeta,
    planner_out: PlannerOutput,
    sample_idx: int,
) -> List[str]:
    type_out = planner_out.concept
    valid = type_out.valid_mask[sample_idx].to(torch.bool)
    ids = type_out.token_ids[sample_idx][valid].tolist()
    toks = tokenizer.convert_ids_to_tokens(ids)
    tok_text = " ".join(toks) if toks else "(empty)"
    return [f"concept: ids={ids} tokens={tok_text}"]


def run_periodic_eval(
    *,
    model: SharedBackboneUnifiedHead,
    tokenizer: AutoTokenizer,
    meta: TokenMeta,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    global_step: int,
    wandb_run,
    model_dtype: torch.dtype = torch.float32,
    use_amp: bool = False,
    planner_sampling_mode: str = "greedy",
    planner_mix_greedy_ratio: float = 0.0,
) -> None:
    sample_count = min(int(EVAL_NUM_SAMPLES), int(input_ids.size(0)))
    if sample_count <= 0:
        return

    # Save training state and switch to eval mode
    was_training = model.training
    model.eval()

    eval_input_ids = input_ids[:sample_count]
    eval_attention_mask = attention_mask[:sample_count]
    
    # Run inference without gradients and with proper autocast
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=model_dtype, enabled=use_amp):
            infer_out = run_executor_inference(
                model,
                input_ids=eval_input_ids,
                attention_mask=eval_attention_mask,
                meta=meta,
                device=device,
                max_new_tokens=EVAL_MAX_NEW_TOKENS,
                planner_tau=EVAL_PLANNER_TAU,
                min_concept_steps=MIN_CONCEPT_STEPS,
                planner_sampling_mode=planner_sampling_mode,
                planner_mix_greedy_ratio=planner_mix_greedy_ratio,
            )

    # Restore training state
    if was_training:
        model.train()

    rows = []
    for i in range(sample_count):
        src_len = int(eval_attention_mask[i].sum().item())
        src_ids = eval_input_ids[i, :src_len].tolist()
        src_text = tokenizer.decode(src_ids, skip_special_tokens=EVAL_SKIP_SPECIAL_TOKENS)

        gen_len = int(infer_out.lengths[i].item())
        gen_ids = infer_out.generated_ids[i, :gen_len].tolist()
        gen_ids = _trim_to_first_eos(gen_ids, meta.eos_id)
        out_text = tokenizer.decode(gen_ids, skip_special_tokens=EVAL_SKIP_SPECIAL_TOKENS)

        concept_lines = _format_concepts_for_sample(
            tokenizer=tokenizer,
            meta=meta,
            planner_out=infer_out.planner_out,
            sample_idx=i,
        )
        concept_text = "\n".join(concept_lines)

        logging.info("=" * 100)
        logging.info(f"[Eval Step {global_step}] Sample {i}")
        logging.info(f"Input: {src_text}")
        logging.info("Concepts:")
        for line in concept_lines:
            logging.info(f"  {line}")
        logging.info(f"Output: {out_text}")

        rows.append([int(global_step), int(i), src_text, concept_text, out_text])

    if wandb_run is not None:
        try:
            import wandb  # type: ignore
            table = wandb.Table(
                columns=["step", "sample_idx", "input", "concept_sequence", "output"],
                data=rows,
            )
            wandb_run.log(
                {
                    "eval/samples": table,
                    "eval/num_samples": float(sample_count),
                },
                step=global_step,
            )
        except Exception as e:
            logging.warning(f"[WARN] failed to log eval samples to wandb: {e}")
