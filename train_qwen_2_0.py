import math
import os
import random
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model


# =========================
# Logging
# =========================
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


# =========================
# Config (edit for your environment)
# =========================
BASE_DIR = "/root/lm_merge/qwen3_0.6b_z4096"
PARQUET_PATH = "/root/data/wiki_en_sentences_flat.parquet"
OUTPUT_DIR = "/root/lm_merge/train_runs/qwen_online_concept_v2"

MAX_SAMPLES = 100000
MAX_INPUT_TOKENS = 64
BATCH_SIZE = 8
GRAD_ACCUM = 2
EPOCHS = 1
LR = 2e-4
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
LOG_STEPS = 20
SAVE_STEPS = 500
SEED = 42

# Concept compression
CONCEPT_VOCAB_SIZE = 4096  # K
NULL_CONCEPT_ID = CONCEPT_VOCAB_SIZE  # K + 1th logit is <NULL>
TAU_INIT = 1.0
TAU_MIN = 0.5

# Layer split
SHALLOW_LAYERS = 8
MIDDLE_LAYERS = 8

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_CANDIDATES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

VERY_NEG = -1e4


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# Dataset
# =========================
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


def build_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)
    upper = torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)
    mask = mask.masked_fill(upper, VERY_NEG)
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, Q, K]


def build_deep_mask(num_concepts: int, tail_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    total = num_concepts + tail_len
    allow = torch.zeros((total, total), device=device, dtype=torch.bool)

    if num_concepts > 0:
        # Concept -> Concept: allowed
        allow[:num_concepts, :num_concepts] = True
        # Concept -> Tail: disallowed (keep False)
        # Tail -> Concept: allowed
        allow[num_concepts:, :num_concepts] = True

    # Tail -> Tail: causal
    tail_causal = torch.tril(torch.ones((tail_len, tail_len), device=device, dtype=torch.bool))
    allow[num_concepts:, num_concepts:] = tail_causal

    mask = torch.zeros((total, total), device=device, dtype=dtype)
    mask = mask.masked_fill(~allow, VERY_NEG)
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, Q, K]


def get_tau(global_step: int, total_steps: int, tau_init: float, tau_min: float) -> float:
    ratio = global_step / max(1, total_steps)
    return tau_init + (tau_min - tau_init) * ratio


def extract_base_causallm(peft_model: nn.Module) -> nn.Module:
    if hasattr(peft_model, "base_model") and hasattr(peft_model.base_model, "model"):
        base = peft_model.base_model.model
        if hasattr(base, "lm_head"):
            return base
    if hasattr(peft_model, "get_base_model"):
        base = peft_model.get_base_model()
        if hasattr(base, "lm_head"):
            return base
    if hasattr(peft_model, "lm_head"):
        return peft_model
    raise RuntimeError("Could not resolve base CausalLM from PEFT model.")


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


def call_decoder_layer(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    try:
        out = layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_attentions=False,
        )
    except TypeError:
        out = layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids)
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


class OnlineConceptCompressionModel(nn.Module):
    def __init__(
        self,
        peft_model: nn.Module,
        concept_vocab_size: int,
        shallow_layers: int,
        middle_layers: int,
    ):
        super().__init__()
        self.peft_model = peft_model
        self.base_causallm = extract_base_causallm(peft_model)

        if not hasattr(self.base_causallm, "model"):
            raise RuntimeError("Expected a decoder backbone at base_causallm.model")
        self.backbone = self.base_causallm.model
        if not hasattr(self.backbone, "layers"):
            raise RuntimeError("Expected decoder layers at base_causallm.model.layers")

        self.layers = self.backbone.layers
        self.embed_tokens = self.backbone.embed_tokens
        self.final_norm = getattr(self.backbone, "norm", nn.Identity())
        self.lm_head = self.base_causallm.lm_head

        self.hidden_size = self.embed_tokens.embedding_dim
        self.concept_vocab_size = concept_vocab_size
        self.null_id = concept_vocab_size

        self.concept_head = nn.Linear(self.hidden_size, concept_vocab_size + 1)
        self.concept_embeddings = nn.Embedding(concept_vocab_size, self.hidden_size)

        n_layers = len(self.layers)
        if shallow_layers + middle_layers >= n_layers:
            raise ValueError(
                f"Invalid split: shallow({shallow_layers}) + middle({middle_layers}) "
                f"must be < total layers ({n_layers})."
            )
        self.shallow_start = 0
        self.shallow_end = shallow_layers
        self.middle_end = shallow_layers + middle_layers

        self.shallow_blocks = self.layers[self.shallow_start:self.shallow_end]
        self.middle_blocks = self.layers[self.shallow_end:self.middle_end]
        self.deep_blocks = self.layers[self.middle_end:]

    def run_blocks(
        self,
        hidden_states: torch.Tensor,
        blocks: nn.ModuleList,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        h = hidden_states
        for block in blocks:
            h = call_decoder_layer(block, h, attention_mask, position_ids)
        return h

    def forward_single_sequence(self, input_ids: torch.Tensor, tau: float) -> Tuple[torch.Tensor, Dict[str, float]]:
        # input_ids: [L], no padding
        assert input_ids.dim() == 1, "input_ids must be 1D for single-sequence forward."
        device = input_ids.device
        L = int(input_ids.size(0))
        if L < 2:
            zero = torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)
            return zero, {"num_tokens": float(L), "num_concepts": 0.0, "compress_ratio": 0.0}

        x = self.embed_tokens(input_ids.unsqueeze(0))  # [1, L, H]
        pos_ids_full = torch.arange(L, device=device, dtype=torch.long).unsqueeze(0)
        mask_full = build_causal_mask(L, device, x.dtype)

        # 1) Shallow on all normal tokens
        h_shallow = self.run_blocks(x, self.shallow_blocks, mask_full, pos_ids_full)  # [1, L, H]
        h_seq = h_shallow[0]  # [L, H]

        # 2) Concept head + ST Gumbel
        concept_logits = self.concept_head(h_seq)  # [L, K+1]
        z_st = F.gumbel_softmax(concept_logits, tau=tau, hard=True, dim=-1)  # ST estimator
        chosen_idx = torch.argmax(z_st, dim=-1)  # [L]
        is_compressed = chosen_idx.ne(self.null_id)  # [L]

        concept_weights = z_st[:, : self.concept_vocab_size]  # [L, K]
        concept_embeds_all = concept_weights @ self.concept_embeddings.weight  # [L, H]

        compressed_positions = torch.nonzero(is_compressed, as_tuple=False).squeeze(-1)  # [C]
        C = int(compressed_positions.numel())

        # 3) Middle blocks ONLY on concept tokens
        if C > 0:
            concept_in = concept_embeds_all.index_select(0, compressed_positions).unsqueeze(0)  # [1, C, H]
            concept_pos_ids = compressed_positions.unsqueeze(0)
            concept_mask = build_causal_mask(C, device, concept_in.dtype)
            concept_mid = self.run_blocks(concept_in, self.middle_blocks, concept_mask, concept_pos_ids)[0]  # [C, H]
            compressed_cumsum = is_compressed.to(torch.long).cumsum(dim=0)
        else:
            concept_mid = h_seq.new_zeros((0, h_seq.size(-1)))
            compressed_cumsum = torch.zeros((L,), device=device, dtype=torch.long)

        # Track tau(t): latest compressed normal token index <= t
        latest_comp = []
        cur = -1
        for i in range(L):
            if bool(is_compressed[i].item()):
                cur = i
            latest_comp.append(cur)
        latest_comp = torch.tensor(latest_comp, device=device, dtype=torch.long)

        # 4) For each prediction step t, run Deep on [concept_prefix(t); tail(t)]
        loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        steps = 0
        for t in range(L - 1):
            tau_t = int(latest_comp[t].item())
            tail_start = tau_t + 1

            tail = h_seq[tail_start : t + 1]  # [tail_len, H], contiguous dynamic tail
            tail_len = int(tail.size(0))
            c_count = int(compressed_cumsum[t].item())

            if c_count > 0:
                concept_prefix = concept_mid[:c_count]  # [c_count, H]
                deep_in = torch.cat([concept_prefix, tail], dim=0).unsqueeze(0)  # [1, S, H]
                concept_pos = compressed_positions[:c_count]
                tail_pos = torch.arange(tail_start, t + 1, device=device, dtype=torch.long)
                pos_ids = torch.cat([concept_pos, tail_pos], dim=0).unsqueeze(0)  # [1, S]
                deep_mask = build_deep_mask(c_count, tail_len, device, deep_in.dtype)
            else:
                deep_in = tail.unsqueeze(0)  # [1, tail_len, H]
                pos_ids = torch.arange(tail_start, t + 1, device=device, dtype=torch.long).unsqueeze(0)
                deep_mask = build_causal_mask(tail_len, device, deep_in.dtype)

            deep_h = self.run_blocks(deep_in, self.deep_blocks, deep_mask, pos_ids)
            deep_h = self.final_norm(deep_h)
            logits = self.lm_head(deep_h[:, -1, :])  # predict x[t+1]
            target = input_ids[t + 1].view(1)
            loss_t = F.cross_entropy(logits.float(), target)
            loss_sum = loss_sum + loss_t
            steps += 1

        loss = loss_sum / max(1, steps)
        stats = {
            "num_tokens": float(L),
            "num_concepts": float(C),
            "compress_ratio": float(C) / float(max(1, L)),
        }
        return loss, stats


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
            "concept_vocab_size": model.concept_vocab_size,
            "null_id": model.null_id,
            "shallow_end": model.shallow_end,
            "middle_end": model.middle_end,
        },
        os.path.join(ckpt_dir, "concept_modules.pt"),
    )
    logging.info(f"[SAVE] checkpoint saved to: {ckpt_dir}")


def train() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file = setup_logging(OUTPUT_DIR)
    set_seed(SEED)
    logging.info(f"[INFO] Logging to: {log_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"[INFO] Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_DIR,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            attn_implementation="eager",
        )
    except TypeError:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_DIR,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    base_model.to(device)

    target_modules = detect_lora_targets(base_model, LORA_TARGET_CANDIDATES)
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=target_modules,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, lora_cfg)
    peft_model.to(device)

    model = OnlineConceptCompressionModel(
        peft_model=peft_model,
        concept_vocab_size=CONCEPT_VOCAB_SIZE,
        shallow_layers=SHALLOW_LAYERS,
        middle_layers=MIDDLE_LAYERS,
    ).to(device)

    # Hard constraint: train only (new modules + LoRA params).
    for name, param in model.named_parameters():
        is_new_module = name.startswith("concept_head.") or name.startswith("concept_embeddings.")
        is_lora = "lora_" in name
        param.requires_grad = bool(is_new_module or is_lora)

    dataset = ParquetSentenceDataset(PARQUET_PATH, max_samples=MAX_SAMPLES)
    collate_fn = Collator(tokenizer=tokenizer, max_len=MAX_INPUT_TOKENS)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
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

    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = math.ceil(len(dataloader) / GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    global_step = 0
    micro_step = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(total=total_steps, desc="train_qwen_2", unit="step")
    running_loss = 0.0
    running_ratio = 0.0

    for epoch in range(EPOCHS):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            B = input_ids.size(0)
            tau = get_tau(global_step, total_steps, TAU_INIT, TAU_MIN)

            seq_losses = []
            token_count = 0.0
            concept_count = 0.0
            for b in range(B):
                seq_len = int(attention_mask[b].sum().item())
                if seq_len < 2:
                    continue
                seq = input_ids[b, :seq_len]
                seq_loss, seq_stats = model.forward_single_sequence(seq, tau=tau)
                seq_losses.append(seq_loss)
                token_count += seq_stats["num_tokens"]
                concept_count += seq_stats["num_concepts"]

            if not seq_losses:
                continue

            loss = torch.stack(seq_losses).mean()
            loss_scaled = loss / GRAD_ACCUM
            loss_scaled.backward()
            micro_step += 1

            if micro_step % GRAD_ACCUM != 0:
                continue

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            running_loss += float(loss.detach().cpu().item())
            batch_ratio = concept_count / max(1.0, token_count)
            running_ratio += float(batch_ratio)
            pbar.update(1)

            if global_step % LOG_STEPS == 0:
                avg_loss = running_loss / LOG_STEPS
                avg_ratio = running_ratio / LOG_STEPS
                lr_now = scheduler.get_last_lr()[0]
                logging.info(
                    f"[Epoch {epoch+1}/{EPOCHS}] Step {global_step}/{total_steps} | "
                    f"Loss {avg_loss:.4f} | CompressRatio {avg_ratio:.4f} | "
                    f"Tau {tau:.4f} | LR {lr_now:.2e}"
                )
                running_loss = 0.0
                running_ratio = 0.0

            if global_step % SAVE_STEPS == 0:
                save_checkpoint(model, tokenizer, global_step, OUTPUT_DIR)

            if global_step >= total_steps:
                break

        if global_step >= total_steps:
            break

    pbar.close()
    save_checkpoint(model, tokenizer, global_step, OUTPUT_DIR)
    logging.info("[DONE] Training finished.")


if __name__ == "__main__":
    train()
