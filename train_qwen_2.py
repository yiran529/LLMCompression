# TODO: aux loss
# TODO: more debug info
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
TAIL_MLP_HIDDEN_RATIO = 2
EPS = 1e-8
COMPRESSION_RATIO = 0.3
BETA_COMMIT = 0.5
LAMBDA_REC = 1.0
LAMBDA_COMMIT = 1.0
LAMBDA_UNIF = 2.0
LAMBDA_LEN = 0.1

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


def build_causal_mask_with_padding(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # attention_mask: [B, L], 1 for valid token, 0 for pad
    B, L = attention_mask.shape
    device = attention_mask.device
    causal = torch.tril(torch.ones((L, L), device=device, dtype=torch.bool))  # [L, L]
    valid = attention_mask.bool()
    allow = causal.unsqueeze(0)  # [1, L, L]
    allow = allow & valid.unsqueeze(1) & valid.unsqueeze(2)  # [B, L, L]
    mask = torch.zeros((B, L, L), device=device, dtype=dtype)
    mask = mask.masked_fill(~allow, VERY_NEG)
    return mask.unsqueeze(1)  # [B, 1, L, L]


def build_middle_mask(concept_valid: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # concept_valid: [B, Cmax], True for valid concept token, False for padding
    B, Cmax = concept_valid.shape
    device = concept_valid.device
    if Cmax == 0:
        return torch.zeros((B, 1, 0, 0), device=device, dtype=dtype)
    causal = torch.tril(torch.ones((Cmax, Cmax), device=device, dtype=torch.bool))  # [Cmax, Cmax]
    allow = causal.unsqueeze(0)  # [1, Cmax, Cmax]
    allow = allow & concept_valid.unsqueeze(1) & concept_valid.unsqueeze(2)  # [B, Cmax, Cmax]
    mask = torch.zeros((B, Cmax, Cmax), device=device, dtype=dtype)
    mask = mask.masked_fill(~allow, VERY_NEG)
    return mask.unsqueeze(1)  # [B, 1, Cmax, Cmax]


def build_deep_mask_batch(
    seq_lens: torch.Tensor,
    concept_lens: torch.Tensor,
    s_max: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # seq_lens: [A], concept_lens: [A], both for active samples at a given t
    # returns: [A, 1, Smax, Smax]
    A = int(seq_lens.numel())
    if A == 0 or s_max == 0:
        return torch.zeros((A, 1, s_max, s_max), device=device, dtype=dtype)

    allow = torch.zeros((A, s_max, s_max), device=device, dtype=torch.bool)
    for a in range(A):
        s = int(seq_lens[a].item())
        c = int(concept_lens[a].item())
        if s <= 0:
            continue
        if c > 0:
            # Concept -> Concept
            allow[a, :c, :c] = True
            # Tail -> Concept
            allow[a, c:s, :c] = True
            # Tail -> Tail (causal)
            tail_len = s - c
            if tail_len > 0:
                allow[a, c:s, c:s] = torch.tril(
                    torch.ones((tail_len, tail_len), device=device, dtype=torch.bool)
                )
        else:
            # Pure tail sequence, causal
            allow[a, :s, :s] = torch.tril(torch.ones((s, s), device=device, dtype=torch.bool))

    mask = torch.zeros((A, s_max, s_max), device=device, dtype=dtype)
    mask = mask.masked_fill(~allow, VERY_NEG)
    return mask.unsqueeze(1)  # [A, 1, Smax, Smax]


def get_tau(global_step: int, total_steps: int, tau_init: float, tau_min: float) -> float:
    ratio = global_step / max(1, total_steps)
    return tau_init + (tau_min - tau_init) * ratio


def commitment_loss(e_soft: torch.Tensor, e_hard: torch.Tensor, beta: float) -> torch.Tensor:
    loss1 = (e_soft.detach() - e_hard).pow(2).mean()
    loss2 = (e_soft - e_hard.detach()).pow(2).mean()
    return loss1 + beta * loss2


def usage_kl_to_uniform(hist: torch.Tensor) -> torch.Tensor:
    z = hist.numel()
    u = torch.full_like(hist, 1.0 / max(1, z))
    return torch.sum(hist * (torch.log(hist + EPS) - torch.log(u + EPS)))


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
        tail_mlp_hidden = self.hidden_size * TAIL_MLP_HIDDEN_RATIO
        self.tail_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, tail_mlp_hidden),
            nn.GELU(),
            nn.Linear(tail_mlp_hidden, self.hidden_size),
        ).to(dtype=self.embed_tokens.weight.dtype)

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

    def _pack_concepts(
        self,
        concept_embeds_all: torch.Tensor,
        is_compressed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # concept_embeds_all: [B, L, H]
        # is_compressed: [B, L]
        B, _, H = concept_embeds_all.shape
        device = concept_embeds_all.device
        concept_counts = is_compressed.long().sum(dim=1)  # [B]
        c_max = int(concept_counts.max().item()) if B > 0 else 0
        concept_padded = concept_embeds_all.new_zeros((B, c_max, H))  # [B, Cmax, H]
        concept_pos = torch.zeros((B, c_max), device=device, dtype=torch.long)  # [B, Cmax]
        concept_valid = torch.zeros((B, c_max), device=device, dtype=torch.bool)  # [B, Cmax]

        if c_max == 0:
            return concept_padded, concept_pos, concept_valid, concept_counts

        for b in range(B):
            pos_b = torch.nonzero(is_compressed[b], as_tuple=False).squeeze(-1)  # [Cb]
            cb = int(pos_b.numel())
            if cb == 0:
                continue
            concept_padded[b, :cb] = concept_embeds_all[b].index_select(0, pos_b)
            concept_pos[b, :cb] = pos_b
            concept_valid[b, :cb] = True
        return concept_padded, concept_pos, concept_valid, concept_counts

    def forward_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tau: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # input_ids: [B, L], attention_mask: [B, L]
        B, L = input_ids.shape
        device = input_ids.device
        valid_mask = attention_mask.bool()  # [B, L]
        lengths = attention_mask.sum(dim=1).to(torch.long)  # [B]

        if int(lengths.max().item()) < 2:
            zero = torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)
            return zero, {
                "num_tokens": float(lengths.sum().item()),
                "num_concepts": 0.0,
                "compress_ratio": 0.0,
                "num_targets": 0.0,
                "loss_rec": 0.0,
                "loss_commit": 0.0,
                "loss_unif": 0.0,
                "loss_len": 0.0,
            }

        # 1) Shallow on all normal tokens (batch parallel)
        x = self.embed_tokens(input_ids)  # [B, L, H]
        pos_ids_full = torch.arange(L, device=device, dtype=torch.long).unsqueeze(0).expand(B, L)  # [B, L]
        mask_full = build_causal_mask_with_padding(attention_mask, x.dtype)  # [B, 1, L, L]
        h_shallow = self.run_blocks(x, self.shallow_blocks, mask_full, pos_ids_full)  # [B, L, H]

        # 2) Concept head + ST Gumbel (batch parallel)
        concept_logits = self.concept_head(h_shallow)  # [B, L, K+1]
        z_soft = F.gumbel_softmax(concept_logits, tau=tau, hard=False, dim=-1)  # [B, L, K+1]
        chosen_idx = torch.argmax(z_soft, dim=-1)  # [B, L]
        is_compressed = chosen_idx.ne(self.null_id) & valid_mask  # [B, L]

        concept_weights_soft = z_soft[:, :, : self.concept_vocab_size]  # [B, L, K]
        e_soft = concept_weights_soft @ self.concept_embeddings.weight  # [B, L, H]
        chosen_safe = chosen_idx.clamp_max(self.concept_vocab_size - 1)  # [B, L]
        e_hard = self.concept_embeddings(chosen_safe)  # [B, L, H]
        e_hard = e_hard * is_compressed.unsqueeze(-1).to(e_hard.dtype)
        concept_embeds_all = e_hard + (e_soft - e_soft.detach())  # STE: e_hard + (e_soft - sg(e_soft))
        concept_embeds_all = concept_embeds_all * valid_mask.unsqueeze(-1).to(concept_embeds_all.dtype)  # [B, L, H]
        
        # 3) Pack non-NULL concepts only, then run middle once on padded batch
        concept_in, concept_pos, concept_valid, concept_counts = self._pack_concepts(concept_embeds_all, is_compressed)
        # concept_in: [B, Cmax, H], concept_pos: [B, Cmax], concept_valid: [B, Cmax], concept_counts: [B]
        c_max = concept_in.size(1)
        if c_max > 0:
            middle_mask = build_middle_mask(concept_valid, concept_in.dtype)  # [B, 1, Cmax, Cmax]
            concept_mid = self.run_blocks(concept_in, self.middle_blocks, middle_mask, concept_pos)  # [B, Cmax, H]
            concept_mid = concept_mid * concept_valid.unsqueeze(-1).to(concept_mid.dtype)  # [B, Cmax, H]
        else:
            concept_mid = concept_in

        # prefix statistics per time step
        compressed_cumsum = is_compressed.long().cumsum(dim=1)  # [B, L]
        pos_grid = torch.arange(L, device=device, dtype=torch.long).unsqueeze(0).expand(B, L)  # [B, L]
        comp_pos_grid = torch.where(is_compressed, pos_grid, torch.full_like(pos_grid, -1))  # [B, L]
        latest_comp = torch.cummax(comp_pos_grid, dim=1).values  # [B, L]

        # 4) Deep blocks: vectorize over all valid (b, t) pairs
        # We build one "deep sequence" for every training position t that has a target x[t+1].
        # Sequence layout follows v2: Z_t = [concept_prefix(<=t); tail(t)].
        T = L - 1
        t_grid = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0).expand(B, T)  # [B, T]
        active_bt = lengths.unsqueeze(1) > (t_grid + 1)  # [B, T], needs target x[t+1]
        active_b, active_t = torch.nonzero(active_bt, as_tuple=True)  # [N], [N]
        N = int(active_b.numel())

        # For each active (b, t):
        # c_n: number of compressed tokens in prefix [0..t]
        # tau_n: latest compressed normal-token position tau(t), -1 if none
        c_n = compressed_cumsum[active_b, active_t]  # [N]
        tau_n = latest_comp[active_b, active_t]  # [N]
        # tail(t) = h_shallow[tau(t)+1 : t], so tail_start = tau+1.
        tail_start_n = tau_n + 1  # [N]
        tail_len_n = active_t - tail_start_n + 1  # [N]
        # Packed deep length = concept prefix length + tail length.
        seq_lens = c_n + tail_len_n  # [N]
        last_index = seq_lens - 1  # [N]

        s_max = int(seq_lens.max().item())
        H = h_shallow.size(-1)
        s_idx = torch.arange(s_max, device=device, dtype=torch.long).unsqueeze(0)  # [1, Smax]
        valid_pos = s_idx < seq_lens.unsqueeze(1)  # [N, Smax]
        is_concept = s_idx < c_n.unsqueeze(1)  # [N, Smax]

        # Map packed tail slots back to original shallow-token positions.
        tail_src = tail_start_n.unsqueeze(1) + (s_idx - c_n.unsqueeze(1))  # [N, Smax]
        tail_src_safe = tail_src.clamp(0, L - 1)
        tail_vals = h_shallow[active_b.unsqueeze(1), tail_src_safe]  # [N, Smax, H]
        # Learnable tail refinement before entering deep blocks (residual form).
        tail_vals = tail_vals + self.tail_mlp(tail_vals)

        if c_max > 0:
            # Gather concept prefix features/positions for each active sample.
            concept_src = s_idx.clamp(max=c_max - 1)  # [1, Smax]
            concept_vals = concept_mid[active_b.unsqueeze(1), concept_src]  # [N, Smax, H]
            concept_pos_vals = concept_pos[active_b.unsqueeze(1), concept_src]  # [N, Smax]
        else:
            # No concept tokens in this batch; concept region stays zero and masked out by valid_pos.
            concept_vals = h_shallow.new_zeros((N, s_max, H))
            concept_pos_vals = torch.zeros((N, s_max), device=device, dtype=torch.long)

        # Stitch Z_t: concept prefix first, then tail; keep absolute positions for RoPE.
        deep_in = torch.where(is_concept.unsqueeze(-1), concept_vals, tail_vals)  # [N, Smax, H]
        deep_in = deep_in * valid_pos.unsqueeze(-1).to(deep_in.dtype)
        deep_pos = torch.where(is_concept, concept_pos_vals, tail_src_safe)  # [N, Smax]
        deep_pos = deep_pos.masked_fill(~valid_pos, 0)

        q_idx = torch.arange(s_max, device=device, dtype=torch.long).view(1, s_max, 1)  # [1, Smax, 1]
        k_idx = torch.arange(s_max, device=device, dtype=torch.long).view(1, 1, s_max)  # [1, 1, Smax]
        seq_v = seq_lens.view(N, 1, 1)
        c_v = c_n.view(N, 1, 1)
        valid_q = q_idx < seq_v
        valid_k = k_idx < seq_v
        tail_q = q_idx >= c_v
        tail_k = k_idx >= c_v
        causal_tail = k_idx <= q_idx
        # v2 deep attention rules (vectorized):
        # 1) Concept -> Concept: allowed           (covered by ~tail_k for concept keys)
        # 2) Tail -> Concept: allowed              (also covered by ~tail_k)
        # 3) Tail -> Tail: causal only             (tail_q & tail_k & causal_tail)
        # 4) Concept -> Tail: disallowed           (not matched by either branch)
        allow = valid_q & valid_k & ((~tail_k) | (tail_q & tail_k & causal_tail))  # [N, Smax, Smax]
        deep_mask = torch.zeros((N, s_max, s_max), device=device, dtype=deep_in.dtype)
        deep_mask = deep_mask.masked_fill(~allow, VERY_NEG).unsqueeze(1)  # [N, 1, Smax, Smax]

        deep_h = self.run_blocks(deep_in, self.deep_blocks, deep_mask, deep_pos)  # [N, Smax, H]
        deep_h = self.final_norm(deep_h)  # [N, Smax, H]
        # Supervise with next-token target at each active timestep t.
        last_h = deep_h[torch.arange(N, device=device), last_index]  # [N, H]
        logits = self.lm_head(last_h)  # [N, V]
        targets = input_ids[active_b, active_t + 1]  # [N]
        loss_sum = F.cross_entropy(logits.float(), targets, reduction="sum")
        valid_targets = N

        loss_rec = loss_sum / max(1, valid_targets)

        # Aux-1: commitment on compressed (non-NULL) positions only.
        if bool(is_compressed.any().item()):
            e_soft_valid = e_soft[is_compressed]
            e_hard_valid = e_hard[is_compressed]
            loss_commit = commitment_loss(e_soft_valid, e_hard_valid, beta=BETA_COMMIT)
        else:
            loss_commit = torch.zeros((), device=device, dtype=loss_rec.dtype)

        # Aux-2: encourage concept-id usage closer to uniform.
        concept_mass = concept_weights_soft * valid_mask.unsqueeze(-1).to(concept_weights_soft.dtype)
        z_hist = concept_mass.sum(dim=(0, 1))
        z_hist = z_hist / (z_hist.sum() + EPS)
        loss_unif = usage_kl_to_uniform(z_hist)

        # Aux-3: length penalty on soft compression count (differentiable).
        p_non_null = 1.0 - z_soft[:, :, self.null_id]
        soft_concept_counts = (p_non_null * valid_mask.to(p_non_null.dtype)).sum(dim=1)  # [B]
        budget = (lengths.float() * COMPRESSION_RATIO).clamp(min=1.0)
        loss_len = F.relu(soft_concept_counts - budget).mean()

        loss = (
            LAMBDA_REC * loss_rec
            + LAMBDA_COMMIT * loss_commit
            + LAMBDA_UNIF * loss_unif
            + LAMBDA_LEN * loss_len
        )
        total_tokens = float(lengths.sum().item())
        total_concepts = float(concept_counts.sum().item())
        stats = {
            "num_tokens": total_tokens,
            "num_concepts": total_concepts,
            "compress_ratio": total_concepts / max(1.0, total_tokens),
            "num_targets": float(valid_targets),
            "loss_rec": float(loss_rec.detach().item()),
            "loss_commit": float(loss_commit.detach().item()),
            "loss_unif": float(loss_unif.detach().item()),
            "loss_len": float(loss_len.detach().item()),
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
            "tail_mlp": model.tail_mlp.state_dict(),
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
        is_new_module = (
            name.startswith("concept_head.")
            or name.startswith("concept_embeddings.")
            or name.startswith("tail_mlp.")
        )
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
    running_rec = 0.0
    running_commit = 0.0
    running_unif = 0.0
    running_len = 0.0

    for epoch in range(EPOCHS):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            tau = get_tau(global_step, total_steps, TAU_INIT, TAU_MIN)

            loss, batch_stats = model.forward_batch(input_ids, attention_mask, tau=tau)
            if batch_stats["num_targets"] < 1:
                continue

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
            batch_ratio = batch_stats["compress_ratio"]
            running_ratio += float(batch_ratio)
            running_rec += float(batch_stats["loss_rec"])
            running_commit += float(batch_stats["loss_commit"])
            running_unif += float(batch_stats["loss_unif"])
            running_len += float(batch_stats["loss_len"])
            pbar.update(1)

            if global_step % LOG_STEPS == 0:
                avg_loss = running_loss / LOG_STEPS
                avg_ratio = running_ratio / LOG_STEPS
                avg_rec = running_rec / LOG_STEPS
                avg_commit = running_commit / LOG_STEPS
                avg_unif = running_unif / LOG_STEPS
                avg_len = running_len / LOG_STEPS
                lr_now = scheduler.get_last_lr()[0]
                logging.info(
                    f"[Epoch {epoch+1}/{EPOCHS}] Step {global_step}/{total_steps} | "
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
