# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


from src.config.train_config import BETA_COMMIT, EPS, TYPE_ID_CONCEPT, TYPE_ID_TEXT, ConceptConfig

def gumbel_softmax_sample(logits: torch.Tensor, tau: float, hard: bool = False) -> torch.Tensor:
    """sample relaxed/straight-through categorical vectors from logits."""
    u = torch.empty_like(logits).uniform_(1e-8, 1 - 1e-8)
    g = -torch.log(-torch.log(u + 1e-10) + 1e-10)
    y = (logits + g) / max(tau, 1e-4)
    y = y - y.max(dim=-1, keepdim=True).values
    y = torch.clamp(y, min=-50, max=50)
    y_soft = F.softmax(y, dim=-1)
    y_soft = torch.nan_to_num(y_soft, nan=0.0, posinf=0.0, neginf=0.0)
    y_soft = y_soft / (y_soft.sum(dim=-1, keepdim=True) + 1e-8)
    if not hard:
        return y_soft
    idx = y_soft.argmax(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(y_soft).scatter_(-1, idx, 1.0)
    return y_hard - y_soft.detach() + y_soft


def _select_planner_tokens(
    *,
    masked_logits: torch.Tensor,
    tau: float,
    sampling_mode: str,
    mix_greedy_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """select planner token distributions and hard token IDs under gumbel/greedy/mix modes."""
    if sampling_mode == "gumbel":
        probs = gumbel_softmax_sample(masked_logits, tau=tau, hard=True)  # [B, V]
        sampled_ids = probs.argmax(dim=-1)  # [B]
        return probs, sampled_ids

    probs_greedy = F.softmax(masked_logits, dim=-1)
    probs_greedy = torch.nan_to_num(probs_greedy, nan=0.0, posinf=0.0, neginf=0.0)
    probs_greedy = probs_greedy / (probs_greedy.sum(dim=-1, keepdim=True) + 1e-8)
    sampled_greedy = masked_logits.argmax(dim=-1)  # [B]

    if sampling_mode == "greedy":
        return probs_greedy, sampled_greedy

    if sampling_mode == "mix":
        ratio = min(1.0, max(0.0, float(mix_greedy_ratio)))
        probs_gumbel = gumbel_softmax_sample(masked_logits, tau=tau, hard=True)  # [B, V]
        sampled_gumbel = probs_gumbel.argmax(dim=-1)  # [B]
        use_greedy = torch.rand((masked_logits.size(0),), device=masked_logits.device) < ratio
        probs = torch.where(use_greedy.unsqueeze(1), probs_greedy, probs_gumbel)
        sampled_ids = torch.where(use_greedy, sampled_greedy, sampled_gumbel)
        return probs, sampled_ids

    raise ValueError(f"Unsupported sampling_mode: {sampling_mode}. Use 'gumbel', 'greedy', or 'mix'.")


def commitment_loss(e_soft: torch.Tensor, e_hard: torch.Tensor, beta: float) -> torch.Tensor:
    """compute VQ-style commitment loss between soft and hard embeddings."""
    loss1 = (e_soft.detach() - e_hard).pow(2).mean()
    loss2 = (e_soft - e_hard.detach()).pow(2).mean()
    return loss1 + beta * loss2

def usage_kl_to_uniform(hist: torch.Tensor) -> torch.Tensor:
    """KL(hist || uniform) to discourage concept collapse."""
    z = hist.numel()
    u = torch.full_like(hist, 1.0 / max(1, z))
    return torch.sum(hist * (torch.log(hist + EPS) - torch.log(u + EPS)))

@dataclass
class ConceptMeta:
    type_id: int
    eos_id: int
    concept_ids: torch.Tensor
    concept_ids_with_eos: torch.Tensor
    max_steps: int
    target_ratio: float

class ConceptMaskCache:
    def __init__(
        self,
        meta: ConceptMeta,
        vocab_size: int,
        base_vocab_size: int,
        blocked_for_executor: List[int],
        device: str,
        allow_base_tokens: bool = True,
    ):
        """build planner mask and executor output block mask for single concept type."""
        very_neg = -1e4
        self.allowed_logits_bias = torch.full((vocab_size,), very_neg, device=device, dtype=torch.float32)
        if base_vocab_size > 0 and allow_base_tokens:
            self.allowed_logits_bias[:base_vocab_size] = 0.0
        self.allowed_logits_bias[meta.concept_ids_with_eos] = 0.0

        # Executor 端输出屏蔽表：True 的位置表示“禁止生成”。
        # 这里通常屏蔽所有 planner 专用 token（<PLAN>/<EOS_CONCEPT>/<C_*>)，
        # 避免解码文本时又吐出概念符号。
        self.executor_block_bool = torch.zeros(vocab_size, device=device, dtype=torch.bool)
        if blocked_for_executor:
            ids = torch.tensor(sorted(set(blocked_for_executor)), device=device, dtype=torch.long)
            self.executor_block_bool[ids] = True

class SharedBackboneUnifiedHead(nn.Module):
    def __init__(
        self,
        base_model: AutoModelForCausalLM,
        num_type_embeddings: int,
        frozen_output_head_prefix_rows: int = 0,
    ):
        """wrap one shared backbone with one unified output head."""
        super().__init__()
        backbone_owner = base_model
        if hasattr(backbone_owner, "get_base_model"):
            backbone_owner = backbone_owner.get_base_model()
        if not hasattr(backbone_owner, "model"):
            raise RuntimeError("Expected a causal LM with `.model` backbone (Qwen/LLaMA-style).")
        self.base_model = base_model
        backbone = backbone_owner.model
        if hasattr(backbone, "lm_head") and hasattr(backbone, "model"):
            backbone = backbone.model
        self.backbone = backbone
        input_embed = base_model.get_input_embeddings()
        vocab_size = input_embed.num_embeddings
        hidden_size = input_embed.embedding_dim
        self.hidden_size = hidden_size

        out_embed = base_model.get_output_embeddings()
        # Step 1) Resolve how many vocab rows are "base" (frozen) vs "new" (trainable).
        if frozen_output_head_prefix_rows <= 0:
            base_vocab_size = vocab_size
        else:
            base_vocab_size = frozen_output_head_prefix_rows
        if base_vocab_size > vocab_size:
            raise ValueError(
                f"frozen_output_head_prefix_rows must be <= vocab_size, got {base_vocab_size} > {vocab_size}"
            )
        new_rows = vocab_size - base_vocab_size

        # Step 2) Build split token embeddings: frozen base rows + trainable new rows.
        self.token_embed_base = nn.Embedding(base_vocab_size, hidden_size)
        self.token_embed_new = nn.Embedding(new_rows, hidden_size) if new_rows > 0 else None
        token_dtype = input_embed.weight.dtype
        self.token_embed_base.to(dtype=token_dtype)
        if self.token_embed_new is not None:
            self.token_embed_new.to(dtype=token_dtype)

        # Step 3) Initialize token embeddings from model input embeddings.
        src_input_weight = input_embed.weight.data
        self.token_embed_base.weight.data.copy_(src_input_weight[:base_vocab_size])
        self.token_embed_base.weight.requires_grad = False
        if self.token_embed_new is not None:
            self.token_embed_new.weight.data.copy_(src_input_weight[base_vocab_size:])

        # Step 4) Build split output head: frozen base rows + trainable new rows.
        self.output_head_base = nn.Linear(hidden_size, base_vocab_size, bias=False)
        self.output_head_new = nn.Linear(hidden_size, new_rows, bias=False) if new_rows > 0 else None
        self.base_vocab_size = base_vocab_size
        self.vocab_size = vocab_size
        self.type_embed = nn.Embedding(num_type_embeddings, hidden_size)
        # Keep type embeddings in the same dtype as token embeddings (e.g. fp16 on GPU).
        self.type_embed.to(dtype=token_dtype)
        # Step 5) Keep output heads in the same dtype as token embeddings for speed.
        head_dtype = token_dtype
        self.output_head_base.to(dtype=head_dtype)
        if self.output_head_new is not None:
            self.output_head_new.to(dtype=head_dtype)

        # Step 6) Initialize head weights from model output embeddings when available,
        # falling back to input embeddings for compatibility.
        if out_embed is not None and out_embed.weight.shape[0] == vocab_size:
            src_weight = out_embed.weight.data
        else:
            src_weight = src_input_weight
        self.output_head_base.weight.data.copy_(src_weight[:base_vocab_size])
        if self.output_head_new is not None:
            self.output_head_new.weight.data.copy_(src_weight[base_vocab_size:])

        nn.init.zeros_(self.type_embed.weight)
        # Freeze base output head rows.
        for p in self.output_head_base.parameters():
            p.requires_grad = False

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """embed token ids with frozen base rows + trainable new rows."""
        if self.token_embed_new is None:
            return self.token_embed_base(token_ids)
        flat_ids = token_ids.reshape(-1)
        out = torch.empty(
            (flat_ids.numel(), self.hidden_size),
            device=flat_ids.device,
            dtype=self.token_embed_base.weight.dtype,
        )
        is_base = flat_ids < self.base_vocab_size
        if torch.any(is_base):
            out[is_base] = self.token_embed_base(flat_ids[is_base]).to(dtype=out.dtype)
        if torch.any(~is_base):
            out[~is_base] = self.token_embed_new(flat_ids[~is_base] - self.base_vocab_size).to(
                dtype=out.dtype
            )
        return out.view(*token_ids.shape, self.hidden_size)

    def get_new_token_embed_weight(self) -> torch.Tensor:
        """return trainable new-token embedding rows (possibly empty)."""
        if self.token_embed_new is None:
            return self.token_embed_base.weight.new_empty((0, self.hidden_size))
        return self.token_embed_new.weight

    def forward_head(self, hidden: torch.Tensor) -> torch.Tensor:
        """compute logits for full vocab with split output head."""
        logits_base = self.output_head_base(hidden)
        if self.output_head_new is None:
            return logits_base
        logits_new = self.output_head_new(hidden)
        return torch.cat([logits_base, logits_new], dim=-1)

    def embed_with_type(self, token_ids: torch.Tensor, type_ids: torch.Tensor) -> torch.Tensor:
        """compose token embeddings with additive type embeddings."""
        return self.embed_tokens(token_ids) + self.type_embed(type_ids)

    def forward_backbone(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        past_key_values=None,
        use_cache: bool = False,
    ):
        """forward pass through shared transformer blocks only."""
        return self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )

@dataclass
class PlannerTypeOutput:
    token_ids: torch.Tensor      # [B, S]
    valid_mask: torch.Tensor     # [B, S]
    actual_lengths: torch.Tensor # [B]
    st_embeds: Optional[torch.Tensor] = None  # [B, S, H], straight-through concept embeddings

@dataclass
class PlannerOutput:
    concept: PlannerTypeOutput
    quota_mass_sum: torch.Tensor
    quota_count: torch.Tensor

class PlannerQuotaController:
    def __init__(self, *, tau: float, eta: float, lambda_init: float, lambda_max: float, device: str):
        self.tau = float(tau)
        self.eta = float(eta)
        self.lambda_max = float(lambda_max)
        self.lambda_value = torch.tensor(lambda_init, device=device, dtype=torch.float32)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"lambda_value": self.lambda_value.detach().cpu()}

    def load_state_dict(self, state: Dict[str, torch.Tensor], device: str) -> None:
        if "lambda_value" in state:
            self.lambda_value = state["lambda_value"].to(device=device, dtype=torch.float32)

    def compute_loss(self, quota_mass_sum: torch.Tensor, quota_count: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if quota_count.item() < 0.5:
            zero = torch.zeros((), device=quota_mass_sum.device, dtype=torch.float32)
            return zero, zero
        bar_m = quota_mass_sum / quota_count
        loss_quota = self.lambda_value.detach() * F.relu(bar_m - self.tau)
        with torch.no_grad():
            new_lambda = self.lambda_value + self.eta * (bar_m.detach() - self.tau)
            self.lambda_value = torch.clamp(new_lambda, min=0.0, max=self.lambda_max)
        return loss_quota, bar_m

def compute_planner_quota_loss(
    planner_out: PlannerOutput,
    quota_controller: PlannerQuotaController,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if quota_controller is None:
        zero = torch.zeros((), device=planner_out.quota_mass_sum.device, dtype=torch.float32)
        return zero, zero
    return quota_controller.compute_loss(planner_out.quota_mass_sum, planner_out.quota_count)

def build_concept_special_tokens(cfg: ConceptConfig) -> List[str]:
    """define planner-side special tokens for single concept vocabulary."""
    special_tokens: List[str] = ["<PLAN>"]
    special_tokens.append("<EOS_CONCEPT>")
    for k in range(cfg.size):
        special_tokens.append(f"<C_{k}>")
    return special_tokens

def build_concept_meta(
    tokenizer: AutoTokenizer,
    cfg: ConceptConfig,
    device: str,
) -> ConceptMeta:
    """resolve concept token IDs and pack them into runtime metadata."""
    eos_id = tokenizer.convert_tokens_to_ids("<EOS_CONCEPT>")
    concept_tokens = [f"<C_{k}>" for k in range(cfg.size)]
    concept_ids = tokenizer.convert_tokens_to_ids(concept_tokens)
    concept_ids_t = torch.tensor(concept_ids, device=device, dtype=torch.long)
    concept_ids_eos = torch.cat(
        [concept_ids_t, torch.tensor([eos_id], device=device, dtype=torch.long)], dim=0
    )
    return ConceptMeta(
        type_id=TYPE_ID_CONCEPT,
        eos_id=eos_id,
        concept_ids=concept_ids_t,
        concept_ids_with_eos=concept_ids_eos,
        max_steps=cfg.max_steps,
        target_ratio=cfg.target_ratio,
    )

def plan_concepts(
    model: SharedBackboneUnifiedHead,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    plan_token_id: int,
    bos_id: int,
    meta: ConceptMeta,
    mask_cache: ConceptMaskCache,
    tau: float,
    min_concept_steps: int,
    base_vocab_size: int,
    device: str,
    sampling_mode: str = "gumbel",
    mix_greedy_ratio: float = 0.0,
    use_cache: bool = True,
) -> PlannerOutput:
    """generate one variable-length concept sequence from source input."""
    bsz, _ = input_ids.shape
    hidden_size = model.hidden_size
    dtype_embed = model.token_embed_base.weight.dtype

    bos_col = torch.full((bsz, 1), bos_id, device=device, dtype=torch.long)
    plan_col = torch.full((bsz, 1), plan_token_id, device=device, dtype=torch.long)
    planner_input_ids = torch.cat([bos_col, input_ids, plan_col], dim=1)
    planner_type_ids = torch.full_like(planner_input_ids, TYPE_ID_TEXT)
    planner_mask = torch.cat(
        [
            torch.ones((bsz, 1), device=device, dtype=torch.long),
            attention_mask,
            torch.ones((bsz, 1), device=device, dtype=torch.long),
        ],
        dim=1,
    )
    planner_pos = torch.cumsum(planner_mask, dim=1) - 1
    planner_pos = planner_pos.clamp(min=0)

    planner_embeds = model.embed_with_type(planner_input_ids, planner_type_ids)
    out = model.forward_backbone(
        inputs_embeds=planner_embeds,
        attention_mask=planner_mask,
        position_ids=planner_pos,
        use_cache=use_cache,
    )
    logits_t = model.forward_head(out.last_hidden_state[:, -1, :])  # [B, V]
    past_kv = out.past_key_values

    ones_step = torch.ones((bsz, 1), device=device, dtype=torch.long)
    eos_id = int(meta.eos_id)
    src_lengths = attention_mask.sum(dim=1).to(torch.long)
    expected = (src_lengths.float() * meta.target_ratio).long().clamp(min=1, max=meta.max_steps)

    commit_sum1 = torch.zeros((), device=device, dtype=torch.float32)
    commit_sum2 = torch.zeros((), device=device, dtype=torch.float32)
    hist_sum = torch.zeros((int(meta.concept_ids.numel()),), device=device, dtype=torch.float32)
    eos_sum = torch.zeros((), device=device, dtype=torch.float32)
    eos_count = torch.zeros((), device=device, dtype=torch.float32)

    concept_table = model.embed_tokens(meta.concept_ids_with_eos)  # [K+1, H]
    type_vec = model.type_embed.weight[int(meta.type_id)].view(1, -1).to(dtype=dtype_embed)

    concept_tokens = torch.full((bsz, meta.max_steps), eos_id, device=device, dtype=torch.long)
    concept_valid = torch.zeros((bsz, meta.max_steps), device=device, dtype=torch.long)
    concept_st_embeds = torch.zeros((bsz, meta.max_steps, hidden_size), device=device, dtype=dtype_embed)
    finished = torch.zeros((bsz,), device=device, dtype=torch.bool)

    quota_mass_sum = torch.zeros((), device=device, dtype=torch.float32)
    quota_count = torch.zeros((), device=device, dtype=torch.float32)

    cache_attention_mask = planner_mask
    cache_position_ids = planner_pos[:, -1:].clone() + 1

    for step in range(meta.max_steps):
        active = ~finished
        if not torch.any(active):
            break

        masked_logits = logits_t.float().clone() + mask_cache.allowed_logits_bias.view(1, -1)
        if torch.any(~active):
            masked_logits[~active] = -1e4
            masked_logits[~active, eos_id] = 0.0

        if min_concept_steps > 1 and step < (min_concept_steps - 1):
            masked_logits[active, eos_id] = -1e4
        if step >= (meta.max_steps - 1):
            forced = torch.full_like(masked_logits[active], -1e4)
            forced[:, eos_id] = 0.0
            masked_logits[active] = forced

        probs, sampled_ids = _select_planner_tokens(
            masked_logits=masked_logits,
            tau=tau,
            sampling_mode=sampling_mode,
            mix_greedy_ratio=mix_greedy_ratio,
        )

        sampled_ids = torch.where(
            active,
            sampled_ids,
            torch.full_like(sampled_ids, eos_id),
        )

        concept_tokens[active, step] = sampled_ids[active]
        concept_valid[active, step] = 1

        if base_vocab_size > 0:
            rows_logits = masked_logits[active]
            logz = torch.logsumexp(rows_logits, dim=-1)
            logz_base = torch.logsumexp(rows_logits[:, :base_vocab_size], dim=-1)
            base_mass = torch.exp(logz_base - logz)
            active_ids = sampled_ids[active]
            count_mask = active_ids.ne(eos_id)
            if torch.any(count_mask):
                quota_mass_sum = quota_mass_sum + base_mass[count_mask].sum()
                quota_count = quota_count + count_mask.sum().to(quota_count.dtype)

        active_rows = active.nonzero(as_tuple=False).squeeze(1)
        probs_subset = probs[active_rows].index_select(1, meta.concept_ids_with_eos)  # [N, K+1]
        hist_sum = hist_sum + probs_subset[:, :-1].sum(dim=0)

        soft_t = torch.matmul(probs_subset.to(concept_table.dtype), concept_table).to(dtype=dtype_embed) + type_vec
        hard_t = model.embed_tokens(sampled_ids[active_rows]).to(dtype=dtype_embed) + type_vec
        st_t = hard_t + (soft_t - soft_t.detach())

        soft_f = soft_t.float()
        hard_f = hard_t.float()
        commit_sum1 = commit_sum1 + (soft_f.detach() - hard_f).pow(2).sum()
        commit_sum2 = commit_sum2 + (soft_f - hard_f.detach()).pow(2).sum()

        eos_target = (step >= (expected[active_rows] - 1)).float()
        eos_logit = masked_logits[active_rows, eos_id]
        eos_sum = eos_sum + F.binary_cross_entropy_with_logits(
            eos_logit.float(), eos_target, reduction="sum"
        )
        eos_count = eos_count + eos_target.numel()

        concept_st_embeds = concept_st_embeds.index_put(
            (active_rows, torch.full_like(active_rows, step)),
            st_t,
        )
        finished = finished | sampled_ids.eq(eos_id)
        if torch.all(finished):
            break

        soft_embed_step = torch.zeros((bsz, hidden_size), device=device, dtype=dtype_embed)
        hard_embed_step = torch.zeros((bsz, hidden_size), device=device, dtype=dtype_embed)
        soft_embed_step[active_rows] = soft_t
        hard_embed_step[active_rows] = hard_t
        if torch.any(~active):
            dummy_ids = torch.full(((~active).sum().item(),), eos_id, device=device, dtype=torch.long)
            hard_embed_step[~active] = model.embed_tokens(dummy_ids) + type_vec
            soft_embed_step[~active] = hard_embed_step[~active]

        st_embed = hard_embed_step + (soft_embed_step - soft_embed_step.detach())  # [B, H]
        cache_attention_mask = torch.cat(
            [cache_attention_mask, torch.ones((bsz, 1), device=device, dtype=torch.long)],
            dim=1
        )
        cache_position_ids = cache_position_ids + 1

        out_next = model.forward_backbone(
            inputs_embeds=st_embed.unsqueeze(1),  # [B, 1, H]
            attention_mask=cache_attention_mask,  # [B, T_curr]
            position_ids=cache_position_ids,      # [B, 1]
            past_key_values=past_kv,
            use_cache=use_cache,
        )
        logits_t = model.forward_head(out_next.last_hidden_state[:, -1, :])  # [B, V]
        past_kv = out_next.past_key_values

    actual_lengths = concept_valid.sum(dim=1)
    concept_out = PlannerTypeOutput(
        token_ids=concept_tokens,
        valid_mask=concept_valid,
        actual_lengths=actual_lengths,
        st_embeds=concept_st_embeds,
    )

    denom = max(1.0, float(bsz * meta.max_steps * hidden_size))
    loss_commit = commit_sum1 / denom + BETA_COMMIT * (commit_sum2 / denom)

    hist_total = hist_sum.sum()
    if float(hist_total.detach().item()) > 0.0:
        hist = hist_sum / hist_total.clamp_min(1.0)
        loss_unif = usage_kl_to_uniform(hist)
    else:
        loss_unif = torch.zeros((), device=device, dtype=torch.float32)

    loss_eos = eos_sum / eos_count.clamp_min(1.0)
    loss_len = F.relu(actual_lengths.float() - expected.float()).mean()

    return (
        PlannerOutput(
            concept=concept_out,
            quota_mass_sum=quota_mass_sum,
            quota_count=quota_count,
        ),
        loss_commit,
        loss_unif,
        loss_eos,
        loss_len,
    )

def build_executor_prefix(
    model: SharedBackboneUnifiedHead,
    *,
    planner_out: PlannerOutput,
    meta: ConceptMeta,
    bos_id: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """build concept-only executor prefix with type ids, mask, reset positions, and embeddings."""
    bsz = planner_out.concept.token_ids.size(0)

    # Each list stores one prefix block; blocks are concatenated at the end.
    # token/type/mask/pos block shape: [B, L_i], embed block shape: [B, L_i, H].
    token_chunks: List[torch.Tensor] = []
    type_chunks: List[torch.Tensor] = []
    mask_chunks: List[torch.Tensor] = []
    embed_chunks: List[torch.Tensor] = []

    # Block 0: BOS-only prefix.
    # Executor starts from BOS and does not see source text tokens.
    bos = torch.full((bsz, 1), bos_id, device=device, dtype=torch.long)
    bos_type = torch.full((bsz, 1), TYPE_ID_TEXT, device=device, dtype=torch.long)
    token_chunks.append(bos)
    type_chunks.append(bos_type)
    mask_chunks.append(torch.ones((bsz, 1), device=device, dtype=torch.long))
    embed_chunks.append(model.embed_with_type(bos, bos_type))

    type_out = planner_out.concept
    type_lens = type_out.valid_mask.sum(dim=1).to(torch.long)
    max_len = int(type_lens.max().item())

    assert max_len > 0, "Planner did not generate any valid concept tokens, cannot build executor prefix."
    tok = type_out.token_ids[:, :max_len]
    msk = type_out.valid_mask[:, :max_len]
    typ = torch.full_like(tok, meta.type_id)
    token_chunks.append(tok)
    type_chunks.append(typ)
    mask_chunks.append(msk)
    if type_out.st_embeds is not None:
        st = type_out.st_embeds[:, :max_len, :]
        st = st * msk.unsqueeze(-1).to(st.dtype)
        embed_chunks.append(st)
    else:
        embed_chunks.append(model.embed_with_type(tok, typ))

    prefix_token_ids = torch.cat(token_chunks, dim=1)
    prefix_type_ids = torch.cat(type_chunks, dim=1)
    prefix_mask = torch.cat(mask_chunks, dim=1)
    prefix_embeds = torch.cat(embed_chunks, dim=1)
    prefix_pos = torch.cumsum(prefix_mask, dim=1) - 1
    prefix_pos = prefix_pos.clamp(min=0).to(torch.long)

    return prefix_embeds, prefix_mask, prefix_pos, prefix_token_ids, prefix_type_ids

def build_decoder_tensors(
    input_ids: torch.Tensor,        # [B, T] 原句 token（已 padding）
    attention_mask: torch.Tensor,   # [B, T] 1=有效 token, 0=padding
    bos_id: int,                    # BOS token id
    eos_id: int,                    # EOS token id
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造自回归(AR)训练用的 decoder 输入/掩码/标签。"""
    bsz = input_ids.size(0)                              # batch 大小 B
    lengths = attention_mask.sum(dim=1).to(torch.long)  # 每条样本真实长度 [B]
    max_len = int(lengths.max().item())                  # batch 内最大真实长度
    dec_len = max_len + 1                                # +1 给 EOS 位置

    # decoder 输入，先全填 eos_id，后面再写有效位置
    decoder_in = torch.full((bsz, dec_len), eos_id, device=device, dtype=torch.long)
    # decoder 注意力 mask，1 表示该位置参与计算
    decoder_mask = torch.zeros((bsz, dec_len), device=device, dtype=torch.long)
    # 监督标签，-100 表示忽略（CrossEntropy 默认 ignore_index）
    labels = torch.full((bsz, dec_len), -100, device=device, dtype=torch.long)

    for b in range(bsz):
        lb = int(lengths[b].item())                      # 第 b 条样本真实长度
        decoder_in[b, 0] = bos_id                        # 起始位置放 BOS
        if lb > 0:
            decoder_in[b, 1 : lb + 1] = input_ids[b, :lb]  # 输入为 [BOS, x1, x2, ...]
            labels[b, :lb] = input_ids[b, :lb]             # 预测目标为 [x1, x2, ...]
        labels[b, lb] = eos_id                            # 最后一个有效目标是 EOS
        decoder_mask[b, : lb + 1] = 1                     # 有效区间长度是 lb+1

    return decoder_in, decoder_mask, labels, lengths


# Backward-compatible alias for older imports.
SharedBackboneTwoHeads = SharedBackboneUnifiedHead

def build_executor_blocklist(
    meta: ConceptMeta,
    plan_token_id: int,
) -> List[int]:
    """return token IDs that executor logits must never generate."""
    blocked: List[int] = [plan_token_id]
    blocked.append(meta.eos_id)
    blocked.extend(meta.concept_ids.tolist())
    return blocked

####################################################################
#                                                                  #
# Inference utilities: planner->prefix->executor greedy decoding.  #
#                                                                  #
####################################################################

@dataclass
class ExecutorInferenceOutput:
    generated_ids: torch.Tensor
    generated_mask: torch.Tensor
    lengths: torch.Tensor
    planner_out: PlannerOutput
    prefix_token_ids: torch.Tensor
    prefix_type_ids: torch.Tensor
    prefix_mask: torch.Tensor
    prefix_pos: torch.Tensor


@torch.no_grad()
def run_executor_inference(
    model: SharedBackboneUnifiedHead,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    plan_token_id: int,
    bos_id: int,
    eos_id: int,
    meta: ConceptMeta,
    mask_cache: ConceptMaskCache,
    device: str,
    max_new_tokens: int = 64,
    planner_tau: float = 0.2,
    min_concept_steps: int = 1,
) -> ExecutorInferenceOutput:
    """run planner->prefix->executor greedy decoding for inference."""
    bsz = input_ids.size(0)
    max_new_tokens = max(0, int(max_new_tokens))

    was_training = model.training
    model.eval()
    try:
        planner_out, *_ = plan_concepts(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            plan_token_id=plan_token_id,
            bos_id=bos_id,
            meta=meta,
            mask_cache=mask_cache,
            tau=planner_tau,
            sampling_mode="greedy",
            min_concept_steps=min_concept_steps,
            base_vocab_size=model.base_vocab_size,
            device=device,
        )
        (
            prefix_embeds,
            prefix_mask,
            prefix_pos,
            prefix_token_ids,
            prefix_type_ids,
        ) = build_executor_prefix(
            model,
            planner_out=planner_out,
            meta=meta,
            bos_id=bos_id,
            device=device,
        )

        assert max_new_tokens > 0, "max_new_tokens must be positive for inference decoding."
        generated_ids = torch.full((bsz, max_new_tokens), eos_id, device=device, dtype=torch.long)
        generated_mask = torch.zeros((bsz, max_new_tokens), device=device, dtype=torch.long)

        # Prime executor cache with [prefix, BOS] so first logits predict token_1.
        bos = torch.full((bsz, 1), bos_id, device=device, dtype=torch.long)
        bos_type = torch.full((bsz, 1), TYPE_ID_TEXT, device=device, dtype=torch.long)
        bos_embed = model.embed_with_type(bos, bos_type)
        bos_mask = torch.ones((bsz, 1), device=device, dtype=torch.long)

        prefix_true_len = prefix_mask.sum(dim=1).to(torch.long)  # [B]
        bos_pos = prefix_true_len.unsqueeze(1)                    # [B, 1]

        prime_embeds = torch.cat([prefix_embeds, bos_embed], dim=1)
        prime_mask = torch.cat([prefix_mask, bos_mask], dim=1)
        prime_pos = torch.cat([prefix_pos, bos_pos], dim=1)

        out = model.forward_backbone(
            inputs_embeds=prime_embeds,
            attention_mask=prime_mask,
            position_ids=prime_pos,
            use_cache=True,
        )
        logits_t = model.forward_head(out.last_hidden_state[:, -1, :]).float()  # [B, V]
        logits_t = logits_t.masked_fill(mask_cache.executor_block_bool.view(1, -1), -1e4)
        past_kv = out.past_key_values

        ones_step = torch.ones((bsz, 1), device=device, dtype=torch.long)
        finished = torch.zeros((bsz,), device=device, dtype=torch.bool)
        fed_len = 1  # already fed BOS
        eos_fill = torch.full((bsz,), eos_id, device=device, dtype=torch.long)
        
        # [FIX] Initialize running attention mask for AR generation.
        cache_attention_mask = prime_mask

        for step in range(max_new_tokens):
            next_ids = logits_t.argmax(dim=-1)        # [B]
            next_ids = torch.where(finished, eos_fill, next_ids)

            generated_ids[:, step] = next_ids
            generated_mask[:, step] = (~finished).to(torch.long)
            finished = finished | next_ids.eq(eos_id)
            if torch.all(finished):
                break

            next_tok = next_ids.unsqueeze(1)  # [B, 1]
            next_type = torch.full_like(next_tok, TYPE_ID_TEXT)
            next_embed = model.embed_with_type(next_tok, next_type)
            step_pos = (prefix_true_len + fed_len).unsqueeze(1)  # [B, 1]

            
            # [FIX] Update attention mask to include the new token.
            cache_attention_mask = torch.cat([cache_attention_mask, ones_step], dim=1)

            out = model.forward_backbone(
                inputs_embeds=next_embed,
                attention_mask=cache_attention_mask,
                position_ids=step_pos,
                past_key_values=past_kv,
                use_cache=True,
            )
            logits_t = model.forward_head(out.last_hidden_state[:, -1, :]).float()
            logits_t = logits_t.masked_fill(mask_cache.executor_block_bool.view(1, -1), -1e4)
            past_kv = out.past_key_values
            fed_len += 1

        lengths = generated_mask.sum(dim=1)
        return ExecutorInferenceOutput(
            generated_ids=generated_ids,
            generated_mask=generated_mask,
            lengths=lengths,
            planner_out=planner_out,
            prefix_token_ids=prefix_token_ids,
            prefix_type_ids=prefix_type_ids,
            prefix_mask=prefix_mask,
            prefix_pos=prefix_pos,
        )
    finally:
        if was_training:
            model.train()
