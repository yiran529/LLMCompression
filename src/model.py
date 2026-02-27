# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


from src.config.train_config import BETA_COMMIT, EPS, TYPE_ID_TEXT, ConceptTypeConfig

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
class ConceptTypeMeta:
    name: str
    type_id: int
    eos_id: int
    concept_ids: torch.Tensor
    concept_ids_with_eos: torch.Tensor
    max_steps: int
    target_ratio: float

class ConceptMaskCache:
    def __init__(
        self,
        metas: List[ConceptTypeMeta],
        vocab_size: int,
        base_vocab_size: int,
        blocked_for_executor: List[int],
        device: str,
    ):
        """build reusable per-type planner masks and executor output block mask."""
        # Planner 每个 concept type 都有一张“词表白名单”偏置表。
        # 用法：logits + bias 后，非白名单 token 会被加上 very_neg（近似 -inf），
        # 从而在 softmax / 采样时几乎不可能被选中。
        self.allowed_logits_bias: Dict[str, torch.Tensor] = {}
        very_neg = -1e4
        for meta in metas:
            bias = torch.full((vocab_size,), very_neg, device=device, dtype=torch.float32)
            # 允许输出：原词表 token + 该 type 的概念 token + 该 type 的 EOS token。
            if base_vocab_size > 0:
                bias[:base_vocab_size] = 0.0
            bias[meta.concept_ids_with_eos] = 0.0
            self.allowed_logits_bias[meta.name] = bias

        # Executor 端输出屏蔽表：True 的位置表示“禁止生成”。
        # 这里通常屏蔽所有 planner 专用 token（<PLAN>/<EOS_i>/<C_i_*>)，
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
            out[is_base] = self.token_embed_base(flat_ids[is_base])
        if torch.any(~is_base):
            out[~is_base] = self.token_embed_new(flat_ids[~is_base] - self.base_vocab_size)
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
    per_type: List[PlannerTypeOutput]
    quota_mass_sum: torch.Tensor
    quota_count: torch.Tensor
    debug_records: Optional[List[Dict[str, Any]]] = None

class PlannerQuotaController:
    def __init__(self, *, tau: float, eta: float, lambda_init: float, device: str):
        self.tau = float(tau)
        self.eta = float(eta)
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
            self.lambda_value = torch.clamp(new_lambda, min=0.0)
        return loss_quota, bar_m

def compute_planner_quota_loss(
    planner_out: PlannerOutput,
    quota_controller: PlannerQuotaController,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if quota_controller is None:
        zero = torch.zeros((), device=planner_out.quota_mass_sum.device, dtype=torch.float32)
        return zero, zero
    return quota_controller.compute_loss(planner_out.quota_mass_sum, planner_out.quota_count)

def build_concept_special_tokens(type_cfgs: List[ConceptTypeConfig]) -> List[str]:
    """define all planner-side special tokens for typed concept vocabularies."""
    special_tokens: List[str] = ["<PLAN>"]
    for cfg in type_cfgs:
        special_tokens.append(f"<EOS_{cfg.name}>")
        for k in range(cfg.size):
            special_tokens.append(f"<C_{cfg.name}_{k}>")
    return special_tokens

def build_concept_metas(
    tokenizer: AutoTokenizer,
    type_cfgs: List[ConceptTypeConfig],
    device: str,
) -> List[ConceptTypeMeta]:
    """resolve typed concept token IDs and pack them into runtime metadata."""
    metas: List[ConceptTypeMeta] = []
    for i, cfg in enumerate(type_cfgs, start=1):
        eos_id = tokenizer.convert_tokens_to_ids(f"<EOS_{cfg.name}>")
        concept_tokens = [f"<C_{cfg.name}_{k}>" for k in range(cfg.size)]
        concept_ids = tokenizer.convert_tokens_to_ids(concept_tokens)
        concept_ids_t = torch.tensor(concept_ids, device=device, dtype=torch.long)
        concept_ids_eos = torch.cat(
            [concept_ids_t, torch.tensor([eos_id], device=device, dtype=torch.long)], dim=0
        )
        metas.append(
            ConceptTypeMeta(
                name=cfg.name,
                type_id=i,
                eos_id=eos_id,
                concept_ids=concept_ids_t,
                concept_ids_with_eos=concept_ids_eos,
                max_steps=cfg.max_steps,
                target_ratio=cfg.target_ratio,
            )
        )
    return metas

def plan_concepts(
    model: SharedBackboneUnifiedHead,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    plan_token_id: int,
    bos_id: int,
    metas: List[ConceptTypeMeta],
    mask_cache: ConceptMaskCache,
    tau: float,
    min_concept_steps: int,
    base_vocab_size: int,
    device: str,
    deterministic: bool = False,
    debug_collect: bool = False,
    debug_topk: int = 5,
    debug_max_steps: int = 8,
    debug_max_samples: int = 4,
) -> PlannerOutput:
    """generate one variable-length concept sequence per type from source input."""
    bsz, _ = input_ids.shape
    hidden_size = model.hidden_size
    dtype_embed = model.token_embed_base.weight.dtype
    max_debug_samples = max(1, min(int(debug_max_samples), bsz))

    # ---------------------------------------------------------------------
    # 1) Build planner prompt.
    # Prompt layout is:
    #   [<BOS>, normal_tokens..., <PLAN>]
    # so the plan token is appended after the normal-token prefix.
    # ---------------------------------------------------------------------
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
    planner_pos = torch.arange(
        planner_input_ids.size(1), device=device, dtype=torch.long
    ).unsqueeze(0).expand(bsz, -1)

    # Prime the KV cache with the full planner prompt.
    # logits_t is the next-token distribution after consuming prompt.
    planner_embeds = model.embed_with_type(planner_input_ids, planner_type_ids)
    out = model.forward_backbone(
        inputs_embeds=planner_embeds,
        attention_mask=planner_mask,
        position_ids=planner_pos,
        use_cache=True,
    )
    prime_hidden = out.last_hidden_state[:, -1, :].float()  # [B, H]
    logits_t = model.forward_head(out.last_hidden_state[:, -1, :])  # [B, V]
    hidden_t = prime_hidden
    past_kv = out.past_key_values

    num_types = len(metas)
    ones_step = torch.ones((bsz, 1), device=device, dtype=torch.long)
    max_steps_by_type = torch.tensor([m.max_steps for m in metas], device=device, dtype=torch.long)
    eos_id_by_type = torch.tensor([m.eos_id for m in metas], device=device, dtype=torch.long)
    type_id_by_type = torch.tensor([m.type_id for m in metas], device=device, dtype=torch.long)
    src_lengths = attention_mask.sum(dim=1).to(torch.long)
    expected_by_type = [
        (src_lengths.float() * m.target_ratio).long().clamp(min=1, max=m.max_steps) for m in metas
    ]

    # Online auxiliary loss accumulators (avoid storing large per-step buffers).
    commit_sum1 = [torch.zeros((), device=device, dtype=torch.float32) for _ in metas]
    commit_sum2 = [torch.zeros((), device=device, dtype=torch.float32) for _ in metas]
    hist_sum = [
        torch.zeros((int(m.concept_ids.numel()),), device=device, dtype=torch.float32) for m in metas
    ]
    eos_sum = [torch.zeros((), device=device, dtype=torch.float32) for _ in metas]
    eos_count = [torch.zeros((), device=device, dtype=torch.float32) for _ in metas]

    # ---------------------------------------------------------------------
    # 2) Precompute type-level constants.
    # ---------------------------------------------------------------------
    # allowed_bias_by_type[t, v] is 0 for allowed tokens of type t and -1e4 otherwise.
    # Adding this to logits enforces the per-type vocabulary.
    allowed_bias_by_type = torch.stack(
        [mask_cache.allowed_logits_bias[m.name] for m in metas], dim=0
    )  # [T, V]
    # concept_tables[t] holds token embeddings for [C_t..., EOS_t].
    concept_tables = [
        model.embed_tokens(m.concept_ids_with_eos) for m in metas
    ]  # list of [K_i+1, H]

    # ---------------------------------------------------------------------
    # 3) Allocate fixed-shape output buffers per type.
    # We fill these buffers asynchronously using (sample_idx, step_in_type).
    # ---------------------------------------------------------------------
    buffers: List[Dict[str, torch.Tensor]] = []
    for meta in metas:
        buffers.append(
            {
                "token_ids": torch.full(
                    (bsz, meta.max_steps), meta.eos_id, device=device, dtype=torch.long
                ),
                "valid": torch.zeros((bsz, meta.max_steps), device=device, dtype=torch.long),
                "st_embeds": torch.zeros(
                    (bsz, meta.max_steps, hidden_size), device=device, dtype=dtype_embed
                ),
            }
        )

    # ---------------------------------------------------------------------
    # 4) Asynchronous per-sample state.
    # Each sample advances independently:
    # - cur_type_idx[b]: which concept type sample b is currently generating.
    # - step_in_type[b]: local step index within current type.
    # - finished[b]: whether sample b has finished all types.
    # ---------------------------------------------------------------------
    cur_type_idx = torch.zeros(bsz, device=device, dtype=torch.long)  # [B], range [0, T)
    step_in_type = torch.zeros(bsz, device=device, dtype=torch.long)  # [B], range [0, max_steps_i)
    finished = torch.zeros(bsz, device=device, dtype=torch.bool)      # [B]

    # Hard safety bound with max_steps constraint preserved.
    # A sample can emit at most sum(max_steps_i) tokens across all types.
    max_total_steps = int(max_steps_by_type.sum().item())
    quota_mass_sum = torch.zeros((), device=device, dtype=torch.float32)
    quota_count = torch.zeros((), device=device, dtype=torch.float32)

    # [FIX] Initialize running attention mask and position IDs for autoregressive generation.
    # The mask must cover [Prompt + Past_Generated], not just the current step.
    # The position IDs should ideally be continuous for RoPE-based models to maintain relative distance to the prompt.
    cache_attention_mask = planner_mask
    cache_position_ids = planner_pos[:, -1:].clone() + 1

    debug_records: Optional[List[Dict[str, Any]]] = [] if debug_collect else None
    if debug_collect and debug_records is not None:
        hidden_diffs: List[float] = []
        logits_diffs: List[float] = []
        hidden_norms: List[float] = []
        logits_norms: List[float] = []
        logits_means: List[float] = []
        logits_stds: List[float] = []
        logits_max_abs: List[float] = []
        concept_logits_diffs: List[float] = []
        concept_logits_centered_diffs: List[float] = []
        concept_logits_stds: List[float] = []
        concept_logits_means: List[float] = []
        concept_logits_max_abs: List[float] = []
        concept_top1_minus_top2: List[float] = []
        prime_type_idx = 0
        prime_allowed_ids = metas[prime_type_idx].concept_ids_with_eos
        prime_concept_logits = logits_t.float().index_select(1, prime_allowed_ids)
        for b in range(max_debug_samples):
            hidden_norms.append(float(prime_hidden[b].norm().item()))
            logits_row = logits_t[b].float()
            logits_norms.append(float(logits_row.norm().item()))
            logits_means.append(float(logits_row.mean().item()))
            logits_stds.append(float(logits_row.std(unbiased=False).item()))
            logits_max_abs.append(float(logits_row.abs().max().item()))
            concept_row = prime_concept_logits[b]
            concept_logits_means.append(float(concept_row.mean().item()))
            concept_logits_stds.append(float(prime_concept_logits[b].std(unbiased=False).item()))
            concept_logits_max_abs.append(float(concept_row.abs().max().item()))
            concept_top2 = torch.topk(concept_row, k=min(2, int(concept_row.numel())), dim=-1).values
            if concept_top2.numel() >= 2:
                concept_top1_minus_top2.append(float((concept_top2[0] - concept_top2[1]).item()))
            else:
                concept_top1_minus_top2.append(0.0)
            if b == 0:
                hidden_diffs.append(0.0)
                logits_diffs.append(0.0)
                concept_logits_diffs.append(0.0)
                concept_logits_centered_diffs.append(0.0)
            else:
                hidden_diffs.append(float((prime_hidden[b] - prime_hidden[0]).abs().max().item()))
                logits_diffs.append(float((logits_row - logits_t[0].float()).abs().max().item()))
                concept_logits_diffs.append(
                    float((prime_concept_logits[b] - prime_concept_logits[0]).abs().max().item())
                )
                concept_centered_b = concept_row - concept_row.mean()
                concept_centered_0 = prime_concept_logits[0] - prime_concept_logits[0].mean()
                concept_logits_centered_diffs.append(
                    float((concept_centered_b - concept_centered_0).abs().max().item())
                )

        debug_records.append(
            {
                "stage": "prime",
                "prompt_true_len": [int(x) for x in planner_mask.sum(dim=1).tolist()[:max_debug_samples]],
                "hidden_norms": hidden_norms,
                "logits_norms": logits_norms,
                "logits_means": logits_means,
                "logits_stds": logits_stds,
                "logits_max_abs": logits_max_abs,
                "hidden_max_abs_diff_vs0": hidden_diffs,
                "logits_max_abs_diff_vs0": logits_diffs,
                "concept_logits_means": concept_logits_means,
                "concept_logits_std": concept_logits_stds,
                "concept_logits_max_abs": concept_logits_max_abs,
                "concept_logits_max_abs_diff_vs0": concept_logits_diffs,
                "concept_logits_centered_max_abs_diff_vs0": concept_logits_centered_diffs,
                "concept_top1_minus_top2": concept_top1_minus_top2,
                "prime_type_name": metas[prime_type_idx].name,
            }
        )

    # ---------------------------------------------------------------------
    # 5) Main asynchronous decoding loop.
    # One loop iteration emits exactly one token per active sample.
    # ---------------------------------------------------------------------
    for _ in range(max_total_steps):
        active = ~finished
        if not torch.any(active):
            break

        # type_idx_safe lets us gather tensors for all rows (including finished rows)
        # without out-of-range indices.
        type_idx_safe = torch.clamp(cur_type_idx, max=num_types - 1)
        # Keep previous local step for indexing current write positions.
        step_before = step_in_type.clone()

        masked_logits = logits_t.float().clone()  # [B, V]

        # Apply type-specific vocab mask sample-wise.
        # Different samples can use different type masks in the same iteration.
        for t in range(num_types):
            row_mask = active & (type_idx_safe == t)
            if torch.any(row_mask):
                masked_logits[row_mask] = masked_logits[row_mask] + allowed_bias_by_type[t].view(1, -1)

        # Inactive rows still flow through batched ops; force deterministic dummy EOS.
        if torch.any(~active):
            masked_logits[~active] = -1e4
            masked_logits[~active, int(eos_id_by_type[-1].item())] = 0.0

        # Enforce per-sample decoding rules based on its current type:
        # - min_concept_steps: EOS blocked at early local steps
        # - max_steps_by_type: force EOS at last allowed local step
        for t in range(num_types):
            row_mask = active & (type_idx_safe == t)
            if not torch.any(row_mask):
                continue
            eos_t = int(eos_id_by_type[t].item())
            if min_concept_steps > 1:
                min_mask = row_mask & (step_before < (min_concept_steps - 1))
                if torch.any(min_mask):
                    masked_logits[min_mask, eos_t] = -1e4
            force_mask = row_mask & (step_before >= (max_steps_by_type[t] - 1))
            if torch.any(force_mask):
                forced = torch.full_like(masked_logits[force_mask], -1e4)
                forced[:, eos_t] = 0.0
                masked_logits[force_mask] = forced

        # Straight-through categorical sample from constrained logits.
        if deterministic:
            sampled_ids = masked_logits.argmax(dim=-1)
            probs = torch.zeros_like(masked_logits)
            probs.scatter_(1, sampled_ids.unsqueeze(1), 1.0)
        else:
            probs = gumbel_softmax_sample(masked_logits, tau=tau, hard=True)  # [B, V]
            sampled_ids = probs.argmax(dim=-1)  # [B]
        # Keep inactive rows deterministic.
        sampled_ids = torch.where(
            active,
            sampled_ids,
            torch.full_like(sampled_ids, int(eos_id_by_type[-1].item())),
        )

        if debug_collect and debug_records is not None:
            if len(debug_records) < max(1, int(debug_max_steps)):
                max_samples = max(1, min(int(debug_max_samples), bsz))
                sample_debug: List[Dict[str, Any]] = []
                for b in range(max_samples):
                    type_idx_b = int(type_idx_safe[b].item())
                    meta_b = metas[type_idx_b]
                    allowed_ids = meta_b.concept_ids_with_eos
                    allowed_logits = masked_logits[b].index_select(0, allowed_ids)
                    allowed_probs = F.softmax(allowed_logits, dim=-1)
                    k = max(1, min(int(debug_topk), int(allowed_probs.numel())))
                    top_vals, top_idx = torch.topk(allowed_probs, k=k, dim=-1)
                    top_token_ids = allowed_ids.index_select(0, top_idx)

                    selected_id = int(sampled_ids[b].item())
                    selected_prob = 0.0
                    match = (allowed_ids == selected_id).nonzero(as_tuple=False)
                    if match.numel() > 0:
                        selected_prob = float(allowed_probs[match[0, 0]].item())

                    entropy = float((-(allowed_probs * torch.log(allowed_probs + 1e-8))).sum().item())
                    hidden_row = hidden_t[b].float()
                    hidden_norm = float(hidden_row.norm().item())
                    hidden_std = float(hidden_row.std(unbiased=False).item())
                    hidden_max_abs = float(hidden_row.abs().max().item())
                    hidden_diff_vs0 = 0.0
                    hidden_centered_diff_vs0 = 0.0
                    hidden_cosine_vs0 = 1.0
                    if b > 0:
                        hidden_row0 = hidden_t[0].float()
                        hidden_diff_vs0 = float((hidden_row - hidden_row0).abs().max().item())
                        hidden_centered = hidden_row - hidden_row.mean()
                        hidden_centered0 = hidden_row0 - hidden_row0.mean()
                        hidden_centered_diff_vs0 = float((hidden_centered - hidden_centered0).abs().max().item())
                        hidden_cosine_vs0 = float(
                            F.cosine_similarity(
                                hidden_row.unsqueeze(0),
                                hidden_row0.unsqueeze(0),
                                dim=-1,
                                eps=1e-8,
                            ).item()
                        )
                    logits_mean = float(allowed_logits.mean().item())
                    concept_logits_std = float(allowed_logits.std(unbiased=False).item())
                    concept_logits_max_abs = float(allowed_logits.abs().max().item())
                    top2 = torch.topk(allowed_logits, k=min(2, int(allowed_logits.numel())), dim=-1).values
                    if top2.numel() >= 2:
                        top1_minus_top2 = float((top2[0] - top2[1]).item())
                    else:
                        top1_minus_top2 = 0.0
                    concept_logits_diff_vs0 = 0.0
                    concept_logits_centered_diff_vs0 = 0.0
                    if b > 0:
                        type_idx_0 = int(type_idx_safe[0].item())
                        if type_idx_0 == type_idx_b:
                            meta_0 = metas[type_idx_0]
                            allowed_ids_0 = meta_0.concept_ids_with_eos
                            allowed_logits_0 = masked_logits[0].index_select(0, allowed_ids_0)
                            concept_logits_diff_vs0 = float((allowed_logits - allowed_logits_0).abs().max().item())
                            allowed_centered = allowed_logits - allowed_logits.mean()
                            allowed_centered_0 = allowed_logits_0 - allowed_logits_0.mean()
                            concept_logits_centered_diff_vs0 = float(
                                (allowed_centered - allowed_centered_0).abs().max().item()
                            )
                    sample_debug.append(
                        {
                            "sample_idx": b,
                            "active": bool(active[b].item()),
                            "type_name": meta_b.name,
                            "step_in_type": int(step_before[b].item()),
                            "selected_id": selected_id,
                            "selected_prob": selected_prob,
                            "entropy": entropy,
                            "hidden_norm": hidden_norm,
                            "hidden_std": hidden_std,
                            "hidden_max_abs": hidden_max_abs,
                            "hidden_max_abs_diff_vs0": hidden_diff_vs0,
                            "hidden_centered_max_abs_diff_vs0": hidden_centered_diff_vs0,
                            "hidden_cosine_vs0": hidden_cosine_vs0,
                            "concept_logits_mean": logits_mean,
                            "concept_logits_std": concept_logits_std,
                            "concept_logits_max_abs": concept_logits_max_abs,
                            "top1_minus_top2": top1_minus_top2,
                            "concept_logits_max_abs_diff_vs0": concept_logits_diff_vs0,
                            "concept_logits_centered_max_abs_diff_vs0": concept_logits_centered_diff_vs0,
                            "topk_token_ids": [int(x) for x in top_token_ids.tolist()],
                            "topk_probs": [float(x) for x in top_vals.tolist()],
                        }
                    )

                debug_records.append(
                    {
                        "decode_step": len(debug_records),
                        "deterministic": bool(deterministic),
                        "tau": float(tau),
                        "samples": sample_debug,
                    }
                )

        # Build one-step inputs for the next backbone call.
        # We keep full [B, ...] tensors for efficient batched forward.
        soft_embed_step = torch.zeros((bsz, hidden_size), device=device, dtype=dtype_embed)
        hard_embed_step = torch.zeros((bsz, hidden_size), device=device, dtype=dtype_embed)
        step_pos = torch.zeros((bsz, 1), device=device, dtype=torch.long)

        for t, meta in enumerate(metas):
            row_mask = active & (type_idx_safe == t)
            if not torch.any(row_mask):
                continue

            rows = row_mask.nonzero(as_tuple=False).squeeze(1)  # [N_t]   N_t 表示当前步、当前类型 t 的有效样本数（batch 里满足 row_mask 的行数）
            local_step = step_before[rows]  # [N_t]
            eos_t = int(meta.eos_id)
            tok_t = sampled_ids[rows]  # [N_t]

            # Write current results into per-type buffers at async coordinates:
            #   buffer[type][sample_row, local_step]
            buffers[t]["token_ids"][rows, local_step] = tok_t
            buffers[t]["valid"][rows, local_step] = 1
            # Used for computing quota loss later.
            if base_vocab_size > 0:
                rows_logits = masked_logits[rows]
                # 原词表 + 当前concept tokens词表（包含EOS）一起构成分母，计算EOS占比。
                logz = torch.logsumexp(rows_logits, dim=-1)
                # 原词表部分的概率质量总和，作为“超额”概念生成的 proxy 指标。
                logz_base = torch.logsumexp(rows_logits[:, :base_vocab_size], dim=-1)
                # 原词表概率质量占比越大，说明生成的概念越“节约”，越不超额。
                base_mass = torch.exp(logz_base - logz)
                count_mask = tok_t.ne(eos_t)
                if torch.any(count_mask):
                    quota_mass_sum = quota_mass_sum + base_mass[count_mask].sum()
                    quota_count = quota_count + count_mask.sum().to(quota_count.dtype)

            # Keep only the current type's concept space for soft embedding and stats.
            probs_subset = probs[rows].index_select(1, meta.concept_ids_with_eos)  # [N_t, K_i+1]
            hist_sum[t] = hist_sum[t] + probs_subset[:, :-1].sum(dim=0)

            # Add type embedding to both soft/hard token embeddings.
            type_vec_t = model.type_embed.weight[int(type_id_by_type[t].item())].view(1, -1).to(dtype=dtype_embed)  # [1, H]
            soft_t = torch.matmul(probs_subset.to(concept_tables[t].dtype), concept_tables[t]).to(dtype=dtype_embed) + type_vec_t
            hard_t = model.embed_tokens(tok_t).to(dtype=dtype_embed) + type_vec_t
            st_t = hard_t + (soft_t - soft_t.detach())
            soft_f = soft_t.float()  # [N_t, H]
            hard_f = hard_t.float()  # [N_t, H]
            # Commitment loss terms:
            # L1 = sum ||sg(soft) - hard||^2, L2 = sum ||soft - sg(hard)||^2
            commit_sum1[t] = commit_sum1[t] + (soft_f.detach() - hard_f).pow(2).sum()
            commit_sum2[t] = commit_sum2[t] + (soft_f - hard_f.detach()).pow(2).sum()

            expected_t = expected_by_type[t][rows]  # [N_t]
            eos_target = (local_step >= (expected_t - 1)).float()  # [N_t]
            eos_logit = masked_logits[rows, eos_t]  # [N_t]
            # EOS loss: sum BCEWithLogits(eos_logit, eos_target)
            eos_sum[t] = eos_sum[t] + F.binary_cross_entropy_with_logits(
                eos_logit.float(), eos_target, reduction="sum"
            )
            eos_count[t] = eos_count[t] + eos_target.numel()

            soft_embed_step[rows] = soft_t
            hard_embed_step[rows] = hard_t
            # Keep a differentiable write path so stage-2 loss can flow back to planner ST embeddings.
            buffers[t]["st_embeds"] = buffers[t]["st_embeds"].index_put((rows, local_step), st_t)
            step_pos[rows, 0] = local_step + 1  # per-type position reset rule.

            # Async state transition per sample:
            # - sampled EOS_t: move to next type, reset local step to 0
            # - otherwise: stay in same type, local step + 1
            stop_t = tok_t.eq(eos_t)
            next_type = cur_type_idx[rows] + stop_t.long()
            cur_type_idx[rows] = next_type
            step_in_type[rows] = torch.where(
                stop_t,
                torch.zeros_like(local_step),
                local_step + 1,
            )
            finished[rows] = next_type >= num_types

        # Finished rows still participate in batched forward with harmless dummy values.
        if torch.any(~active):
            dummy_ids = torch.full(
                ((~active).sum().item(),),
                int(eos_id_by_type[-1].item()),
                device=device,
                dtype=torch.long,
            )
            hard_embed_step[~active] = model.embed_tokens(dummy_ids)
            soft_embed_step[~active] = hard_embed_step[~active]
            step_pos[~active, 0] = 0

        # Straight-through one-step input to advance shared KV cache.
        st_embed = hard_embed_step + (soft_embed_step - soft_embed_step.detach())  # [B, H]

        # [FIX] Update attention mask and position IDs for the next step.
        # AR generation requires position IDs to increase monotonically to maintain RoPE relative distances,
        # unless specific "restart" semantics are intended. Given concepts are generated sequentially, 
        # continuous position IDs are safer to avoid attention anomalies.
        # We ignore the per-type `step_pos` reset logic for the BACKBONE forward pass to keep the LM context valid.
        
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
            use_cache=True,
        )
        logits_t = model.forward_head(out_next.last_hidden_state[:, -1, :])  # [B, V]
        hidden_t = out_next.last_hidden_state[:, -1, :].float()
        past_kv = out_next.past_key_values

    # ---------------------------------------------------------------------
    # 6) Pack per-type buffers back into PlannerOutput format.
    # ---------------------------------------------------------------------
    per_type_outputs: List[PlannerTypeOutput] = []
    for t, meta in enumerate(metas):
        token_ids = buffers[t]["token_ids"]
        valid_mask = buffers[t]["valid"]
        actual_lengths = valid_mask.sum(dim=1)

        per_type_outputs.append(
            PlannerTypeOutput(
                token_ids=token_ids,
                valid_mask=valid_mask,
                actual_lengths=actual_lengths,
                st_embeds=buffers[t]["st_embeds"],
            )
        )

    loss_commit = torch.zeros((), device=device, dtype=torch.float32)
    loss_unif = torch.zeros((), device=device, dtype=torch.float32)
    loss_eos = torch.zeros((), device=device, dtype=torch.float32)
    loss_len = torch.zeros((), device=device, dtype=torch.float32)
    n = max(1, len(metas))
    for t, meta in enumerate(metas):
        denom = max(1.0, float(bsz * meta.max_steps * hidden_size))
        loss_commit = loss_commit + commit_sum1[t] / denom + BETA_COMMIT * (commit_sum2[t] / denom)

        hist = hist_sum[t]
        hist = hist / (hist.sum() + EPS)
        loss_unif = loss_unif + usage_kl_to_uniform(hist)

        denom_eos = eos_count[t].clamp_min(1.0)
        loss_eos = loss_eos + eos_sum[t] / denom_eos

        expected = expected_by_type[t]
        actual = per_type_outputs[t].actual_lengths
        loss_len = loss_len + F.relu(actual.float() - expected.float()).mean()

    loss_commit = loss_commit / n
    loss_unif = loss_unif / n
    loss_eos = loss_eos / n
    loss_len = loss_len / n

    return (
        PlannerOutput(
            per_type=per_type_outputs,
            quota_mass_sum=quota_mass_sum,
            quota_count=quota_count,
            debug_records=debug_records,
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
    metas: List[ConceptTypeMeta],
    bos_id: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """build concept-only executor prefix with type ids, mask, reset positions, and embeddings."""
    # Batch size B.
    bsz = planner_out.per_type[0].token_ids.size(0)

    # Each list stores one prefix block; blocks are concatenated at the end.
    # token/type/mask/pos block shape: [B, L_i], embed block shape: [B, L_i, H].
    token_chunks: List[torch.Tensor] = []
    type_chunks: List[torch.Tensor] = []
    mask_chunks: List[torch.Tensor] = []
    pos_chunks: List[torch.Tensor] = []
    embed_chunks: List[torch.Tensor] = []

    # Block 0: BOS-only prefix.
    # Executor starts from BOS and does not see source text tokens.
    bos = torch.full((bsz, 1), bos_id, device=device, dtype=torch.long)
    bos_type = torch.full((bsz, 1), TYPE_ID_TEXT, device=device, dtype=torch.long)
    token_chunks.append(bos)
    type_chunks.append(bos_type)
    mask_chunks.append(torch.ones((bsz, 1), device=device, dtype=torch.long))
    pos_chunks.append(torch.zeros((bsz, 1), device=device, dtype=torch.long))
    embed_chunks.append(model.embed_with_type(bos, bos_type))

    # Blocks 1..T: append each concept-type segment from planner output.
    for meta, type_out in zip(metas, planner_out.per_type):
        # Trim each type block to the maximum valid length in the batch to save memory.
        # type_lens: [B], max_len: scalar.
        type_lens = type_out.valid_mask.sum(dim=1).to(torch.long)
        max_len = int(type_lens.max().item())
        if max_len == 0:
            continue
        # tok/msk/typ: [B, max_len].
        tok = type_out.token_ids[:, :max_len]
        msk = type_out.valid_mask[:, :max_len]
        typ = torch.full_like(tok, meta.type_id)

        token_chunks.append(tok)
        type_chunks.append(typ)
        mask_chunks.append(msk)
        if type_out.st_embeds is not None:
            # st: [B, max_len, H], zero out padded positions with msk.
            st = type_out.st_embeds[:, :max_len, :]
            st = st * msk.unsqueeze(-1).to(st.dtype)
            embed_chunks.append(st)
        else:
            # Fallback to standard token+type embedding if ST embeds are absent.
            embed_chunks.append(model.embed_with_type(tok, typ))
        # Critical rule: position indices restart from 1 for each type block.
        # pos block shape: [B, max_len].
        pos_chunks.append(
            torch.arange(1, max_len + 1, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(bsz, -1)
        )

    # Concatenate all prefix blocks into final executor inputs.
    # prefix_token_ids/prefix_type_ids/prefix_mask/prefix_pos: [B, Lp]
    # prefix_embeds: [B, Lp, H]
    prefix_token_ids = torch.cat(token_chunks, dim=1)
    prefix_type_ids = torch.cat(type_chunks, dim=1)
    prefix_mask = torch.cat(mask_chunks, dim=1)
    prefix_pos = torch.cat(pos_chunks, dim=1)
    prefix_embeds = torch.cat(embed_chunks, dim=1)
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
    metas: List[ConceptTypeMeta],
    plan_token_id: int,
) -> List[int]:
    """return token IDs that executor logits must never generate."""
    blocked: List[int] = [plan_token_id]
    for meta in metas:
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
    metas: List[ConceptTypeMeta],
    mask_cache: ConceptMaskCache,
    device: str,
    max_new_tokens: int = 64,
    planner_tau: float = 0.2,
    min_concept_steps: int = 1,
    planner_deterministic: bool = False,
    planner_debug_collect: bool = False,
    planner_debug_topk: int = 5,
    planner_debug_max_steps: int = 8,
    planner_debug_max_samples: int = 4,
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
            metas=metas,
            mask_cache=mask_cache,
            tau=planner_tau,
            min_concept_steps=min_concept_steps,
            base_vocab_size=model.base_vocab_size,
            device=device,
            deterministic=planner_deterministic,
            debug_collect=planner_debug_collect,
            debug_topk=planner_debug_topk,
            debug_max_steps=planner_debug_max_steps,
            debug_max_samples=planner_debug_max_samples,
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
            metas=metas,
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
