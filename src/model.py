# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .config import BETA_COMMIT, EPS, TYPE_ID_TEXT, ConceptTypeConfig
except ImportError:
    from config import BETA_COMMIT, EPS, TYPE_ID_TEXT, ConceptTypeConfig


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
        if not hasattr(base_model, "model"):
            raise RuntimeError("Expected a causal LM with `.model` backbone (Qwen/LLaMA-style).")
        self.base_model = base_model
        self.backbone = base_model.model
        self.token_embed = base_model.get_input_embeddings()
        out_embed = base_model.get_output_embeddings()
        vocab_size = self.token_embed.num_embeddings
        hidden_size = self.token_embed.embedding_dim

        self.output_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.type_embed = nn.Embedding(num_type_embeddings, hidden_size)

        if out_embed is not None and out_embed.weight.shape == self.output_head.weight.shape:
            self.output_head.weight.data.copy_(out_embed.weight.data)
        else:
            self.output_head.weight.data.copy_(self.token_embed.weight.data)

        nn.init.zeros_(self.type_embed.weight)
        self.register_buffer("_output_head_grad_mask", None, persistent=False)
        if frozen_output_head_prefix_rows > 0:
            self.freeze_output_head_prefix_rows(frozen_output_head_prefix_rows)

    def freeze_output_head_prefix_rows(self, frozen_rows: int):
        """freeze [0:frozen_rows) rows in output head and keep remaining rows trainable."""
        vocab_size = self.output_head.weight.shape[0]
        if frozen_rows <= 0:
            self._output_head_grad_mask = None
            return
        if frozen_rows >= vocab_size:
            raise ValueError(
                f"frozen_rows must be in [0, vocab_size), got {frozen_rows} for vocab_size={vocab_size}"
            )
        mask = torch.ones((vocab_size, 1), dtype=self.output_head.weight.dtype, device=self.output_head.weight.device)
        mask[:frozen_rows] = 0
        self._output_head_grad_mask = mask
        self.output_head.weight.register_hook(self._mask_output_head_grad)

    def _mask_output_head_grad(self, grad: torch.Tensor) -> torch.Tensor:
        if grad is None or self._output_head_grad_mask is None:
            return grad
        return grad * self._output_head_grad_mask.to(dtype=grad.dtype)

    def embed_with_type(self, token_ids: torch.Tensor, type_ids: torch.Tensor) -> torch.Tensor:
        """compose token embeddings with additive type embeddings."""
        return self.token_embed(token_ids) + self.type_embed(type_ids)

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
    soft_embeds: torch.Tensor    # [B, S, H]
    hard_embeds: torch.Tensor    # [B, S, H]
    eos_logits: torch.Tensor     # [B, S]
    probs_subset: torch.Tensor   # [B, S, K+1] (concept + eos)
    actual_lengths: torch.Tensor # [B]

@dataclass
class PlannerOutput:
    per_type: List[PlannerTypeOutput]
    quota_mass_sum: torch.Tensor
    quota_count: torch.Tensor

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
            self.lambda_value.add_(self.eta * (bar_m.detach() - self.tau))
            self.lambda_value.clamp_(min=0.0)
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
) -> PlannerOutput:
    """generate one variable-length concept sequence per type from source input."""
    bsz, _ = input_ids.shape
    hidden_size = model.token_embed.embedding_dim
    dtype_embed = model.token_embed.weight.dtype

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
    logits_t = model.output_head(out.last_hidden_state[:, -1, :])  # [B, V]
    past_kv = out.past_key_values

    num_types = len(metas)
    ones_step = torch.ones((bsz, 1), device=device, dtype=torch.long)
    max_steps_by_type = torch.tensor([m.max_steps for m in metas], device=device, dtype=torch.long)
    eos_id_by_type = torch.tensor([m.eos_id for m in metas], device=device, dtype=torch.long)
    type_id_by_type = torch.tensor([m.type_id for m in metas], device=device, dtype=torch.long)

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
        model.token_embed.weight.index_select(0, m.concept_ids_with_eos) for m in metas
    ]  # list of [K_i+1, H]

    # ---------------------------------------------------------------------
    # 3) Allocate fixed-shape output buffers per type.
    # We fill these buffers asynchronously using (sample_idx, step_in_type).
    # ---------------------------------------------------------------------
    buffers: List[Dict[str, torch.Tensor]] = []
    for meta in metas:
        kplus = int(meta.concept_ids_with_eos.numel())
        buffers.append(
            {
                "token_ids": torch.full(
                    (bsz, meta.max_steps), meta.eos_id, device=device, dtype=torch.long
                ),
                "valid": torch.zeros((bsz, meta.max_steps), device=device, dtype=torch.long),
                "soft": torch.zeros(
                    (bsz, meta.max_steps, hidden_size), device=device, dtype=dtype_embed
                ),
                "hard": torch.zeros(
                    (bsz, meta.max_steps, hidden_size), device=device, dtype=dtype_embed
                ),
                "eos_logits": torch.zeros(
                    (bsz, meta.max_steps), device=device, dtype=torch.float32
                ),
                "probs": torch.zeros(
                    (bsz, meta.max_steps, kplus), device=device, dtype=torch.float32
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
        probs = gumbel_softmax_sample(masked_logits, tau=tau, hard=True)  # [B, V]
        sampled_ids = probs.argmax(dim=-1)  # [B]
        # Keep inactive rows deterministic.
        sampled_ids = torch.where(
            active,
            sampled_ids,
            torch.full_like(sampled_ids, int(eos_id_by_type[-1].item())),
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

            rows = row_mask.nonzero(as_tuple=False).squeeze(1)  # [N_t]
            local_step = step_before[rows]  # [N_t]
            eos_t = int(meta.eos_id)
            tok_t = sampled_ids[rows]  # [N_t]

            # Write current results into per-type buffers at async coordinates:
            #   buffer[type][sample_row, local_step]
            buffers[t]["token_ids"][rows, local_step] = tok_t
            buffers[t]["valid"][rows, local_step] = 1
            buffers[t]["eos_logits"][rows, local_step] = masked_logits[rows, eos_t]

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
            buffers[t]["probs"][rows, local_step] = probs_subset.float()

            # Add type embedding to both soft/hard token embeddings.
            type_vec_t = model.type_embed.weight[int(type_id_by_type[t].item())].view(1, -1)  # [1, H]
            soft_t = torch.matmul(probs_subset.to(concept_tables[t].dtype), concept_tables[t]) + type_vec_t
            hard_t = model.token_embed(tok_t) + type_vec_t
            buffers[t]["soft"][rows, local_step] = soft_t
            buffers[t]["hard"][rows, local_step] = hard_t

            soft_embed_step[rows] = soft_t
            hard_embed_step[rows] = hard_t
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
            hard_embed_step[~active] = model.token_embed(dummy_ids)
            soft_embed_step[~active] = hard_embed_step[~active]
            step_pos[~active, 0] = 0

        # Straight-through one-step input to advance shared KV cache.
        st_embed = hard_embed_step + (soft_embed_step - soft_embed_step.detach())  # [B, H]
        out_next = model.forward_backbone(
            inputs_embeds=st_embed.unsqueeze(1),  # [B, 1, H]
            attention_mask=ones_step,             # [B, 1]
            position_ids=step_pos,                # [B, 1]
            past_key_values=past_kv,
            use_cache=True,
        )
        logits_t = model.output_head(out_next.last_hidden_state[:, -1, :])  # [B, V]
        past_kv = out_next.past_key_values

    # ---------------------------------------------------------------------
    # 6) Pack per-type buffers back into PlannerOutput format.
    # ---------------------------------------------------------------------
    per_type_outputs: List[PlannerTypeOutput] = []
    for t, meta in enumerate(metas):
        token_ids = buffers[t]["token_ids"]
        valid_mask = buffers[t]["valid"]
        soft_embeds = buffers[t]["soft"]
        hard_embeds = buffers[t]["hard"]
        eos_logits = buffers[t]["eos_logits"]
        probs_subset = buffers[t]["probs"]
        actual_lengths = valid_mask.sum(dim=1)

        per_type_outputs.append(
            PlannerTypeOutput(
                token_ids=token_ids,
                valid_mask=valid_mask,
                soft_embeds=soft_embeds,
                hard_embeds=hard_embeds,
                eos_logits=eos_logits,
                probs_subset=probs_subset,
                actual_lengths=actual_lengths,
            )
        )

    return PlannerOutput(
        per_type=per_type_outputs,
        quota_mass_sum=quota_mass_sum,
        quota_count=quota_count,
    )

def build_executor_prefix(
    model: SharedBackboneUnifiedHead,
    *,
    planner_out: PlannerOutput,
    metas: List[ConceptTypeMeta],
    bos_id: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """build concept-only executor prefix with type ids, mask, and reset positions."""
    bsz = planner_out.per_type[0].token_ids.size(0)

    token_chunks: List[torch.Tensor] = []
    type_chunks: List[torch.Tensor] = []
    mask_chunks: List[torch.Tensor] = []
    pos_chunks: List[torch.Tensor] = []

    # Executor starts from BOS and does not see source text tokens.
    bos = torch.full((bsz, 1), bos_id, device=device, dtype=torch.long)
    token_chunks.append(bos)
    type_chunks.append(torch.full((bsz, 1), TYPE_ID_TEXT, device=device, dtype=torch.long))
    mask_chunks.append(torch.ones((bsz, 1), device=device, dtype=torch.long))
    pos_chunks.append(torch.zeros((bsz, 1), device=device, dtype=torch.long))

    for meta, type_out in zip(metas, planner_out.per_type):
        token_chunks.append(type_out.token_ids)
        type_chunks.append(torch.full_like(type_out.token_ids, meta.type_id))
        mask_chunks.append(type_out.valid_mask)
        # Critical rule: position indices restart from 1 for each type block.
        pos_chunks.append(
            torch.arange(1, meta.max_steps + 1, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(bsz, -1)
        )

    prefix_token_ids = torch.cat(token_chunks, dim=1)
    prefix_type_ids = torch.cat(type_chunks, dim=1)
    prefix_mask = torch.cat(mask_chunks, dim=1)
    prefix_pos = torch.cat(pos_chunks, dim=1)
    prefix_embeds = model.embed_with_type(prefix_token_ids, prefix_type_ids)
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

def compute_planner_losses(
    planner_out: PlannerOutput,
    metas: List[ConceptTypeMeta],
    src_lengths: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """aggregate planner-side regularizers for concept quality and budget control."""
    loss_commit = torch.zeros((), device=src_lengths.device, dtype=torch.float32)
    loss_unif = torch.zeros((), device=src_lengths.device, dtype=torch.float32)
    loss_eos = torch.zeros((), device=src_lengths.device, dtype=torch.float32)
    loss_len = torch.zeros((), device=src_lengths.device, dtype=torch.float32)

    for meta, type_out in zip(metas, planner_out.per_type):
        # 1) Commitment: keep hard concept picks close to soft distributions.
        valid = type_out.valid_mask.unsqueeze(-1).float()
        soft_v = type_out.soft_embeds * valid
        hard_v = type_out.hard_embeds * valid
        loss_commit = loss_commit + commitment_loss(soft_v, hard_v, beta=BETA_COMMIT)

        # 2) Usage uniformity: avoid collapsing to a small subset of concept IDs.
        concept_probs = type_out.probs_subset[:, :, :-1]  # remove EOS column
        weighted = concept_probs * type_out.valid_mask.unsqueeze(-1).float()
        hist = weighted.sum(dim=(0, 1))
        hist = hist / (hist.sum() + EPS)
        loss_unif = loss_unif + usage_kl_to_uniform(hist)

        # 3) EOS target: encourage stopping near a ratio-based expected concept length.
        expected = (src_lengths.float() * meta.target_ratio).long().clamp(
            min=1, max=meta.max_steps
        )
        eos_targets = torch.zeros_like(type_out.eos_logits)
        for b in range(eos_targets.size(0)):
            k = int(expected[b].item())
            eos_targets[b, k - 1 :] = 1.0
        eos_bce = F.binary_cross_entropy_with_logits(
            type_out.eos_logits.float(), eos_targets.float(), reduction="none"
        )
        valid_float = type_out.valid_mask.float()
        loss_eos = loss_eos + (eos_bce * valid_float).sum() / (valid_float.sum() + 1e-6)

        # 4) Length budget: penalize sequences longer than expected.
        loss_len = loss_len + F.relu(type_out.actual_lengths.float() - expected.float()).mean()

    n = max(1, len(metas))
    return loss_commit / n, loss_unif / n, loss_eos / n, loss_len / n

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
