# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


from src.config.train_config import BETA_COMMIT, EPS, TYPE_ID_CONCEPT, TYPE_ID_TEXT, ConceptConfig


def resolve_qwen3_special_token_ids(tokenizer: AutoTokenizer) -> Tuple[int, int, int]:
    """Resolve BOS/EOS/PAD ids for Qwen3 using explicit token strings."""
    bos_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    pad_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    assert bos_id is not None and eos_id is not None and pad_id is not None, (
        "Tokenizer is missing required Qwen3 special tokens.")

    return int(bos_id), int(eos_id), int(pad_id)

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


def _detach_past_key_values(past_key_values):
    """Detach KV cache tensors so later planner steps cannot backprop into earlier steps."""
    if isinstance(past_key_values, tuple):
        detached_layers = []
        for layer in past_key_values:
            if isinstance(layer, tuple):
                detached_layers.append(tuple(x.detach() if torch.is_tensor(x) else x for x in layer))
            else:
                detached_layers.append(layer.detach() if torch.is_tensor(layer) else layer)
        return tuple(detached_layers)
    return past_key_values


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


def _run_kmeans(weight: torch.Tensor, num_clusters: int, num_iters: int = 20) -> torch.Tensor:
    """Run lightweight K-means on embedding rows and return centers [K, H]."""
    n_rows = int(weight.size(0))
    assert n_rows >= num_clusters > 0, f"Invalid num_clusters={num_clusters} for n_rows={n_rows}."

    data = weight.float()
    device = data.device

    init_ids = torch.randperm(n_rows, device=device)[:num_clusters]
    centers = data.index_select(0, init_ids).clone()

    for _ in range(max(1, int(num_iters))):
        dists = torch.cdist(data, centers, p=2)  # [N, K]
        labels = torch.argmin(dists, dim=1)      # [N]

        new_centers = torch.zeros_like(centers)
        counts = torch.bincount(labels, minlength=num_clusters)
        new_centers.index_add_(0, labels, data)

        non_empty = counts > 0
        if torch.any(non_empty):
            new_centers[non_empty] = new_centers[non_empty] / counts[non_empty].unsqueeze(1).to(data.dtype)
        if torch.any(~non_empty):
            refill_ids = torch.randperm(n_rows, device=device)[: int((~non_empty).sum().item())]
            new_centers[~non_empty] = data.index_select(0, refill_ids)

        if torch.allclose(new_centers, centers, atol=1e-5, rtol=1e-4):
            centers = new_centers
            break
        centers = new_centers

    return centers


def _kmeans_initialize_new_token_rows(
    *,
    input_embed: nn.Module,
    output_embed: nn.Module,
    meta: "TokenMeta",
    base_vocab_size: int,
    num_iters: int = 20,
) -> None:
    """Initialize new token rows using K-means centers from old vocabulary rows."""
    concept_ids = meta.concept_ids.to(dtype=torch.long, device=input_embed.weight.device)
    k = int(concept_ids.numel())

    with torch.no_grad():
        # Use only original valid vocab for K-means (exclude padding rows if any)
        kmeans_vocab_size = min(base_vocab_size, meta.original_vocab_size)
        src_in_base = input_embed.weight[:kmeans_vocab_size]
        src_out_base = output_embed.weight[:kmeans_vocab_size]

        in_centers = _run_kmeans(src_in_base, num_clusters=k, num_iters=num_iters).to(dtype=input_embed.weight.dtype)
        out_centers = _run_kmeans(src_out_base, num_clusters=k, num_iters=num_iters).to(dtype=output_embed.weight.dtype)

        # Concept rows come from K-means cluster centers.
        input_embed.weight.index_copy_(0, concept_ids, in_centers)
        output_embed.weight.index_copy_(0, concept_ids.to(output_embed.weight.device), out_centers)

        # Keep control tokens anchored to known semantic rows.
        input_embed.weight[meta.plan_token_id].copy_(input_embed.weight[meta.bos_id])
        input_embed.weight[meta.exec_token_id].copy_(input_embed.weight[meta.bos_id])
        input_embed.weight[meta.concept_eos_id].copy_(input_embed.weight[meta.eos_id])
        output_embed.weight[meta.plan_token_id].copy_(output_embed.weight[meta.bos_id])
        output_embed.weight[meta.exec_token_id].copy_(output_embed.weight[meta.bos_id])
        output_embed.weight[meta.concept_eos_id].copy_(output_embed.weight[meta.eos_id])

@dataclass
class TokenMeta:
    """统一管理所有特殊token ID，避免混淆"""
    
    # ===== Base tokenizer tokens =====
    bos_id: int                        # 句子开始，如 <s>
    eos_id: int                        # 句子结束，如 </s>
    pad_id: int                        # padding token
    
    # ===== Planning system tokens =====
    plan_token_id: int                 # <PLAN> - 触发概念生成
    exec_token_id: int                 # <EXEC> - executor 前缀起始
    concept_eos_id: int                # <EOS_CONCEPT> - 概念序列结束
    
    # ===== Concept vocabulary =====
    concept_ids: torch.Tensor          # [K] - <C_0>, <C_1>, ..., <C_{K-1}>
    concept_ids_with_eos: torch.Tensor # [K+1] - includes <EOS_CONCEPT>
    
    # ===== Type embeddings =====
    type_id_text: int                  # TYPE_ID_TEXT (0)
    type_id_concept: int               # TYPE_ID_CONCEPT (1)
    
    # ===== Training config =====
    max_concept_steps: int             # 最大概念序列长度
    target_concept_ratio: float        # 目标压缩比
    
    # ===== Vocab sizes =====
    original_vocab_size: int           # 原始有效词汇大小（用于K-means，不含padding）
    
    # ===== Derived properties =====
    @property
    def new_rows(self) -> int:
        """模型需要新增的token embedding行数"""
        # 注意: 这里的逻辑假设新增的特殊 token 是直接追加在原本 vocab 之后的
        # <PLAN> (1) + <EXEC> (1) + <EOS_CONCEPT> (1) + K个 concept = K + 3
        return int(self.concept_ids.numel()) + 3
    
    @property
    def planner_vocab_size(self) -> int:
        """planner输出头大小 (K+1)"""
        return int(self.concept_ids_with_eos.numel())
    
    @property
    def concept_eos_local_id(self) -> int:
        """concept EOS在planner词表中的local位置 (应该是最后一个: K)"""
        return int(self.concept_ids_with_eos.numel() - 1)
    
    def concept_local_to_global(self, local_ids: torch.Tensor) -> torch.Tensor:
        """将planner local ids (0..K) 转换为全局 vocab ids
        Args:
            local_ids: [B] or [B, S] 等任意形状的 local concept vocab indices (0..K)
        Returns:
            global_ids: 相同形状，全局词表索引
        """
        # Assert: local_ids 必须在 [0, K] 范围内
        assert torch.all(local_ids >= 0) and torch.all(local_ids < self.planner_vocab_size), (
            f"local_ids out of range: got [{local_ids.min().item()}, {local_ids.max().item()}], "
            f"expected [0, {self.planner_vocab_size - 1}]"
        )
        
        original_shape = local_ids.shape
        flat_ids = local_ids.reshape(-1)
        global_ids = self.concept_ids_with_eos.index_select(0, flat_ids)
        return global_ids.view(original_shape)


class SharedBackboneUnifiedHead(nn.Module):
    def __init__(
        self,
        base_model: AutoModelForCausalLM,
        num_type_embeddings: int,
        frozen_output_head_prefix_rows: int = 0,
        meta: Optional[TokenMeta] = None,
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
        assert frozen_output_head_prefix_rows >= 0, "frozen_output_head_prefix_rows must be non-negative"
        assert frozen_output_head_prefix_rows <= vocab_size, "frozen_output_head_prefix_rows cannot exceed vocab size"
        base_vocab_size = frozen_output_head_prefix_rows
        new_rows = meta.new_rows
        planner_rows = meta.planner_vocab_size
        assert vocab_size == base_vocab_size + new_rows, f"Vocab shape mismatch: vocab_size={vocab_size}, base_vocab_size={base_vocab_size}, new_rows={new_rows}."

        # Hard-apply K-means init to base model embedding tables before copying slices.
        _kmeans_initialize_new_token_rows(
            input_embed=input_embed,
            output_embed=out_embed,
            meta=meta,
            base_vocab_size=base_vocab_size,
            num_iters=20,
        )

        # Step 2) Build split token embeddings: frozen base rows + trainable new rows.
        token_dtype = input_embed.weight.dtype
        self.token_embed_base = nn.Embedding(base_vocab_size, hidden_size).to(dtype=token_dtype)
        self.token_embed_new = nn.Embedding(new_rows, hidden_size).to(dtype=token_dtype) 
            

        # Step 3) Initialize token embeddings from model input embeddings.
        src_input_weight = input_embed.weight.data
        self.token_embed_base.weight.data.copy_(src_input_weight[:base_vocab_size])
        self.token_embed_base.weight.requires_grad = False
        self.token_embed_new.weight.data.copy_(src_input_weight[base_vocab_size:])

        # Step 4) Build split output head: frozen base rows + trainable new rows.
        self.output_head_base = nn.Linear(hidden_size, base_vocab_size, bias=False)
        self.output_head_new = nn.Linear(hidden_size, planner_rows, bias=False)
        self.base_vocab_size = base_vocab_size
        self.vocab_size = vocab_size
        self.type_embed = nn.Embedding(num_type_embeddings, hidden_size)
        # Keep type embeddings in the same dtype as token embeddings (e.g. fp16 on GPU).
        self.type_embed.to(dtype=token_dtype)
        # Step 5) Keep output heads in the same dtype as token embeddings for speed.
        head_dtype = token_dtype
        self.output_head_base.to(dtype=head_dtype)
        self.output_head_new.to(dtype=head_dtype)

        # Step 6) Initialize head weights strictly from model output embeddings.
        assert out_embed.weight.shape[0] == vocab_size, "output embeddings must exist and match vocab size for proper initialization."
        src_weight = out_embed.weight.detach()
        planner_token_ids = meta.concept_ids_with_eos.to(device=src_weight.device, dtype=torch.long)
        self.output_head_base.weight.data.copy_(src_weight[:base_vocab_size])
        self.output_head_new.weight.data.copy_(src_weight.index_select(0, planner_token_ids))

        nn.init.zeros_(self.type_embed.weight)
        # Freeze base output head rows.
        for p in self.output_head_base.parameters():
            p.requires_grad = False

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """embed token ids with frozen base rows + trainable new rows."""
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
        return self.token_embed_new.weight

    def executor_forward_head(self, hidden: torch.Tensor) -> torch.Tensor:
        """compute executor logits with base head only."""
        return self.output_head_base(hidden)

    def planner_forward_head(self, hidden: torch.Tensor) -> torch.Tensor:
        """compute planner logits on planner-only vocabulary (K+1)."""
        return self.output_head_new(hidden)

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

def build_concept_special_tokens(cfg: ConceptConfig) -> List[str]:
    """define planner-side special tokens for single concept vocabulary."""
    special_tokens: List[str] = ["<PLAN>", "<EXEC>"]
    special_tokens.append("<EOS_CONCEPT>")
    for k in range(cfg.size):
        special_tokens.append(f"<C_{k}>")
    return special_tokens

def build_token_meta(
    tokenizer: AutoTokenizer,
    cfg: ConceptConfig,
    device: str,
    original_vocab_size: int = None,
) -> TokenMeta:
    """统一构建所有token ID元数据"""
    vocab_size = len(tokenizer)
    if original_vocab_size is None:
        original_vocab_size = vocab_size
    
    # Qwen3 官方 tokenizer_config: bos=null, eos=<|im_end|>, pad=<|endoftext|>.
    bos_id, eos_id, pad_id = resolve_qwen3_special_token_ids(tokenizer)
    plan_token_id = tokenizer.convert_tokens_to_ids("<PLAN>")
    exec_token_id = tokenizer.convert_tokens_to_ids("<EXEC>")
    concept_eos_id = tokenizer.convert_tokens_to_ids("<EOS_CONCEPT>")
    
    # Assert: 所有基础 token IDs 必须有效
    assert bos_id is not None and 0 <= bos_id < vocab_size, f"Invalid bos_id: {bos_id}"
    assert eos_id is not None and 0 <= eos_id < vocab_size, f"Invalid eos_id: {eos_id}"
    assert pad_id is not None and 0 <= pad_id < vocab_size, f"Invalid pad_id: {pad_id}"
    assert plan_token_id is not None and 0 <= plan_token_id < vocab_size, (
        f"Invalid plan_token_id: {plan_token_id}. Ensure <PLAN> token is added to tokenizer."
    )
    assert exec_token_id is not None and 0 <= exec_token_id < vocab_size, (
        f"Invalid exec_token_id: {exec_token_id}. Ensure <EXEC> token is added to tokenizer."
    )
    assert concept_eos_id is not None and 0 <= concept_eos_id < vocab_size, (
        f"Invalid concept_eos_id: {concept_eos_id}. Ensure <EOS_CONCEPT> token is added to tokenizer."
    )
    
    # 获取 concept tokens
    concept_tokens = [f"<C_{k}>" for k in range(cfg.size)]
    concept_ids = tokenizer.convert_tokens_to_ids(concept_tokens)
    
    # Assert: 所有 concept IDs 必须有效
    assert all(cid is not None and 0 <= cid < vocab_size for cid in concept_ids), (
        f"Invalid concept_ids: {concept_ids}. Ensure all <C_k> tokens are added to tokenizer."
    )
    
    concept_ids_t = torch.tensor(concept_ids, device=device, dtype=torch.long)
    concept_ids_eos = torch.cat(
        [concept_ids_t, torch.tensor([concept_eos_id], device=device, dtype=torch.long)], dim=0
    )
    
    # Assert: concept_eos_id 必须在最后位置
    assert int(concept_ids_eos[-1].item()) == concept_eos_id, (
        f"Internal error: concept_ids_with_eos must end with concept_eos_id. "
        f"Got {concept_ids_eos[-1].item()}, expected {concept_eos_id}."
    )
    
    return TokenMeta(
        bos_id=int(bos_id),
        eos_id=int(eos_id),
        pad_id=int(pad_id),
        plan_token_id=int(plan_token_id),
        exec_token_id=int(exec_token_id),
        concept_eos_id=int(concept_eos_id),
        concept_ids=concept_ids_t,
        concept_ids_with_eos=concept_ids_eos,
        type_id_text=TYPE_ID_TEXT,
        type_id_concept=TYPE_ID_CONCEPT,
        max_concept_steps=cfg.max_steps,
        target_concept_ratio=cfg.target_ratio,
        original_vocab_size=int(original_vocab_size),
    )

def plan_concepts(
    model: SharedBackboneUnifiedHead,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    meta: TokenMeta,
    tau: float,
    min_concept_steps: int,
    device: str,
    sampling_mode: str = "gumbel",
    mix_greedy_ratio: float = 0.0,
    use_cache: bool = True,
) -> PlannerOutput:
    """generate one variable-length concept sequence from source input."""
    # =========================================================================
    # 1. 基础变量初始化与提取
    # =========================================================================
    bsz, _ = input_ids.shape
    hidden_size = model.hidden_size
    dtype_embed = model.token_embed_base.weight.dtype

    # 从 meta 获取所需的 token IDs
    bos_id = meta.bos_id
    plan_token_id = meta.plan_token_id

    # =========================================================================
    # 2. 构造 Planner 的输入序列：[BOS] + Source Text + [EOS] + [PLAN]
    # =========================================================================
    bos_col = torch.full((bsz, 1), bos_id, device=device, dtype=torch.long)
    eos_col = torch.full((bsz, 1), meta.eos_id, device=device, dtype=torch.long)
    plan_col = torch.full((bsz, 1), plan_token_id, device=device, dtype=torch.long)
    planner_input_ids = torch.cat([bos_col, input_ids, eos_col, plan_col], dim=1)
    planner_type_ids = torch.full_like(planner_input_ids, TYPE_ID_TEXT)
    planner_mask = torch.cat(
        [
            torch.ones((bsz, 1), device=device, dtype=torch.long),
            attention_mask,
            torch.ones((bsz, 1), device=device, dtype=torch.long),
            torch.ones((bsz, 1), device=device, dtype=torch.long),
        ],
        dim=1,
    )
    planner_pos = torch.cumsum(planner_mask, dim=1) - 1
    planner_pos = planner_pos.clamp(min=0)

    # =========================================================================
    # 3. 初始前向传播 (Backbone)：获取 <PLAN> token 的隐状态与初始 KV Cache
    # =========================================================================
    planner_embeds = model.embed_with_type(planner_input_ids, planner_type_ids)
    out = model.forward_backbone(
        inputs_embeds=planner_embeds,
        attention_mask=planner_mask,
        position_ids=planner_pos,
        use_cache=use_cache,
    )
    
    # 从 meta 获取 planner 词表相关信息
    eos_id = int(meta.concept_eos_id)
    eos_local_id = meta.concept_eos_local_id

    logits_t = model.planner_forward_head(out.last_hidden_state[:, -1, :])  # [B, K+1]
    assert logits_t.size(1) == meta.planner_vocab_size, (
        f"Planner head output dim mismatch: got {logits_t.size(1)}, expected {meta.planner_vocab_size}"
    )
    past_kv = out.past_key_values

    # =========================================================================
    # 4. 计算目标生成长度与初始化各种 Loss 统计量
    # =========================================================================
    src_lengths = attention_mask.sum(dim=1).to(torch.long)
    expected = (src_lengths.float() * meta.target_concept_ratio).long().clamp(min=1, max=meta.max_concept_steps)

    commit_sum1 = torch.zeros((), device=device, dtype=torch.float32)
    commit_sum2 = torch.zeros((), device=device, dtype=torch.float32)
    hist_sum = torch.zeros((int(meta.concept_ids.numel()),), device=device, dtype=torch.float32)
    eos_sum = torch.zeros((), device=device, dtype=torch.float32)
    eos_count = torch.zeros((), device=device, dtype=torch.float32)

    # =========================================================================
    # 5. 初始化自回归 (AR) 生成所需的状态容器
    # =========================================================================
    concept_table = model.embed_tokens(meta.concept_ids_with_eos)  # [K+1, H]
    type_vec = model.type_embed.weight[int(meta.type_id_concept)].view(1, -1).to(dtype=dtype_embed)

    concept_tokens = torch.full((bsz, meta.max_concept_steps), eos_id, device=device, dtype=torch.long)
    concept_valid = torch.zeros((bsz, meta.max_concept_steps), device=device, dtype=torch.long)
    concept_st_embeds = torch.zeros((bsz, meta.max_concept_steps, hidden_size), device=device, dtype=dtype_embed)
    finished = torch.zeros((bsz,), device=device, dtype=torch.bool)

    cache_attention_mask = planner_mask
    cache_position_ids = planner_pos[:, -1:].clone() + 1

    # =========================================================================
    # 6. 自回归循环：逐步生成 Concept Tokens
    # =========================================================================
    for step in range(meta.max_concept_steps):
        active = ~finished
        if not torch.any(active):
            break

        # =====================================================================
        # 6.1 处理边界条件：屏蔽无效 Token 并强制控制输出长度
        # =====================================================================
        masked_logits = logits_t.float().clone()
        if torch.any(~active):
            masked_logits[~active] = -1e4
            masked_logits[~active, eos_local_id] = 0.0

        if min_concept_steps > 1 and step < (min_concept_steps - 1):
            masked_logits[active, eos_local_id] = -1e4
        if step >= (meta.max_concept_steps - 1):
            forced = torch.full_like(masked_logits[active], -1e4)
            forced[:, eos_local_id] = 0.0
            masked_logits[active] = forced

        # =====================================================================
        # 6.2 Token 采样 (Gumbel-Softmax / Greedy / Mix)
        # =====================================================================
        probs, sampled_ids_local = _select_planner_tokens(
            masked_logits=masked_logits,
            tau=tau,
            sampling_mode=sampling_mode,
            mix_greedy_ratio=mix_greedy_ratio,
        )

        assert torch.all(sampled_ids_local >= 0) and torch.all(sampled_ids_local < meta.planner_vocab_size), (
            f"sampled_ids_local out of range at step {step}: "
            f"got [{sampled_ids_local.min().item()}, {sampled_ids_local.max().item()}], "
            f"expected [0, {meta.planner_vocab_size - 1}]"
        )

        sampled_ids = meta.concept_local_to_global(sampled_ids_local)
        sampled_ids = torch.where(active, sampled_ids, torch.full_like(sampled_ids, eos_id))

        concept_tokens[active, step] = sampled_ids[active]
        concept_valid[active, step] = 1

        # =====================================================================
        # 6.3 统计与计算当前步的 Loss (CommitLoss, 词频KL, EOS Loss等)
        # =====================================================================
        active_rows = active.nonzero(as_tuple=False).squeeze(1)
        probs_subset = probs[active_rows]  # [N, K+1]
        hist_sum = hist_sum + probs_subset[:, :-1].sum(dim=0)

        soft_t = torch.matmul(probs_subset.to(concept_table.dtype), concept_table).to(dtype=dtype_embed) + type_vec
        hard_t = model.embed_tokens(sampled_ids[active_rows]).to(dtype=dtype_embed) + type_vec
        st_t = hard_t + (soft_t - soft_t.detach())

        soft_f = soft_t.float()
        hard_f = hard_t.float()
        commit_sum1 = commit_sum1 + (soft_f.detach() - hard_f).pow(2).sum()
        commit_sum2 = commit_sum2 + (soft_f - hard_f.detach()).pow(2).sum()

        eos_target = (step >= (expected[active_rows] - 1)).float()
        eos_logit = masked_logits[active_rows, eos_local_id]
        eos_sum = eos_sum + F.binary_cross_entropy_with_logits(
            eos_logit.float(), eos_target, reduction="sum"
        )
        eos_count = eos_count + eos_target.numel()

        # =====================================================================
        # 6.4 构造用于下一步自回归的 Straight-Through (ST) 嵌入向量
        # =====================================================================
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
        
        # =====================================================================
        # 6.5 执行下一次前向传播，更新 KV Cache 与 Logits
        # =====================================================================
        cache_attention_mask = torch.cat(
            [cache_attention_mask, torch.ones((bsz, 1), device=device, dtype=torch.long)],
            dim=1
        )
        cache_position_ids = cache_position_ids + 1

        out_next = model.forward_backbone(
            # Stop cross-step gradient: step t+1 cannot update step t concept embeddings.
            inputs_embeds=st_embed.detach().unsqueeze(1),  # [B, 1, H]
            attention_mask=cache_attention_mask,  # [B, T_curr]
            position_ids=cache_position_ids,      # [B, 1]
            past_key_values=_detach_past_key_values(past_kv),
            use_cache=use_cache,
        )
        logits_t = model.planner_forward_head(out_next.last_hidden_state[:, -1, :])  # [B, K+1]
        past_kv = _detach_past_key_values(out_next.past_key_values)

    # =========================================================================
    # 7. 汇总输出结果并合并计算全局 Loss
    # =========================================================================
    actual_lengths = concept_valid.sum(dim=1)
    concept_out = PlannerTypeOutput(
        token_ids=concept_tokens,
        valid_mask=concept_valid,
        actual_lengths=actual_lengths,
        st_embeds=concept_st_embeds,
    )

    denom = max(1.0, float(bsz * meta.max_concept_steps * hidden_size))
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
    meta: TokenMeta,
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

    # Block 0: EXEC control token.
    # Executor starts from <EXEC>, then consumes concept prefix and later text BOS.
    exec_tok = torch.full((bsz, 1), meta.exec_token_id, device=device, dtype=torch.long)
    exec_type = torch.full((bsz, 1), TYPE_ID_TEXT, device=device, dtype=torch.long)
    token_chunks.append(exec_tok)
    type_chunks.append(exec_type)
    mask_chunks.append(torch.ones((bsz, 1), device=device, dtype=torch.long))
    embed_chunks.append(model.embed_with_type(exec_tok, exec_type))

    type_out = planner_out.concept
    type_lens = type_out.valid_mask.sum(dim=1).to(torch.long)
    max_len = int(type_lens.max().item())

    assert max_len > 0, "Planner did not generate any valid concept tokens, cannot build executor prefix."
    tok = type_out.token_ids[:, :max_len]
    msk = type_out.valid_mask[:, :max_len]
    typ = torch.full_like(tok, meta.type_id_concept)
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

def build_executor_blocklist(
    meta: TokenMeta,
) -> List[int]:
    """return token IDs that executor logits must never generate.
    
    Note: This function is kept for backward compatibility.
    """
    blocked: List[int] = [meta.plan_token_id]
    blocked.append(meta.exec_token_id)
    blocked.append(meta.concept_eos_id)
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
    meta: TokenMeta,
    device: str,
    max_new_tokens: int = 64,
    planner_tau: float = 0.2,
    min_concept_steps: int = 1,
) -> ExecutorInferenceOutput:
    """run planner->prefix->executor greedy decoding for inference."""
    bsz = input_ids.size(0)
    max_new_tokens = max(0, int(max_new_tokens))

    # 从 meta 获取 token IDs
    bos_id = meta.bos_id
    eos_id = meta.eos_id

    was_training = model.training
    model.eval()
    try:
        planner_out, *_ = plan_concepts(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            meta=meta,
            tau=planner_tau,
            sampling_mode="greedy",
            min_concept_steps=min_concept_steps,
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
        logits_t = model.executor_forward_head(out.last_hidden_state[:, -1, :]).float()  # [B, V_base]
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
            logits_t = model.executor_forward_head(out.last_hidden_state[:, -1, :]).float()
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
