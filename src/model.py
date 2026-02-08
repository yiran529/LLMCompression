from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig


def build_causal_mask_with_padding(
    attention_mask: torch.Tensor, dtype: torch.dtype, very_neg: float
) -> torch.Tensor:
    # attention_mask: [B, L], 1 for valid token, 0 for pad
    _, length = attention_mask.shape
    device = attention_mask.device
    causal = torch.tril(torch.ones((length, length), device=device, dtype=torch.bool))  # [L, L]
    valid = attention_mask.bool()
    allow = causal.unsqueeze(0)  # [1, L, L]
    allow = allow & valid.unsqueeze(1) & valid.unsqueeze(2)  # [B, L, L]
    mask = torch.zeros((attention_mask.size(0), length, length), device=device, dtype=dtype)
    mask = mask.masked_fill(~allow, very_neg)
    return mask.unsqueeze(1)  # [B, 1, L, L]


def build_middle_mask(concept_valid: torch.Tensor, dtype: torch.dtype, very_neg: float) -> torch.Tensor:
    # concept_valid: [B, Cmax], True for valid concept token, False for padding
    batch_size, cmax = concept_valid.shape
    device = concept_valid.device
    if cmax == 0:
        return torch.zeros((batch_size, 1, 0, 0), device=device, dtype=dtype)
    causal = torch.tril(torch.ones((cmax, cmax), device=device, dtype=torch.bool))  # [Cmax, Cmax]
    allow = causal.unsqueeze(0)  # [1, Cmax, Cmax]
    allow = allow & concept_valid.unsqueeze(1) & concept_valid.unsqueeze(2)  # [B, Cmax, Cmax]
    mask = torch.zeros((batch_size, cmax, cmax), device=device, dtype=dtype)
    mask = mask.masked_fill(~allow, very_neg)
    return mask.unsqueeze(1)  # [B, 1, Cmax, Cmax]


def commitment_loss(e_soft: torch.Tensor, e_hard: torch.Tensor, beta: float) -> torch.Tensor:
    loss1 = (e_soft.detach() - e_hard).pow(2).mean()
    loss2 = (e_soft - e_hard.detach()).pow(2).mean()
    return loss1 + beta * loss2


def usage_kl_to_uniform(hist: torch.Tensor, eps: float) -> torch.Tensor:
    num_bins = hist.numel()
    uniform = torch.full_like(hist, 1.0 / max(1, num_bins))
    return torch.sum(hist * (torch.log(hist + eps) - torch.log(uniform + eps)))


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
    def __init__(self, peft_model: nn.Module, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
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
        self.concept_vocab_size = cfg.concept_vocab_size
        self.null_id = cfg.concept_vocab_size

        self.concept_head = nn.Linear(self.hidden_size, self.concept_vocab_size + 1)
        self.concept_embeddings = nn.Embedding(self.concept_vocab_size, self.hidden_size)
        tail_mlp_hidden = self.hidden_size * cfg.tail_mlp_hidden_ratio
        self.tail_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, tail_mlp_hidden),
            nn.GELU(),
            nn.Linear(tail_mlp_hidden, self.hidden_size),
        ).to(dtype=self.embed_tokens.weight.dtype)

        n_layers = len(self.layers)
        if cfg.shallow_layers + cfg.middle_layers >= n_layers:
            raise ValueError(
                f"Invalid split: shallow({cfg.shallow_layers}) + middle({cfg.middle_layers}) "
                f"must be < total layers ({n_layers})."
            )
        self.shallow_start = 0
        self.shallow_end = cfg.shallow_layers
        self.middle_end = cfg.shallow_layers + cfg.middle_layers

        self.shallow_blocks = self.layers[self.shallow_start : self.shallow_end]
        self.middle_blocks = self.layers[self.shallow_end : self.middle_end]
        self.deep_blocks = self.layers[self.middle_end :]

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
        batch_size, _, hidden = concept_embeds_all.shape
        device = concept_embeds_all.device
        concept_counts = is_compressed.long().sum(dim=1)  # [B]
        cmax = int(concept_counts.max().item()) if batch_size > 0 else 0
        concept_padded = concept_embeds_all.new_zeros((batch_size, cmax, hidden))  # [B, Cmax, H]
        concept_pos = torch.zeros((batch_size, cmax), device=device, dtype=torch.long)  # [B, Cmax]
        concept_valid = torch.zeros((batch_size, cmax), device=device, dtype=torch.bool)  # [B, Cmax]

        if cmax == 0:
            return concept_padded, concept_pos, concept_valid, concept_counts

        for b in range(batch_size):
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
        batch_size, length = input_ids.shape
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
        pos_ids_full = torch.arange(length, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, length)
        mask_full = build_causal_mask_with_padding(attention_mask, x.dtype, self.cfg.very_neg)  # [B, 1, L, L]
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
        concept_embeds_all = concept_embeds_all * valid_mask.unsqueeze(-1).to(concept_embeds_all.dtype)

        # 3) Pack non-NULL concepts only, then run middle once on padded batch
        concept_in, concept_pos, concept_valid, concept_counts = self._pack_concepts(concept_embeds_all, is_compressed)
        # concept_in: [B, Cmax, H], concept_pos: [B, Cmax], concept_valid: [B, Cmax], concept_counts: [B]
        cmax = concept_in.size(1)
        if cmax > 0:
            middle_mask = build_middle_mask(concept_valid, concept_in.dtype, self.cfg.very_neg)  # [B, 1, Cmax, Cmax]
            concept_mid = self.run_blocks(concept_in, self.middle_blocks, middle_mask, concept_pos)  # [B, Cmax, H]
            concept_mid = concept_mid * concept_valid.unsqueeze(-1).to(concept_mid.dtype)  # [B, Cmax, H]
        else:
            concept_mid = concept_in

        # prefix statistics per time step
        compressed_cumsum = is_compressed.long().cumsum(dim=1)  # [B, L]
        pos_grid = torch.arange(length, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, length)
        comp_pos_grid = torch.where(is_compressed, pos_grid, torch.full_like(pos_grid, -1))  # [B, L]
        latest_comp = torch.cummax(comp_pos_grid, dim=1).values  # [B, L]

        # 4) Deep blocks: vectorize over all valid (b, t) pairs
        # We build one "deep sequence" for every training position t that has a target x[t+1].
        # Sequence layout follows v2: Z_t = [concept_prefix(<=t); tail(t)].
        time_steps = length - 1
        t_grid = torch.arange(time_steps, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, time_steps)
        active_bt = lengths.unsqueeze(1) > (t_grid + 1)  # [B, T], needs target x[t+1]
        active_b, active_t = torch.nonzero(active_bt, as_tuple=True)  # [N], [N]
        num_active = int(active_b.numel())

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

        smax = int(seq_lens.max().item())
        hidden = h_shallow.size(-1)
        s_idx = torch.arange(smax, device=device, dtype=torch.long).unsqueeze(0)  # [1, Smax]
        valid_pos = s_idx < seq_lens.unsqueeze(1)  # [N, Smax]
        is_concept = s_idx < c_n.unsqueeze(1)  # [N, Smax]

        # Map packed tail slots back to original shallow-token positions.
        tail_src = tail_start_n.unsqueeze(1) + (s_idx - c_n.unsqueeze(1))  # [N, Smax]
        tail_src_safe = tail_src.clamp(0, length - 1)
        tail_vals = h_shallow[active_b.unsqueeze(1), tail_src_safe]  # [N, Smax, H]
        # Learnable tail refinement before entering deep blocks (residual form).
        tail_vals = tail_vals + self.tail_mlp(tail_vals)

        if cmax > 0:
            # Gather concept prefix features/positions for each active sample.
            concept_src = s_idx.clamp(max=cmax - 1)  # [1, Smax]
            concept_vals = concept_mid[active_b.unsqueeze(1), concept_src]  # [N, Smax, H]
            concept_pos_vals = concept_pos[active_b.unsqueeze(1), concept_src]  # [N, Smax]
        else:
            # No concept tokens in this batch; concept region stays zero and masked out by valid_pos.
            concept_vals = h_shallow.new_zeros((num_active, smax, hidden))
            concept_pos_vals = torch.zeros((num_active, smax), device=device, dtype=torch.long)

        # Stitch Z_t: concept prefix first, then tail; keep absolute positions for RoPE.
        deep_in = torch.where(is_concept.unsqueeze(-1), concept_vals, tail_vals)  # [N, Smax, H]
        deep_in = deep_in * valid_pos.unsqueeze(-1).to(deep_in.dtype)
        deep_pos = torch.where(is_concept, concept_pos_vals, tail_src_safe)  # [N, Smax]
        deep_pos = deep_pos.masked_fill(~valid_pos, 0)

        q_idx = torch.arange(smax, device=device, dtype=torch.long).view(1, smax, 1)  # [1, Smax, 1]
        k_idx = torch.arange(smax, device=device, dtype=torch.long).view(1, 1, smax)  # [1, 1, Smax]
        seq_v = seq_lens.view(num_active, 1, 1)
        c_v = c_n.view(num_active, 1, 1)
        valid_q = q_idx < seq_v
        valid_k = k_idx < seq_v
        tail_q = q_idx >= c_v
        tail_k = k_idx >= c_v
        causal_tail = k_idx <= q_idx
        allow = valid_q & valid_k & ((~tail_k) | (tail_q & tail_k & causal_tail))  # [N, Smax, Smax]
        deep_mask = torch.zeros((num_active, smax, smax), device=device, dtype=deep_in.dtype)
        deep_mask = deep_mask.masked_fill(~allow, self.cfg.very_neg).unsqueeze(1)  # [N, 1, Smax, Smax]

        deep_h = self.run_blocks(deep_in, self.deep_blocks, deep_mask, deep_pos)  # [N, Smax, H]
        deep_h = self.final_norm(deep_h)  # [N, Smax, H]
        # Supervise with next-token target at each active timestep t.
        last_h = deep_h[torch.arange(num_active, device=device), last_index]  # [N, H]
        logits = self.lm_head(last_h)  # [N, V]
        targets = input_ids[active_b, active_t + 1]  # [N]
        loss_sum = F.cross_entropy(logits.float(), targets, reduction="sum")
        valid_targets = num_active

        loss_rec = loss_sum / max(1, valid_targets)

        # Aux-1: commitment on compressed (non-NULL) positions only.
        if bool(is_compressed.any().item()):
            e_soft_valid = e_soft[is_compressed]
            e_hard_valid = e_hard[is_compressed]
            loss_commit = commitment_loss(e_soft_valid, e_hard_valid, beta=self.cfg.beta_commit)
        else:
            loss_commit = torch.zeros((), device=device, dtype=loss_rec.dtype)

        # Aux-2: encourage concept-id usage closer to uniform.
        concept_mass = concept_weights_soft * valid_mask.unsqueeze(-1).to(concept_weights_soft.dtype)
        z_hist = concept_mass.sum(dim=(0, 1))
        z_hist = z_hist / (z_hist.sum() + self.cfg.eps)
        loss_unif = usage_kl_to_uniform(z_hist, self.cfg.eps)

        # Aux-3: length penalty on soft compression count (differentiable).
        p_non_null = 1.0 - z_soft[:, :, self.null_id]
        soft_concept_counts = (p_non_null * valid_mask.to(p_non_null.dtype)).sum(dim=1)  # [B]
        budget = (lengths.float() * self.cfg.compression_ratio).clamp(min=1.0)
        loss_len = F.relu(soft_concept_counts - budget).mean()

        loss = (
            self.cfg.lambda_rec * loss_rec
            + self.cfg.lambda_commit * loss_commit
            + self.cfg.lambda_unif * loss_unif
            + self.cfg.lambda_len * loss_len
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

    def infer_next_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        tau: float = 1.0,
        use_gumbel: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # input_ids: [B, L], attention_mask: [B, L]
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be rank-2 [B, L], got shape={tuple(input_ids.shape)}")
        if input_ids.size(1) < 1:
            raise ValueError("input_ids length must be >= 1 for inference.")

        batch_size, length = input_ids.shape
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, length), device=device, dtype=torch.long)
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"attention_mask shape must match input_ids, got {tuple(attention_mask.shape)} vs {tuple(input_ids.shape)}"
            )

        valid_mask = attention_mask.bool()
        lengths = attention_mask.sum(dim=1).to(torch.long)

        # 1) Shallow on all visible tokens
        x = self.embed_tokens(input_ids)
        pos_ids_full = torch.arange(length, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, length)
        mask_full = build_causal_mask_with_padding(attention_mask, x.dtype, self.cfg.very_neg)
        h_shallow = self.run_blocks(x, self.shallow_blocks, mask_full, pos_ids_full)

        # 2) Concept decision for current prefix (deterministic by default)
        concept_logits = self.concept_head(h_shallow)  # [B, L, K+1]
        if use_gumbel:
            z_soft = F.gumbel_softmax(concept_logits, tau=tau, hard=False, dim=-1)
            chosen_idx = torch.argmax(z_soft, dim=-1)
            concept_weights = z_soft[:, :, : self.concept_vocab_size]
            e_soft = concept_weights @ self.concept_embeddings.weight
            chosen_safe = chosen_idx.clamp_max(self.concept_vocab_size - 1)
            e_hard = self.concept_embeddings(chosen_safe)
            is_compressed = chosen_idx.ne(self.null_id) & valid_mask
            e_hard = e_hard * is_compressed.unsqueeze(-1).to(e_hard.dtype)
            concept_embeds_all = e_hard + (e_soft - e_soft.detach())
        else:
            chosen_idx = torch.argmax(concept_logits, dim=-1)
            is_compressed = chosen_idx.ne(self.null_id) & valid_mask
            chosen_safe = chosen_idx.clamp_max(self.concept_vocab_size - 1)
            concept_embeds_all = self.concept_embeddings(chosen_safe)
            concept_embeds_all = concept_embeds_all * is_compressed.unsqueeze(-1).to(concept_embeds_all.dtype)
        concept_embeds_all = concept_embeds_all * valid_mask.unsqueeze(-1).to(concept_embeds_all.dtype)

        # 3) Middle on packed concepts
        concept_in, concept_pos, concept_valid, concept_counts = self._pack_concepts(concept_embeds_all, is_compressed)
        cmax = concept_in.size(1)
        if cmax > 0:
            middle_mask = build_middle_mask(concept_valid, concept_in.dtype, self.cfg.very_neg)
            concept_mid = self.run_blocks(concept_in, self.middle_blocks, middle_mask, concept_pos)
            concept_mid = concept_mid * concept_valid.unsqueeze(-1).to(concept_mid.dtype)
        else:
            concept_mid = concept_in

        # 4) Build Z_t only for the current step t = current length - 1 for each sample
        vocab_size = int(self.lm_head.weight.size(0))
        next_logits = h_shallow.new_full((batch_size, vocab_size), self.cfg.very_neg)
        active_mask = lengths > 0
        active_b = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
        num_active = int(active_b.numel())
        if num_active == 0:
            stats = {
                "num_tokens": 0.0,
                "num_concepts": 0.0,
                "compress_ratio": 0.0,
            }
            return next_logits, stats

        active_t = lengths[active_b] - 1  # [N]
        compressed_cumsum = is_compressed.long().cumsum(dim=1)
        pos_grid = torch.arange(length, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, length)
        comp_pos_grid = torch.where(is_compressed, pos_grid, torch.full_like(pos_grid, -1))
        latest_comp = torch.cummax(comp_pos_grid, dim=1).values

        c_n = compressed_cumsum[active_b, active_t]  # [N]
        tau_n = latest_comp[active_b, active_t]  # [N]
        tail_start_n = tau_n + 1
        tail_len_n = active_t - tail_start_n + 1
        seq_lens = c_n + tail_len_n
        last_index = seq_lens - 1

        smax = int(seq_lens.max().item())
        hidden = h_shallow.size(-1)
        s_idx = torch.arange(smax, device=device, dtype=torch.long).unsqueeze(0)  # [1, Smax]
        valid_pos = s_idx < seq_lens.unsqueeze(1)
        is_concept = s_idx < c_n.unsqueeze(1)

        tail_src = tail_start_n.unsqueeze(1) + (s_idx - c_n.unsqueeze(1))
        tail_src_safe = tail_src.clamp(0, length - 1)
        tail_vals = h_shallow[active_b.unsqueeze(1), tail_src_safe]
        tail_vals = tail_vals + self.tail_mlp(tail_vals)

        if cmax > 0:
            concept_src = s_idx.clamp(max=cmax - 1)
            concept_vals = concept_mid[active_b.unsqueeze(1), concept_src]
            concept_pos_vals = concept_pos[active_b.unsqueeze(1), concept_src]
        else:
            concept_vals = h_shallow.new_zeros((num_active, smax, hidden))
            concept_pos_vals = torch.zeros((num_active, smax), device=device, dtype=torch.long)

        deep_in = torch.where(is_concept.unsqueeze(-1), concept_vals, tail_vals)
        deep_in = deep_in * valid_pos.unsqueeze(-1).to(deep_in.dtype)
        deep_pos = torch.where(is_concept, concept_pos_vals, tail_src_safe)
        deep_pos = deep_pos.masked_fill(~valid_pos, 0)

        q_idx = torch.arange(smax, device=device, dtype=torch.long).view(1, smax, 1)
        k_idx = torch.arange(smax, device=device, dtype=torch.long).view(1, 1, smax)
        seq_v = seq_lens.view(num_active, 1, 1)
        c_v = c_n.view(num_active, 1, 1)
        valid_q = q_idx < seq_v
        valid_k = k_idx < seq_v
        tail_q = q_idx >= c_v
        tail_k = k_idx >= c_v
        causal_tail = k_idx <= q_idx
        allow = valid_q & valid_k & ((~tail_k) | (tail_q & tail_k & causal_tail))
        deep_mask = torch.zeros((num_active, smax, smax), device=device, dtype=deep_in.dtype)
        deep_mask = deep_mask.masked_fill(~allow, self.cfg.very_neg).unsqueeze(1)

        deep_h = self.run_blocks(deep_in, self.deep_blocks, deep_mask, deep_pos)
        deep_h = self.final_norm(deep_h)
        last_h = deep_h[torch.arange(num_active, device=device), last_index]
        active_logits = self.lm_head(last_h)
        next_logits[active_b] = active_logits

        total_tokens = float(lengths.sum().item())
        total_concepts = float(concept_counts.sum().item())
        stats = {
            "num_tokens": total_tokens,
            "num_concepts": total_concepts,
            "compress_ratio": total_concepts / max(1.0, total_tokens),
        }
        return next_logits, stats

    @staticmethod
    def _sample_from_logits(
        logits: torch.Tensor,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        if (not do_sample) or temperature <= 0.0:
            return torch.argmax(logits, dim=-1)

        scores = logits.float() / max(temperature, 1e-5)
        vocab_size = scores.size(-1)

        if top_k > 0:
            k = min(int(top_k), vocab_size)
            topk_vals, _ = torch.topk(scores, k=k, dim=-1)
            kth = topk_vals[:, -1].unsqueeze(-1)
            scores = scores.masked_fill(scores < kth, float("-inf"))

        if 0.0 < top_p < 1.0:
            sorted_scores, sorted_idx = torch.sort(scores, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_scores, dim=-1)
            cum_probs = sorted_probs.cumsum(dim=-1)
            remove = cum_probs > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
            filtered = torch.full_like(scores, float("-inf"))
            filtered.scatter_(1, sorted_idx, sorted_scores)
            scores = filtered

        probs = F.softmax(scores, dim=-1)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 32,
        tau: float = 1.0,
        use_gumbel: bool = False,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be rank-2 [B, L], got shape={tuple(input_ids.shape)}")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0.")

        batch_size = input_ids.size(0)
        device = input_ids.device
        generated = input_ids.clone()
        if attention_mask is None:
            gen_mask = torch.ones_like(generated, dtype=torch.long, device=device)
        else:
            if attention_mask.shape != input_ids.shape:
                raise ValueError(
                    f"attention_mask shape must match input_ids, got {tuple(attention_mask.shape)} vs {tuple(input_ids.shape)}"
                )
            gen_mask = attention_mask.clone().to(device=device, dtype=torch.long)

        finished = torch.zeros((batch_size,), device=device, dtype=torch.bool)
        if eos_token_id is not None:
            lengths = gen_mask.sum(dim=1).to(torch.long)
            has_token = lengths > 0
            if bool(has_token.any().item()):
                row = torch.nonzero(has_token, as_tuple=False).squeeze(-1)
                col = lengths[row] - 1
                last_tok = generated[row, col]
                finished[row] = last_tok.eq(int(eos_token_id))

        for _ in range(max_new_tokens):
            step_logits, _ = self.infer_next_logits(
                input_ids=generated,
                attention_mask=gen_mask,
                tau=tau,
                use_gumbel=use_gumbel,
            )
            next_tokens = self._sample_from_logits(
                logits=step_logits,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            prev_finished = finished
            if eos_token_id is not None:
                fill_id = int(eos_token_id if pad_token_id is None else pad_token_id)
                fill = torch.full_like(next_tokens, fill_id)
                next_tokens = torch.where(prev_finished, fill, next_tokens)
                next_mask = (~prev_finished).to(dtype=gen_mask.dtype)
                just_eos = next_tokens.eq(int(eos_token_id)) & (~prev_finished)
                finished = prev_finished | just_eos
            else:
                next_mask = torch.ones_like(next_tokens, dtype=gen_mask.dtype)

            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
            gen_mask = torch.cat([gen_mask, next_mask.unsqueeze(1)], dim=1)

            if eos_token_id is not None and bool(finished.all().item()):
                break

        return generated, gen_mask
