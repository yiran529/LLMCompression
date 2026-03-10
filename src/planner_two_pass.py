# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from src.config.train_config import BETA_COMMIT
from src.model import (
    NEG_INF_LOGIT,
    PlannerOutput,
    PlannerTypeOutput,
    SharedBackboneUnifiedHead,
    TokenMeta,
    _detach_past_key_values,
    usage_kl_to_uniform,
)


@dataclass
class PlannerSamplingTrace:
    gumbel_noise: Optional[torch.Tensor] = None     # [B, S, Vp]
    mix_use_greedy: Optional[torch.Tensor] = None   # [B, S]


def _normalize_probs(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    return probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)


def _gumbel_soft_probs_from_noise(
    masked_logits: torch.Tensor,
    tau: float,
    gumbel_noise: torch.Tensor,
) -> torch.Tensor:
    y = (masked_logits + gumbel_noise) / max(float(tau), 1e-4)
    y = y - y.max(dim=-1, keepdim=True).values
    y = torch.clamp(y, min=-50, max=50)
    probs = F.softmax(y, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    return probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)


def _sample_planner_tokens_with_trace(
    *,
    masked_logits: torch.Tensor,
    tau: float,
    sampling_mode: str,
    mix_greedy_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if sampling_mode == "gumbel":
        u = torch.empty_like(masked_logits).uniform_(1e-8, 1 - 1e-8)
        g = -torch.log(-torch.log(u + 1e-10) + 1e-10)
        probs = _gumbel_soft_probs_from_noise(masked_logits, tau, g)
        sampled_ids = probs.argmax(dim=-1)
        return probs, sampled_ids, g, None

    probs_greedy = _normalize_probs(masked_logits)
    sampled_greedy = masked_logits.argmax(dim=-1)
    if sampling_mode == "greedy":
        return probs_greedy, sampled_greedy, None, None

    if sampling_mode == "mix":
        ratio = min(1.0, max(0.0, float(mix_greedy_ratio)))
        u = torch.empty_like(masked_logits).uniform_(1e-8, 1 - 1e-8)
        g = -torch.log(-torch.log(u + 1e-10) + 1e-10)
        probs_gumbel = _gumbel_soft_probs_from_noise(masked_logits, tau, g)
        sampled_gumbel = probs_gumbel.argmax(dim=-1)
        use_greedy = torch.rand((masked_logits.size(0),), device=masked_logits.device) < ratio
        probs = torch.where(use_greedy.unsqueeze(1), probs_greedy, probs_gumbel)
        sampled_ids = torch.where(use_greedy, sampled_greedy, sampled_gumbel)
        return probs, sampled_ids, g, use_greedy

    raise ValueError(f"Unsupported sampling_mode: {sampling_mode}. Use 'gumbel', 'greedy', or 'mix'.")


def _planner_soft_probs(
    *,
    masked_logits: torch.Tensor,
    tau: float,
    sampling_mode: str,
    mix_greedy_ratio: float,
    aligned_gumbel_noise: Optional[torch.Tensor] = None,
    aligned_mix_use_greedy: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if sampling_mode == "gumbel":
        assert aligned_gumbel_noise is not None, "Missing rollout gumbel noise for strict replay alignment."
        return _gumbel_soft_probs_from_noise(masked_logits, tau, aligned_gumbel_noise)
    if sampling_mode == "mix":
        _ = mix_greedy_ratio  # Keep signature stable; row selection comes from rollout trace.
        assert aligned_gumbel_noise is not None, "Missing rollout gumbel noise for strict replay alignment."
        assert aligned_mix_use_greedy is not None, "Missing rollout mix-greedy mask for strict replay alignment."
        greedy_probs = _normalize_probs(masked_logits)
        tau_probs = _gumbel_soft_probs_from_noise(masked_logits, tau, aligned_gumbel_noise)
        return torch.where(aligned_mix_use_greedy.unsqueeze(1), greedy_probs, tau_probs)
    return _normalize_probs(masked_logits)


def _map_global_concept_ids_to_local(
    global_token_ids: torch.Tensor,
    concept_ids_with_eos: torch.Tensor,
) -> torch.Tensor:
    matches = global_token_ids.unsqueeze(-1).eq(concept_ids_with_eos.view(1, 1, -1))
    assert bool(matches.any(dim=-1).all().item()), "Rollout token contains ids outside concept vocabulary."
    return matches.to(dtype=torch.long).argmax(dim=-1)


def rollout_concepts_detached(
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
) -> Tuple[PlannerOutput, PlannerSamplingTrace]:
    """Run planner AR rollout without gradients and record stochastic sampling trace."""
    with torch.no_grad():
        bsz, _ = input_ids.shape
        hidden_size = model.hidden_size
        dtype_embed = model.token_embed_base.weight.dtype

        bos_col = torch.full((bsz, 1), meta.bos_id, device=device, dtype=torch.long)
        eos_col = torch.full((bsz, 1), meta.eos_id, device=device, dtype=torch.long)
        plan_col = torch.full((bsz, 1), meta.plan_token_id, device=device, dtype=torch.long)
        planner_input_ids = torch.cat([bos_col, input_ids, eos_col, plan_col], dim=1)
        planner_type_ids = torch.full_like(planner_input_ids, meta.type_id_text)
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

        planner_embeds = model.embed_with_type(planner_input_ids, planner_type_ids)
        out = model.forward_backbone(
            inputs_embeds=planner_embeds,
            attention_mask=planner_mask,
            position_ids=planner_pos,
            use_cache=use_cache,
        )
        logits_t = model.planner_forward_head(out.last_hidden_state[:, -1, :])
        logits_t = torch.nan_to_num(logits_t, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
        past_kv = out.past_key_values

        eos_id = int(meta.concept_eos_id)
        eos_local_id = int(meta.concept_eos_local_id)
        planner_vocab_size = int(meta.planner_vocab_size)

        concept_table = model.embed_tokens(meta.concept_ids_with_eos)
        type_vec = model.type_embed.weight[int(meta.type_id_concept)].view(1, -1).to(dtype=dtype_embed)

        concept_tokens = torch.full((bsz, meta.max_concept_steps), eos_id, device=device, dtype=torch.long)
        concept_valid = torch.zeros((bsz, meta.max_concept_steps), device=device, dtype=torch.long)
        finished = torch.zeros((bsz,), device=device, dtype=torch.bool)

        gumbel_noise_trace = None
        if sampling_mode in {"gumbel", "mix"}:
            gumbel_noise_trace = torch.zeros(
                (bsz, meta.max_concept_steps, planner_vocab_size),
                device=device,
                dtype=torch.float32,
            )
        mix_use_greedy_trace = None
        if sampling_mode == "mix":
            mix_use_greedy_trace = torch.zeros((bsz, meta.max_concept_steps), device=device, dtype=torch.bool)

        cache_attention_mask = planner_mask
        cache_position_ids = planner_pos[:, -1:].clone() + 1

        for step in range(meta.max_concept_steps):
            active = ~finished
            if not torch.any(active):
                break

            masked_logits = logits_t.float().clone()
            if torch.any(~active):
                masked_logits[~active] = NEG_INF_LOGIT
                masked_logits[~active, eos_local_id] = 0.0

            if min_concept_steps > 1 and step < (min_concept_steps - 1):
                masked_logits[active, eos_local_id] = NEG_INF_LOGIT
            if step >= (meta.max_concept_steps - 1):
                forced = torch.full_like(masked_logits[active], NEG_INF_LOGIT)
                forced[:, eos_local_id] = 0.0
                masked_logits[active] = forced

            probs, sampled_ids_local, g_noise, use_greedy = _sample_planner_tokens_with_trace(
                masked_logits=masked_logits,
                tau=tau,
                sampling_mode=sampling_mode,
                mix_greedy_ratio=mix_greedy_ratio,
            )
            if gumbel_noise_trace is not None:
                assert g_noise is not None
                gumbel_noise_trace[:, step, :] = g_noise.to(dtype=gumbel_noise_trace.dtype)
            if mix_use_greedy_trace is not None:
                assert use_greedy is not None
                mix_use_greedy_trace[:, step] = use_greedy

            sampled_ids = meta.concept_local_to_global(sampled_ids_local)
            sampled_ids = torch.where(active, sampled_ids, torch.full_like(sampled_ids, eos_id))

            concept_tokens[active, step] = sampled_ids[active]
            concept_valid[active, step] = 1

            finished = finished | sampled_ids.eq(eos_id)
            if torch.all(finished):
                break

            active_rows = active.nonzero(as_tuple=False).squeeze(1)
            probs_subset = probs[active_rows]
            soft_t = torch.matmul(probs_subset.to(concept_table.dtype), concept_table).to(dtype=dtype_embed) + type_vec
            hard_t = model.embed_tokens(sampled_ids[active_rows]).to(dtype=dtype_embed) + type_vec

            soft_embed_step = torch.zeros((bsz, hidden_size), device=device, dtype=dtype_embed)
            hard_embed_step = torch.zeros((bsz, hidden_size), device=device, dtype=dtype_embed)
            soft_embed_step[active_rows] = soft_t
            hard_embed_step[active_rows] = hard_t
            if torch.any(~active):
                dummy_ids = torch.full(((~active).sum().item(),), eos_id, device=device, dtype=torch.long)
                hard_embed_step[~active] = model.embed_tokens(dummy_ids) + type_vec
                soft_embed_step[~active] = hard_embed_step[~active]

            st_embed = hard_embed_step + (soft_embed_step - soft_embed_step.detach())
            cache_attention_mask = torch.cat(
                [cache_attention_mask, torch.ones((bsz, 1), device=device, dtype=torch.long)],
                dim=1,
            )
            cache_position_ids = cache_position_ids + 1

            out_next = model.forward_backbone(
                inputs_embeds=st_embed.detach().unsqueeze(1),
                attention_mask=cache_attention_mask,
                position_ids=cache_position_ids,
                past_key_values=_detach_past_key_values(past_kv),
                use_cache=use_cache,
            )
            logits_t = model.planner_forward_head(out_next.last_hidden_state[:, -1, :])
            logits_t = torch.nan_to_num(logits_t, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
            past_kv = _detach_past_key_values(out_next.past_key_values)

        actual_lengths = concept_valid.sum(dim=1)
        planner_out = PlannerOutput(
            concept=PlannerTypeOutput(
                token_ids=concept_tokens,
                valid_mask=concept_valid,
                actual_lengths=actual_lengths,
                st_embeds=None,
            )
        )
        rollout_trace = PlannerSamplingTrace(
            gumbel_noise=gumbel_noise_trace,
            mix_use_greedy=mix_use_greedy_trace,
        )
    return planner_out, rollout_trace


def replay_planner_parallel(
    model: SharedBackboneUnifiedHead,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    rollout_out: PlannerOutput,
    rollout_trace: PlannerSamplingTrace,
    meta: TokenMeta,
    tau: float,
    min_concept_steps: int,
    sampling_mode: str = "gumbel",
    mix_greedy_ratio: float = 0.0,
) -> Tuple[PlannerOutput, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replay fixed concept rollout with one parallel forward pass and compute planner losses."""
    # ----------------------------------------------------------------------
    # 1) Read basic sizes/dtypes used across replay.
    # ----------------------------------------------------------------------
    bsz, _ = input_ids.shape
    max_steps = int(meta.max_concept_steps)
    hidden_size = int(model.hidden_size)
    device = input_ids.device
    dtype_embed = model.token_embed_base.weight.dtype

    # ----------------------------------------------------------------------
    # 2) Load fixed rollout trajectory and validate expected shapes.
    # ----------------------------------------------------------------------
    concept_tokens = rollout_out.concept.token_ids
    concept_valid = rollout_out.concept.valid_mask
    assert concept_tokens.shape == (bsz, max_steps),  \
        f"Invalid concept_tokens shape: got {tuple(concept_tokens.shape)}, expected {(bsz, max_steps)}."
    assert concept_valid.shape == (bsz, max_steps), \
        f"Invalid concept_valid shape: got {tuple(concept_valid.shape)}, expected {(bsz, max_steps)}."

    # ----------------------------------------------------------------------
    # 3) Build planner prompt: [BOS] + source + [EOS] + [PLAN].
    # ----------------------------------------------------------------------
    bos_col = torch.full((bsz, 1), meta.bos_id, device=device, dtype=torch.long)
    eos_col = torch.full((bsz, 1), meta.eos_id, device=device, dtype=torch.long)
    plan_col = torch.full((bsz, 1), meta.plan_token_id, device=device, dtype=torch.long)
    planner_input_ids = torch.cat([bos_col, input_ids, eos_col, plan_col], dim=1)
    planner_type_ids = torch.full_like(planner_input_ids, int(meta.type_id_text))
    planner_mask = torch.cat(
        [
            torch.ones((bsz, 1), device=device, dtype=torch.long),
            attention_mask,
            torch.ones((bsz, 1), device=device, dtype=torch.long),
            torch.ones((bsz, 1), device=device, dtype=torch.long),
        ],
        dim=1,
    )

    # ----------------------------------------------------------------------
    # 4) Build replay teacher-forcing sequence for all planner steps.
    # We append rollout concept tokens (except the last step) as model inputs.
    # ----------------------------------------------------------------------
    assert max_steps > 1, "max_concept_steps must be > 1."
    teacher_token_ids = concept_tokens[:, : max_steps - 1]
    teacher_type_ids = torch.full_like(teacher_token_ids, int(meta.type_id_concept))
    replay_input_ids = torch.cat([planner_input_ids, teacher_token_ids], dim=1)
    replay_type_ids = torch.cat([planner_type_ids, teacher_type_ids], dim=1)
    replay_mask = torch.cat(
        [planner_mask, torch.ones((bsz, max_steps - 1), device=device, dtype=torch.long)],
        dim=1,
    )

    # ----------------------------------------------------------------------
    # 5) Run one parallel backbone forward over replay sequence.
    # ----------------------------------------------------------------------
    replay_pos = torch.cumsum(replay_mask, dim=1) - 1
    replay_pos = replay_pos.clamp(min=0)

    replay_embeds = model.embed_with_type(replay_input_ids, replay_type_ids)
    replay_out = model.forward_backbone(
        inputs_embeds=replay_embeds,
        attention_mask=replay_mask,
        position_ids=replay_pos,
        use_cache=False,
    )

    # ----------------------------------------------------------------------
    # 6) Slice hidden states at planner decision positions and project
    # them to planner logits [B, S, K+1].
    # ----------------------------------------------------------------------
    step_positions = (
        torch.arange(max_steps, device=device, dtype=torch.long)
        + (planner_input_ids.size(1) - 1)
    )
    step_hidden = replay_out.last_hidden_state.index_select(1, step_positions)
    logits_steps = model.planner_forward_head(step_hidden)  # [B, S, K+1]
    logits_steps = torch.nan_to_num(logits_steps, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)

    # ----------------------------------------------------------------------
    # 7) Prepare ids/targets and initialize loss accumulators.
    # ----------------------------------------------------------------------
    eos_id = int(meta.concept_eos_id)
    eos_local_id = int(meta.concept_eos_local_id)
    planner_vocab_size = int(meta.planner_vocab_size)
    concept_ids_with_eos = meta.concept_ids_with_eos.to(device=device, dtype=torch.long)
    sampled_ids_local_all = _map_global_concept_ids_to_local(concept_tokens, concept_ids_with_eos)
    if sampling_mode in {"gumbel", "mix"}:
        assert rollout_trace.gumbel_noise is not None, "Missing rollout gumbel trace for strict replay alignment."
    if sampling_mode == "mix":
        assert rollout_trace.mix_use_greedy is not None, "Missing rollout mix mask trace for strict replay alignment."

    src_lengths = attention_mask.sum(dim=1).to(torch.long)
    expected = (src_lengths.float() * meta.target_concept_ratio).long().clamp(min=1, max=max_steps)

    # Running sums for planner auxiliary losses.
    commit_sum1 = torch.zeros((), device=device, dtype=torch.float32)
    commit_sum2 = torch.zeros((), device=device, dtype=torch.float32)
    hist_sum = torch.zeros((int(meta.concept_ids.numel()),), device=device, dtype=torch.float32)
    eos_sum = torch.zeros((), device=device, dtype=torch.float32)
    eos_count = torch.zeros((), device=device, dtype=torch.float32)

    # Lookup tables used to build soft/hard/ST concept embeddings.
    concept_table = model.embed_tokens(concept_ids_with_eos)
    type_vec = model.type_embed.weight[int(meta.type_id_concept)].view(1, -1).to(dtype=dtype_embed)
    concept_st_embeds = torch.zeros((bsz, max_steps, hidden_size), device=device, dtype=dtype_embed)

    # ----------------------------------------------------------------------
    # 8) Per-step replay loop:
    # - enforce masking/min-length/forced-eos rules
    # - build ST probs from fixed rollout hard ids + differentiable soft probs
    # - accumulate commit/uniform/eos losses
    # - write ST embeddings for executor prefix
    # ----------------------------------------------------------------------
    for step in range(max_steps):
        active = concept_valid[:, step].bool()
        masked_logits = logits_steps[:, step, :].float().clone()

        if torch.any(~active):
            masked_logits[~active] = NEG_INF_LOGIT
            masked_logits[~active, eos_local_id] = 0.0

        if min_concept_steps > 1 and step < (min_concept_steps - 1) and torch.any(active):
            masked_logits[active, eos_local_id] = NEG_INF_LOGIT
        if step >= (max_steps - 1) and torch.any(active):
            forced = torch.full_like(masked_logits[active], NEG_INF_LOGIT)
            forced[:, eos_local_id] = 0.0
            masked_logits[active] = forced

        probs_soft = _planner_soft_probs(
            masked_logits=masked_logits,
            tau=tau,
            sampling_mode=sampling_mode,
            mix_greedy_ratio=mix_greedy_ratio,
            aligned_gumbel_noise=(
                rollout_trace.gumbel_noise[:, step, :]
                if rollout_trace.gumbel_noise is not None
                else None
            ),
            aligned_mix_use_greedy=(
                rollout_trace.mix_use_greedy[:, step]
                if rollout_trace.mix_use_greedy is not None
                else None
            ),
        )
        sampled_ids_local = sampled_ids_local_all[:, step]
        assert torch.all(sampled_ids_local >= 0) and torch.all(sampled_ids_local < planner_vocab_size), (
            f"sampled_ids_local out of range at step {step}: "
            f"got [{sampled_ids_local.min().item()}, {sampled_ids_local.max().item()}], "
            f"expected [0, {planner_vocab_size - 1}]"
        )
        probs_hard = F.one_hot(sampled_ids_local, num_classes=planner_vocab_size).to(dtype=probs_soft.dtype)
        probs = probs_hard - probs_soft.detach() + probs_soft

        active_rows = active.nonzero(as_tuple=False).squeeze(1)
        if active_rows.numel() == 0:
            continue

        probs_subset = probs[active_rows]
        hist_sum = hist_sum + probs_subset[:, :-1].sum(dim=0)

        sampled_ids = concept_tokens[:, step]
        soft_t = torch.matmul(probs_subset.to(concept_table.dtype), concept_table).to(dtype=dtype_embed) + type_vec
        hard_t = model.embed_tokens(sampled_ids[active_rows]).to(dtype=dtype_embed) + type_vec
        st_t = hard_t + (soft_t - soft_t.detach())

        soft_f = soft_t.float()
        hard_f = hard_t.float()
        commit_sum1 = commit_sum1 + (soft_f.detach() - hard_f).pow(2).sum()
        commit_sum2 = commit_sum2 + (soft_f - hard_f.detach()).pow(2).sum()

        eos_target = (step >= (expected[active_rows] - 1)).float()
        eos_logit = masked_logits[active_rows, eos_local_id].clamp(-50.0, 50.0)
        eos_sum = eos_sum + F.binary_cross_entropy_with_logits(eos_logit.float(), eos_target, reduction="sum")
        eos_count = eos_count + eos_target.numel()

        concept_st_embeds = concept_st_embeds.index_put(
            (active_rows, torch.full_like(active_rows, step)),
            st_t,
        )

    # ----------------------------------------------------------------------
    # 9) Finalize planner output tensors.
    # ----------------------------------------------------------------------
    actual_lengths = concept_valid.sum(dim=1)
    concept_out = PlannerTypeOutput(
        token_ids=concept_tokens,
        valid_mask=concept_valid,
        actual_lengths=actual_lengths,
        st_embeds=concept_st_embeds,
    )

    # ----------------------------------------------------------------------
    # 10) Reduce running sums into final scalar losses.
    # ----------------------------------------------------------------------
    denom = max(1.0, float(bsz * max_steps * hidden_size))
    loss_commit = commit_sum1 / denom + BETA_COMMIT * (commit_sum2 / denom)

    hist_total = hist_sum.sum()
    if float(hist_total.detach().item()) > 0.0:
        hist = hist_sum / hist_total.clamp_min(1.0)
        loss_unif = usage_kl_to_uniform(hist)
    else:
        loss_unif = torch.zeros((), device=device, dtype=torch.float32)

    loss_eos = eos_sum / eos_count.clamp_min(1.0)
    loss_len = F.relu(actual_lengths.float() - expected.float()).mean()

    # ----------------------------------------------------------------------
    # 11) Return planner output and auxiliary losses.
    # ----------------------------------------------------------------------
    return PlannerOutput(concept=concept_out), loss_commit, loss_unif, loss_eos, loss_len


def plan_concepts_two_pass(
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
) -> Tuple[PlannerOutput, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-pass planner: detached AR rollout + differentiable parallel replay."""
    rollout_out, rollout_trace = rollout_concepts_detached(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        meta=meta,
        tau=tau,
        min_concept_steps=min_concept_steps,
        device=device,
        sampling_mode=sampling_mode,
        mix_greedy_ratio=mix_greedy_ratio,
        use_cache=use_cache,
    )
    return replay_planner_parallel(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        rollout_out=rollout_out,
        rollout_trace=rollout_trace,
        meta=meta,
        tau=tau,
        min_concept_steps=min_concept_steps,
        sampling_mode=sampling_mode,
        mix_greedy_ratio=mix_greedy_ratio,
    )
