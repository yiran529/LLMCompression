# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass
from typing import Iterable, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config.inference_config import *
from src.model import *
from src.utils.inference_utils import *


# Planner debug switches are intentionally hardcoded here (not in config).
PLANNER_DEBUG_ENABLED = True
PLANNER_DEBUG_TOPK = 5
PLANNER_DEBUG_MAX_STEPS = 8
PLANNER_DEBUG_MAX_SAMPLES = 4


def _format_topk_line(tokenizer, token_ids: List[int], probs: List[float]) -> str:
    pairs: List[str] = []
    for token_id, prob in zip(token_ids, probs):
        token = tokenizer.convert_ids_to_tokens([token_id])[0]
        pairs.append(f"{token_id}:{token}({prob:.4f})")
    return " | ".join(pairs)


def _print_planner_debug(infer_out, tokenizer) -> None:
    records = getattr(infer_out.planner_out, "debug_records", None)
    if not records:
        print("[PlannerDebug] no debug records")
        return

    print("[PlannerDebug] begin")
    for rec in records:
        if rec.get("stage") == "prime":
            prompt_lens = [int(x) for x in rec.get("prompt_true_len", [])]
            hidden_norms = [float(x) for x in rec.get("hidden_norms", [])]
            logits_norms = [float(x) for x in rec.get("logits_norms", [])]
            logits_means = [float(x) for x in rec.get("logits_means", [])]
            logits_stds = [float(x) for x in rec.get("logits_stds", [])]
            logits_max_abs = [float(x) for x in rec.get("logits_max_abs", [])]
            hidden_diffs = [float(x) for x in rec.get("hidden_max_abs_diff_vs0", [])]
            logits_diffs = [float(x) for x in rec.get("logits_max_abs_diff_vs0", [])]
            concept_means = [float(x) for x in rec.get("concept_logits_means", [])]
            concept_stds = [float(x) for x in rec.get("concept_logits_std", [])]
            concept_max_abs = [float(x) for x in rec.get("concept_logits_max_abs", [])]
            concept_diffs = [float(x) for x in rec.get("concept_logits_max_abs_diff_vs0", [])]
            concept_centered_diffs = [float(x) for x in rec.get("concept_logits_centered_max_abs_diff_vs0", [])]
            concept_margins = [float(x) for x in rec.get("concept_top1_minus_top2", [])]
            prime_type_name = str(rec.get("prime_type_name", "unknown"))

            print("  PrimeForward | per-sample stats (vs sample0)")
            for i in range(len(prompt_lens)):
                print(
                    f"    sample={i} prompt_true_len={prompt_lens[i]} "
                    f"hidden_norm={hidden_norms[i]:.6f} logits_norm={logits_norms[i]:.6f} "
                    f"logits_mean={logits_means[i]:.6e} logits_std={logits_stds[i]:.6e} logits_max_abs={logits_max_abs[i]:.6e} "
                    f"hidden_max_abs_diff_vs0={hidden_diffs[i]:.6e} "
                    f"logits_max_abs_diff_vs0={logits_diffs[i]:.6e} "
                    f"{prime_type_name}_concept_mean={concept_means[i]:.6e} "
                    f"{prime_type_name}_concept_std={concept_stds[i]:.6e} "
                    f"{prime_type_name}_concept_max_abs={concept_max_abs[i]:.6e} "
                    f"{prime_type_name}_concept_margin(top1-top2)={concept_margins[i]:.6e} "
                    f"{prime_type_name}_concept_diff_vs0={concept_diffs[i]:.6e} "
                    f"{prime_type_name}_concept_centered_diff_vs0={concept_centered_diffs[i]:.6e}"
                )
            continue

        decode_step = int(rec.get("decode_step", -1))
        deterministic = bool(rec.get("deterministic", False))
        tau = float(rec.get("tau", 0.0))
        samples = rec.get("samples", [])
        print(f"  Step {decode_step} | deterministic={deterministic} | tau={tau:.4f}")

        selected_ids: List[int] = []
        selected_probs: List[float] = []
        entropies: List[float] = []
        top1_ids: List[int] = []

        for s in samples:
            sample_idx = int(s["sample_idx"])
            type_name = str(s["type_name"])
            step_in_type = int(s["step_in_type"])
            selected_id = int(s["selected_id"])
            selected_prob = float(s["selected_prob"])
            entropy = float(s["entropy"])
            hidden_norm = float(s.get("hidden_norm", 0.0))
            hidden_std = float(s.get("hidden_std", 0.0))
            hidden_max_abs = float(s.get("hidden_max_abs", 0.0))
            hidden_diff_vs0 = float(s.get("hidden_max_abs_diff_vs0", 0.0))
            hidden_centered_diff_vs0 = float(s.get("hidden_centered_max_abs_diff_vs0", 0.0))
            hidden_cosine_vs0 = float(s.get("hidden_cosine_vs0", 1.0))
            concept_logits_mean = float(s.get("concept_logits_mean", 0.0))
            concept_logits_std = float(s.get("concept_logits_std", 0.0))
            concept_logits_max_abs = float(s.get("concept_logits_max_abs", 0.0))
            top1_minus_top2 = float(s.get("top1_minus_top2", 0.0))
            concept_logits_diff_vs0 = float(s.get("concept_logits_max_abs_diff_vs0", 0.0))
            concept_logits_centered_diff_vs0 = float(s.get("concept_logits_centered_max_abs_diff_vs0", 0.0))
            topk_ids = [int(x) for x in s["topk_token_ids"]]
            topk_probs = [float(x) for x in s["topk_probs"]]
            top1_id = topk_ids[0] if topk_ids else -1

            selected_ids.append(selected_id)
            selected_probs.append(selected_prob)
            entropies.append(entropy)
            top1_ids.append(top1_id)

            print(
                f"    sample={sample_idx} type={type_name} local_step={step_in_type} "
                f"selected={selected_id} p_sel={selected_prob:.4f} H={entropy:.4f} "
                f"hidden_norm={hidden_norm:.6e} hidden_std={hidden_std:.6e} hidden_max_abs={hidden_max_abs:.6e} "
                f"hidden_diff_vs0={hidden_diff_vs0:.6e} hidden_centered_diff_vs0={hidden_centered_diff_vs0:.6e} "
                f"hidden_cosine_vs0={hidden_cosine_vs0:.6f} "
                f"concept_mean={concept_logits_mean:.6e} concept_std={concept_logits_std:.6e} "
                f"concept_max_abs={concept_logits_max_abs:.6e} margin(top1-top2)={top1_minus_top2:.6e} "
                f"concept_diff_vs0={concept_logits_diff_vs0:.6e} "
                f"concept_centered_diff_vs0={concept_logits_centered_diff_vs0:.6e}"
            )
            print(f"      topk: {_format_topk_line(tokenizer, topk_ids, topk_probs)}")

        if selected_ids:
            same_selected = len(set(selected_ids)) == 1
            same_top1 = len(set(top1_ids)) == 1
            p_span = max(selected_probs) - min(selected_probs)
            h_span = max(entropies) - min(entropies)
            print(
                "    summary: "
                f"same_selected={same_selected} same_top1={same_top1} "
                f"p_sel_span={p_span:.6f} entropy_span={h_span:.6f}"
            )

    print("[PlannerDebug] end")


def run_inference() -> None:
    runtime = load_runtime()
    tokenizer = runtime.tokenizer
    texts = get_input_texts()

    sample_offset = 0
    for batch_texts in iter_batches(texts, INFER_BATCH_SIZE):
        tok = tokenizer(
            batch_texts,
            add_special_tokens=False,
            truncation=True,
            max_length=INFER_MAX_INPUT_TOKENS,
            return_tensors="pt",
            padding=True,
        )
        input_ids = tok["input_ids"].to(runtime.device)
        attention_mask = tok["attention_mask"].to(runtime.device)

        if PLANNER_DEBUG_ENABLED:
            print("[PlannerDebug] input preview")
            max_preview = min(4, input_ids.size(0))
            for i in range(max_preview):
                real_len = int(attention_mask[i].sum().item())
                head_ids = input_ids[i, : min(20, input_ids.size(1))].tolist()
                print(f"  sample={i} real_len={real_len} input_ids_head20={head_ids}")

        infer_out = run_executor_inference(
            runtime.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            plan_token_id=runtime.plan_token_id,
            bos_id=runtime.bos_id,
            eos_id=runtime.eos_id,
            metas=runtime.metas,
            mask_cache=runtime.mask_cache,
            device=runtime.device,
            max_new_tokens=INFER_MAX_NEW_TOKENS,
            planner_tau=INFER_PLANNER_TAU,
            min_concept_steps=INFER_MIN_CONCEPT_STEPS,
            planner_deterministic=INFER_PLANNER_DETERMINISTIC,
            planner_debug_collect=PLANNER_DEBUG_ENABLED,
            planner_debug_topk=PLANNER_DEBUG_TOPK,
            planner_debug_max_steps=PLANNER_DEBUG_MAX_STEPS,
            planner_debug_max_samples=PLANNER_DEBUG_MAX_SAMPLES,
        )

        if PLANNER_DEBUG_ENABLED:
            _print_planner_debug(infer_out, tokenizer)

        bsz = len(batch_texts)
        for i in range(bsz):
            global_idx = sample_offset + i
            gen_len = int(infer_out.lengths[i].item())
            gen_ids = infer_out.generated_ids[i, :gen_len].tolist()
            gen_ids = trim_to_first_eos(gen_ids, runtime.eos_id)
            out_text = tokenizer.decode(gen_ids, skip_special_tokens=INFER_SKIP_SPECIAL_TOKENS)

            print("=" * 100)
            print(f"[Sample {global_idx}]")
            print(f"Input : {batch_texts[i]}")
            print("Concepts:")
            for line in format_concepts(
                tokenizer=tokenizer,
                metas=runtime.metas,
                planner_out=infer_out.planner_out,
                sample_idx=i,
            ):
                print(f"  {line}")
            print(f"Output: {out_text}")

        sample_offset += bsz


def main() -> None:
    run_inference()


if __name__ == "__main__":
    main()
