# -*- coding: utf-8 -*-
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from src.config.train_config import *
from src.model import *
from src.utils.train_utils import *

def train():
    """run two-stage concept-first training with a shared backbone and one output head."""
    # =========================
    # 0) Logging and basic runtime info
    # =========================
    log_file = setup_logging(OUTPUT_DIR)
    logging.info(f"[INFO] log file: {log_file}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"[INFO] device: {device}")

    # =========================
    # 1) Resume checkpoint settings
    # =========================
    resume_ckpt_dir = RESUME_CHECKPOINT_DIR.strip()
    resume_backbone_dir = os.path.join(resume_ckpt_dir, "backbone") if resume_ckpt_dir else ""
    resume_heads_path = os.path.join(resume_ckpt_dir, "two_stage_heads.pt") if resume_ckpt_dir else ""
    resume_trainer_state_path = os.path.join(resume_ckpt_dir, "trainer_state.pt") if resume_ckpt_dir else ""
    resume_step = 0
    if RESUME_ENABLED:
        assert resume_ckpt_dir, "RESUME_ENABLED=True but RESUME_CHECKPOINT_DIR is empty."
        assert os.path.isdir(resume_backbone_dir), f"resume backbone dir not found: {resume_backbone_dir}"
        assert os.path.isfile(resume_heads_path), f"resume heads file not found: {resume_heads_path}"
        assert os.path.isfile(resume_trainer_state_path), f"resume trainer_state file not found: {resume_trainer_state_path}"
        logging.info(f"[INFO] resume enabled from: {resume_ckpt_dir}")

    # =========================
    # 2) Tokenizer
    # =========================
    # Extend tokenizer with typed concept tokens used only by stage-1 planning.
    tokenizer = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=True)
    
    # =========================
    # 3) Model dtype + backbone loading
    # =========================
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    model_dtype = dtype_map.get(MODEL_DTYPE, torch.float32)

    model_kwargs = {
        "torch_dtype": model_dtype if torch.cuda.is_available() else torch.float32,
    }
    if torch.cuda.is_available() and ATTENTION_IMPL:
        # Optional acceleration when GPU/runtime supports the target attention implementation.
        model_kwargs["attn_implementation"] = ATTENTION_IMPL
    model_base = AutoModelForCausalLM.from_pretrained(BASE_DIR, **model_kwargs).to(device)

    # =========================
    # 4) Align tokenizer vocab size with model config, then add concept tokens
    # =========================
    # Get model's original vocab size (from config, may include padding rows)
    base_vocab_size = model_base.get_input_embeddings().num_embeddings
    # Save original tokenizer size (valid vocab, for K-means initialization)
    original_tokenizer_vocab_size = len(tokenizer)
    
    # If tokenizer size < model config size, add padding tokens to align
    padding_gap = base_vocab_size - original_tokenizer_vocab_size
    if padding_gap > 0:
        padding_tokens = [f"<PAD_{i}>" for i in range(padding_gap)]
        tokenizer.add_special_tokens({"additional_special_tokens": padding_tokens})
        logging.info(f"[INFO] added {padding_gap} padding tokens to align with model vocab_size={base_vocab_size}")
    
    # Now add concept tokens (they will start from base_vocab_size)
    special_tokens = build_concept_special_tokens(CONCEPT_CONFIG)
    added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    logging.info(f"[INFO] added {added} concept tokens, new tokenizer size: {len(tokenizer)}")

    # HF 默认保留旧行、只初始化新增行
    model_base.resize_token_embeddings(len(tokenizer))
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        modules_to_save=LORA_MODULES_TO_SAVE,
    )
    if RESUME_ENABLED:
        model_base = PeftModel.from_pretrained(model_base, resume_backbone_dir, is_trainable=True).to(device)
    else:
        model_base = get_peft_model(model_base, lora_cfg)
    
    # Optional: Gradient Checkpointing to save VRAM
    if USE_GRADIENT_CHECKPOINTING:
        model_base.gradient_checkpointing_enable()
        logging.info("[INFO] gradient checkpointing enabled (planner kv_cache will be disabled)")

    model_base.print_trainable_parameters()

    # =========================
    # 5) Build unified model and optional resume head states
    # =========================
    # 构建统一的 TokenMeta，管理所有 token ID
    meta = build_token_meta(tokenizer, CONCEPT_CONFIG, device=device, original_vocab_size=original_tokenizer_vocab_size)
    num_type_embeddings = 2
    model = SharedBackboneUnifiedHead(
        model_base,
        num_type_embeddings=num_type_embeddings,
        frozen_output_head_prefix_rows=base_vocab_size,
        meta=meta,
    ).to(device)
    if RESUME_ENABLED:
        head_state = torch.load(resume_heads_path, map_location=device)
        if "output_head_base" in head_state:
            model.output_head_base.load_state_dict(head_state["output_head_base"])
            if model.output_head_new is not None and head_state.get("output_head_new") is not None:
                saved_new_weight = head_state["output_head_new"]["weight"]
                target_rows = model.output_head_new.weight.shape[0]
                if saved_new_weight.shape[0] != target_rows:
                    raise RuntimeError(
                        f"output_head_new rows mismatch: ckpt={saved_new_weight.shape[0]} vs expected={target_rows}. "
                        "Please use a checkpoint trained with the same concept vocabulary size."
                    )
                model.output_head_new.load_state_dict(head_state["output_head_new"])
        else:
            raise RuntimeError("No output head state found in checkpoint.")
        model.type_embed.load_state_dict(head_state["type_embed"])
        token_embed_new = head_state.get("token_embed_new")
        if token_embed_new is not None and token_embed_new.numel() > 0:
            expected_new_rows = model.vocab_size - model.base_vocab_size
            assert model.token_embed_new is not None, "Checkpoint has token_embed_new but model has no new-token embedding rows."
            assert token_embed_new.shape[0] == expected_new_rows, (
                f"token_embed_new rows mismatch: {token_embed_new.shape[0]} vs {expected_new_rows}"
            )
            model.token_embed_new.weight.data.copy_(token_embed_new.to(
                device=device,
                dtype=model.token_embed_new.weight.dtype,
            ))
        saved_step = head_state.get("step", 0)
        if isinstance(saved_step, int):
            resume_step = saved_step

    # =========================
    # 6) Optional precision / compile tweaks
    # =========================
    # Optional FP32 casting for specific trainable params.
    # On A800, keeping large heads in bf16/fp16 is usually faster.
    fp32_casted = 0
    for name, param in model.named_parameters():
        if not param.requires_grad or param.dtype == torch.float32:
            continue
        
        # Ensure all trainable params (LoRA, heads, embeddings) are cast to FP32 for GradScaler stability.
        # "lora_only" implicitly includes other new trainable modules like output_head/token_embed.
        should_cast = False
        if FP32_TRAINABLE == "all":
            should_cast = True
        elif FP32_TRAINABLE == "lora_only":
            conditions = ["lora", "output_head", "token_embed", "type_embed"]
            if any(k in name.lower() for k in conditions):
                should_cast = True

        if should_cast:
            param.data = param.data.float()
            fp32_casted += 1
    if fp32_casted > 0:
        logging.info(f"[INFO] casted {fp32_casted} trainable params to fp32 for AMP stability")

    if USE_COMPILE:
        try:
            compiled = torch.compile(model, mode=COMPILE_MODE, fullgraph=False)
            model = compiled
            logging.info(f"[INFO] torch.compile enabled ({COMPILE_MODE})")
        except Exception as e:
            logging.warning(f"[WARN] torch.compile failed: {e}")

    # =========================
    # 7) Build masks
    # =========================
    vocab_size = model_base.get_input_embeddings().num_embeddings
    logging.info(f"[INFO] Model and tokenizer initialized. Starting data loading...")

    # =========================
    # 8) Dataset and dataloader
    # =========================
    dataset = ParquetSentenceDataset(PARQUET_PATH, max_samples=100000)
    collate_fn = Collator(tokenizer=tokenizer, max_len=MAX_INPUT_TOKENS)
    dataloader_kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "num_workers": 8,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_fn,
        "drop_last": True,
        "prefetch_factor": 4,
        "persistent_workers": True,
    }
    if torch.cuda.is_available():
        # Supported only in PyTorch 2.0+ and CUDA runtime.
        dataloader_kwargs["pin_memory_device"] = "cuda"
    dataloader = DataLoader(
        dataset,
        **dataloader_kwargs,
    )

    # =========================
    # 9) Optimizer / scheduler / AMP / resume trainer state
    # =========================
    optim_params = [p for p in model.parameters() if p.requires_grad]
    assert optim_params, "No trainable parameters found. Check model.requires_grad settings."
    output_head_params = list(model.output_head_new.parameters()) if model.output_head_new is not None else []
    token_embed_params = [model.token_embed_new.weight] if model.token_embed_new is not None else []
    output_head_param_ids = {id(p) for p in output_head_params}
    token_embed_param_ids = {id(p) for p in token_embed_params}
    base_params = [
        p for p in optim_params if id(p) not in output_head_param_ids and id(p) not in token_embed_param_ids
    ]
    optimizer_groups = []
    if base_params:
        optimizer_groups.append(
            {"params": base_params, "lr": LR, "betas": (0.9, 0.95), "weight_decay": 0.01}
        )
    if output_head_params:
        optimizer_groups.append(
            {"params": output_head_params, "lr": LR, "betas": (0.9, 0.95), "weight_decay": 0.0}
        )
    if token_embed_params:
        optimizer_groups.append(
            {"params": token_embed_params, "lr": LR, "betas": (0.9, 0.95), "weight_decay": 0.0}
        )
    optimizer = torch.optim.AdamW(optimizer_groups)
    total_steps = math.ceil(len(dataloader) / GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    use_amp = torch.cuda.is_available() and model_dtype in (torch.float16, torch.bfloat16)
    use_scaler = torch.cuda.is_available() and model_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if torch.cuda.is_available() else None
    if RESUME_ENABLED:
        if not os.path.isfile(resume_trainer_state_path):
            logging.warning(
                f"[WARN] trainer_state.pt not found at {resume_trainer_state_path}, "
                "resume without optimizer/scheduler/scaler state"
            )
        else:
            trainer_state = torch.load(resume_trainer_state_path, map_location="cpu")
            if "optimizer" in trainer_state:
                optimizer.load_state_dict(trainer_state["optimizer"])
            if "scheduler" in trainer_state:
                scheduler.load_state_dict(trainer_state["scheduler"])
            if scaler is not None and scaler.is_enabled() and "scaler" in trainer_state:
                scaler.load_state_dict(trainer_state["scaler"])
            resume_step = int(trainer_state.get("step", resume_step))
        

    logging.info(f"[INFO] vocab size: {vocab_size}")
    logging.info(f"[INFO] output_head frozen rows: 0, {base_vocab_size})")
    new_rows = int(model.output_head_new.weight.shape[0]) if model.output_head_new is not None else 0
    effective_head_params = new_rows * model.hidden_size
    logging.info(
        f"[INFO] planner head trainable rows: {new_rows} "
        f"({effective_head_params:,} params)"
    )
    logging.info("[INFO] concept type: single")
    logging.info(f"[INFO] steps: total={total_steps}, warmup={warmup_steps}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    ratio = 100.0 * trainable_params / max(1, all_params)
    logging.info(f"[INFO] trainable params: {trainable_params:,} / {all_params:,} ({ratio:.4f}%)")

    # =========================
    # 10) Initialize wandb tracking (best-default behavior)
    # =========================
    wandb_run = init_wandb_run(
        model=model,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        trainable_params=trainable_params,
        all_params=all_params,
        new_rows=new_rows,
        device=device,
    )

    # =========================
    # 11) Training loop with two-stage forward, logging, wandb metrics, and checkpointing
    # =========================
    # ---- [Init] training loop state / progress bar ----
    model.train()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_batches = (len(dataloader) * EPOCHS) / GRAD_ACCUM
    progress_bar = tqdm(total=total_batches, desc="training", unit="step")
    start_time = time.time()

    global_step = resume_step
    log_every = max(1, LOG_STEPS)
    eval_every = max(0, int(EVAL_STEPS))
    for epoch in range(EPOCHS):
        # ---- [Epoch] reset per-epoch counters ----
        epoch_losses: List[float] = []
        optimizer.zero_grad(set_to_none=True)
        tokens_since_last_step = 0
        step_timer_start = time.perf_counter()

        # ---- [Epoch] init async prefetcher ----
        micro_step = 0
        prefetcher = CUDAPrefetcher(dataloader, device=device)
        batch = prefetcher.next()

        while batch is not None:
            # ---- [Batch] load tensors + decide profiling ----
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            tokens_since_last_step += int(attention_mask.sum().item())

            tau = get_adaptive_tau(global_step, total_steps, TAU_INIT, TAU_MIN)
            will_do_step = ((micro_step + 1) % GRAD_ACCUM == 0) or (prefetcher.next_batch is None)
            # Align profiling steps with the later wandb logging condition:
            # after optimizer step, logs happen when `global_step % log_every == 1`.
            should_profile_step = will_do_step and (global_step % log_every == 0)
            profile_cuda = should_profile_step and torch.cuda.is_available()
            stage_metrics: Dict[str, float] = {}
            tf_mask_target_ratio = 0.0
            tf_mask_applied_ratio = 0.0
            planner_mix_greedy_ratio = 0.0
            planner_mix_greedy_ratio = get_planner_mix_greedy_ratio(
                global_step=global_step,
                total_steps=total_steps,
                ratio_min=TRAIN_PLANNER_MIX_GREEDY_RATIO_MIN,
                ratio_max=TRAIN_PLANNER_MIX_GREEDY_RATIO_MAX,
            ) if TRAIN_PLANNER_SAMPLING_MODE == "mix" else 0.0

            # ---- [Forward] planner + executor in AMP autocast ----
            with torch.amp.autocast("cuda", dtype=model_dtype, enabled=use_amp):
                # [Planner] source text -> typed concept sequences.
                planner_stage_state = cuda_stage_begin(profile_cuda)
                (
                    planner_out,
                    loss_commit,
                    loss_unif,
                    loss_eos,
                    loss_len,
                ) = plan_concepts(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    meta=meta,
                    tau=tau,
                    sampling_mode=TRAIN_PLANNER_SAMPLING_MODE,
                    mix_greedy_ratio=planner_mix_greedy_ratio,
                    min_concept_steps=MIN_CONCEPT_STEPS,
                    device=device,
                    use_cache=not USE_GRADIENT_CHECKPOINTING,
                )
                stage_metrics.update(cuda_stage_end("planner", planner_stage_state))

                # [Executor] concept prefix + AR reconstruction.
                execute_stage_state = cuda_stage_begin(profile_cuda)
                (
                    prefix_embeds,
                    prefix_mask,
                    prefix_pos,
                    _prefix_token_ids,
                    _prefix_type_ids,
                ) = build_executor_prefix(
                    model,
                    planner_out=planner_out,
                    meta=meta,
                    device=device,
                )

                decoder_in, decoder_mask, labels, _ = build_decoder_tensors(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bos_id=meta.bos_id,
                    eos_id=meta.eos_id,
                    device=device,
                )

                # augment: optional stage-2 decoder corruption via random base-vocab token replacement.
                decoder_in, tf_mask_target_ratio, tf_mask_applied_ratio = apply_stage2_tf_token_masking(
                    decoder_in=decoder_in,
                    decoder_mask=decoder_mask,
                    global_step=global_step,
                    total_steps=total_steps,
                    enabled=ENABLE_STAGE2_TF_MASKING,
                    ratio_max=STAGE2_TF_MASKING_MAX_RATIO,
                    ratio_min=STAGE2_TF_MASKING_MIN_RATIO,
                    random_token_upper_bound=base_vocab_size,
                )

                decoder_type_ids = torch.full_like(decoder_in, TYPE_ID_TEXT)
                decoder_embeds = model.embed_with_type(decoder_in, decoder_type_ids)

                full_embeds = torch.cat([prefix_embeds, decoder_embeds], dim=1)
                # full_mask 表示“整段输入里哪些 token 是有效上下文，哪些是 padding”
                full_mask = torch.cat([prefix_mask, decoder_mask], dim=1)

                # Text positions continue after the effective (non-padding) prefix length.
                prefix_true_len = prefix_mask.sum(dim=1).to(torch.long)
                dec_pos = (
                    torch.arange(decoder_in.size(1), device=device, dtype=torch.long)
                    .unsqueeze(0)
                    .expand(decoder_in.size(0), -1)
                )
                dec_pos = dec_pos + prefix_true_len.unsqueeze(1)
                full_pos = torch.cat([prefix_pos, dec_pos], dim=1)

                out = model.forward_backbone(
                    inputs_embeds=full_embeds,
                    attention_mask=full_mask,
                    position_ids=full_pos,
                    use_cache=False,
                )
                hidden = out.last_hidden_state[:, prefix_embeds.size(1) :, :]
                logits = model.executor_forward_head(hidden)

                loss_rec = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
                loss = (
                    LAMBDA_REC * loss_rec
                    + LAMBDA_COMMIT * loss_commit
                    + LAMBDA_UNIF * loss_unif
                    + LAMBDA_EOS * loss_eos
                ).float()
                stage_metrics.update(cuda_stage_end("execute", execute_stage_state))

            # ---- [Guard] skip invalid loss early ----
            if not torch.isfinite(loss):
                logging.warning("[WARN] non-finite loss, skip batch")
                optimizer.zero_grad(set_to_none=True)
                batch = prefetcher.next()
                continue

            # ---- [Backward] gradient accumulation backward ----
            bptt_stage_state = cuda_stage_begin(profile_cuda)
            loss_scaled = loss / GRAD_ACCUM
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()
            stage_metrics.update(cuda_stage_end("bptt", bptt_stage_state))

            micro_step += 1
            is_last_in_epoch = prefetcher.next_batch is None
            do_step = (micro_step % GRAD_ACCUM == 0) or is_last_in_epoch
            # ---- [Accum] only step optimizer on accumulation boundary ----
            if not do_step:
                batch = prefetcher.next()
                continue

            # ---- [Optim] unscale grads before grad checks / clipping ----
            optim_stage_state = cuda_stage_begin(profile_cuda)
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)

            # ---- [Guard] collect trainable params with valid grads ----
            trainable_named_with_grad = [
                (name, p) for name, p in model.named_parameters() if p.requires_grad and p.grad is not None
            ]
            trainable_with_grad = [p for _, p in trainable_named_with_grad]
            if not trainable_with_grad:
                if scaler is not None and scaler.is_enabled():
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)
                tokens_since_last_step = 0
                step_timer_start = time.perf_counter()
                batch = prefetcher.next()
                continue

            # ---- [Guard] skip step if any grad is NaN/Inf ----
            has_non_finite_grad = False
            bad_grad_name = ""
            bad_grad_ref = None
            for name, p in trainable_named_with_grad:
                if not torch.isfinite(p.grad).all():
                    has_non_finite_grad = True
                    bad_grad_name = name
                    bad_grad_ref = p.grad
                    break
            if has_non_finite_grad:
                scale_info = ""
                if scaler is not None and scaler.is_enabled():
                    scale_info = f", grad_scale={float(scaler.get_scale()):.1f}"
                detail = ""
                if bad_grad_ref is not None:
                    grad_view = bad_grad_ref.detach()
                    if grad_view.is_sparse:
                        grad_view = grad_view.coalesce().values()
                    finite_mask = torch.isfinite(grad_view)
                    finite_ratio = float(finite_mask.float().mean().item())
                    grad_abs_max = float(torch.nan_to_num(grad_view.float(), nan=0.0, posinf=0.0, neginf=0.0).abs().max().item())
                    detail = (
                        f" first_bad_param={bad_grad_name}, "
                        f"finite_ratio={finite_ratio:.6f}, grad_abs_max={grad_abs_max:.4e}"
                    )
                logging.warning(f"[WARN] non-finite gradients detected, skip optimizer step{scale_info}{detail}")
                if scaler is not None and scaler.is_enabled():
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)
                tokens_since_last_step = 0
                step_timer_start = time.perf_counter()
                batch = prefetcher.next()
                continue

            # ---- [Optim] clip grad norm + optimizer step ----
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_with_grad, max_norm=1.0)

            did_step = False
            if torch.isfinite(grad_norm):
                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                did_step = True
            else:
                logging.warning("[WARN] non-finite grad norm, skip optimizer step")
                if scaler is not None and scaler.is_enabled():
                    scaler.update()
            stage_metrics.update(cuda_stage_end("optimizer", optim_stage_state))

            optimizer.zero_grad(set_to_none=True)

            # ---- [Step] scheduler / throughput / logging / wandb ----
            if did_step:
                scheduler.step()
                global_step += 1
                epoch_losses.append(float(loss.detach().cpu()))

                step_wall_time = time.perf_counter() - step_timer_start
                step_tokens = tokens_since_last_step
                tokens_since_last_step = 0
                step_timer_start = time.perf_counter()

                if global_step % log_every == 1:
                    recent = epoch_losses[-min(log_every, len(epoch_losses)) :]
                    avg_loss = sum(recent) / max(1, len(recent))
                    avg_len = planner_out.concept.actual_lengths.float().mean().item()
                    # ---- [Metric] concept-token diversity for current planned batch ----
                    concept_diversity_metrics = compute_concept_diversity_metrics(
                        planner_out=planner_out,
                        meta=meta,
                    )
                    seq_div_mean = concept_diversity_metrics["train/concept_diversity_seq_mean"]
                    batch_div_ratio = concept_diversity_metrics["train/concept_diversity_batch_ratio"]
                    step_extra_metrics = dict(concept_diversity_metrics)
                    # ---- [Metric] stage-2 teacher-forcing masking ratio ----
                    if ENABLE_STAGE2_TF_MASKING:
                        step_extra_metrics["train/stage2_tf_mask_ratio_target"] = tf_mask_target_ratio
                        step_extra_metrics["train/stage2_tf_mask_ratio_applied"] = tf_mask_applied_ratio
                    if TRAIN_PLANNER_SAMPLING_MODE == "mix":
                        step_extra_metrics["train/planner_mix_greedy_ratio"] = planner_mix_greedy_ratio

                    logging.info(
                        f"[Epoch {epoch + 1}/{EPOCHS}] "
                        f"Step {global_step} | "
                        f"Loss {float(loss.detach().cpu()):.4f} (avg {avg_loss:.4f}) | "
                        f"Rec {float(loss_rec.detach().cpu()):.4f} | "
                        f"Commit {float(loss_commit.detach().cpu()):.4f} | "
                        f"Unif {float(loss_unif.detach().cpu()):.4f} | "
                        f"EOS {float(loss_eos.detach().cpu()):.4f} | "
                        f"Len {float(loss_len.detach().cpu()):.4f} | "
                        f"ConceptLen {avg_len:.2f} | "
                        f"SeqDiv {seq_div_mean:.4f} | "
                        f"BatchDiv {batch_div_ratio:.4f} | "
                        f"TFMask {tf_mask_applied_ratio:.4f} | "
                        f"PlanMix {planner_mix_greedy_ratio:.4f} | "
                        f"Tau {tau:.4f} | "
                        f"LR {scheduler.get_last_lr()[0]:.2e} | "
                        f"Tok/s {step_tokens / max(step_wall_time, 1e-9):.1f}"
                    )

                    log_wandb_step_metrics(
                        wandb_run=wandb_run,
                        global_step=global_step,
                        epoch=epoch,
                        meta=meta,
                        avg_len=avg_len,
                        avg_loss=avg_loss,
                        loss=loss,
                        loss_rec=loss_rec,
                        loss_commit=loss_commit,
                        loss_unif=loss_unif,
                        loss_eos=loss_eos,
                        loss_len=loss_len,
                        grad_norm=grad_norm,
                        tau=tau,
                        lr=scheduler.get_last_lr()[0],
                        step_wall_time=step_wall_time,
                        step_tokens=step_tokens,
                        scaler=scaler,
                        stage_metrics=stage_metrics,
                        extra_metrics=step_extra_metrics,
                    )

                # ---- [Step] periodic eval and checkpoint ----
                if eval_every > 0 and global_step % eval_every == 0 or is_last_in_epoch:
                    run_periodic_eval(
                        model=model,
                        tokenizer=tokenizer,
                        meta=meta,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        device=device,
                        global_step=global_step,
                        wandb_run=wandb_run,
                        model_dtype=model_dtype,
                        use_amp=use_amp,
                    )

                if global_step % SAVE_STEPS == 0:
                    save_checkpoint(
                        model,
                        tokenizer,
                        meta,
                        global_step,
                        OUTPUT_DIR,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                    )
            else:
                tokens_since_last_step = 0
                step_timer_start = time.perf_counter()

            # ---- [UI] progress bar + ETA ----
            progress_bar.update(1)
            elapsed = time.time() - start_time
            done = progress_bar.n
            if done > 0:
                tpb = elapsed / done
                remain = total_batches - done
                eta = timedelta(seconds=int(max(0, remain) * tpb))
                progress_bar.set_postfix({"epoch": f"{epoch + 1}/{EPOCHS}", "eta": str(eta)})

            batch = prefetcher.next()

    # ---- [Finalize] final checkpoint after all epochs ----
    total_seconds = int(time.time() - start_time)
    total_time = timedelta(seconds=total_seconds)
    logging.info(f"[DONE] training complete, total time: {total_time}")
    save_checkpoint(
        model,
        tokenizer,
        meta,
        "final",
        OUTPUT_DIR,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )

    # =========================
    # 12) Finalize progress bars and wandb run
    # =========================
    progress_bar.close()
    finish_wandb_run(
        wandb_run,
        total_seconds=int(time.time() - start_time),
        final_global_step=global_step,
    )

if __name__ == "__main__":
    train()
