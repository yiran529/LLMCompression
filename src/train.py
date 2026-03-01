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
    # 2) Tokenizer + special planning tokens
    # =========================
    # Extend tokenizer with typed concept tokens used only by stage-1 planning.
    tokenizer = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=True)
    special_tokens = build_concept_special_tokens(CONCEPT_TYPE_CONFIGS)
    added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    logging.info(f"[INFO] added special tokens: {added}")

    plan_token_id = tokenizer.convert_tokens_to_ids("<PLAN>")
    bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id or tokenizer.bos_token_id
    if bos_id is None or eos_id is None:
        raise RuntimeError("Tokenizer must provide BOS/EOS ids.")

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
    # 4) Resize embeddings + apply LoRA (or load LoRA on resume)
    # =========================
    base_vocab_size = model_base.get_input_embeddings().num_embeddings
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
    model_base.print_trainable_parameters()

    # =========================
    # 5) Build unified model and optional resume head states
    # =========================
    metas = build_concept_metas(tokenizer, CONCEPT_TYPE_CONFIGS, device=device)
    num_type_embeddings = 1 + len(metas)
    model = SharedBackboneUnifiedHead(
        model_base,
        num_type_embeddings=num_type_embeddings,
        frozen_output_head_prefix_rows=base_vocab_size,
    ).to(device)
    model.planner_quota = PlannerQuotaController(
        tau=QUOTA_TAU,
        eta=QUOTA_ETA,
        lambda_init=QUOTA_LAMBDA_INIT,
        lambda_max=QUOTA_LAMBDA_MAX,
        device=device,
    )
    if RESUME_ENABLED:
        head_state = torch.load(resume_heads_path, map_location=device)
        if "output_head_base" in head_state:
            model.output_head_base.load_state_dict(head_state["output_head_base"])
            if model.output_head_new is not None and head_state.get("output_head_new") is not None:
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
        if "planner_quota" in head_state and head_state["planner_quota"] is not None:
            model.planner_quota.load_state_dict(head_state["planner_quota"], device=device)
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
            compiled.planner_quota = model.planner_quota
            model = compiled
            logging.info(f"[INFO] torch.compile enabled ({COMPILE_MODE})")
        except Exception as e:
            logging.warning(f"[WARN] torch.compile failed: {e}")

    # =========================
    # 7) Build masks
    # =========================
    blocked_ids = build_executor_blocklist(metas, plan_token_id=plan_token_id)
    vocab_size = model_base.get_input_embeddings().num_embeddings
    mask_cache = ConceptMaskCache(
        metas=metas,
        vocab_size=vocab_size,
        base_vocab_size=base_vocab_size,
        blocked_for_executor=blocked_ids,
        device=device,
    )
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
    logging.info(f"[INFO] output_head frozen rows: [0, {base_vocab_size})")
    new_rows = max(0, vocab_size - base_vocab_size)
    effective_head_params = new_rows * model.hidden_size
    logging.info(
        f"[INFO] output_head trainable rows: [{base_vocab_size}, {vocab_size}) "
        f"({new_rows} rows, {effective_head_params:,} params)"
    )
    logging.info(f"[INFO] concept types: {[m.name for m in metas]}")
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
    model.train()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_batches = (len(dataloader) * EPOCHS) / GRAD_ACCUM
    progress_bar = tqdm(total=total_batches, desc="training", unit="step")
    start_time = time.time()

    global_step = resume_step
    log_every = max(1, LOG_STEPS)
    eval_every = max(0, int(EVAL_STEPS))
    for epoch in range(EPOCHS):
        epoch_losses: List[float] = []
        optimizer.zero_grad(set_to_none=True)
        tokens_since_last_step = 0
        step_timer_start = time.perf_counter()

        micro_step = 0
        prefetcher = CUDAPrefetcher(dataloader, device=device)
        batch = prefetcher.next()

        while batch is not None:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            tokens_since_last_step += int(attention_mask.sum().item())

            tau = get_adaptive_tau(global_step, total_steps, TAU_INIT, TAU_MIN)
            will_do_step = ((micro_step + 1) % GRAD_ACCUM == 0) or (prefetcher.next_batch is None)
            should_profile_step = will_do_step and ((global_step + 1) % log_every == 0)
            profile_cuda = should_profile_step and torch.cuda.is_available()
            stage_metrics: Dict[str, float] = {}

            with torch.amp.autocast("cuda", dtype=model_dtype, enabled=use_amp):
                planner_stage_state = cuda_stage_begin(profile_cuda)
                # Stage 1 (Planner): source text -> typed concept sequences.
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
                    plan_token_id=plan_token_id,
                    bos_id=bos_id,
                    metas=metas,
                    mask_cache=mask_cache,
                    tau=tau,
                    min_concept_steps=MIN_CONCEPT_STEPS,
                    base_vocab_size=base_vocab_size,
                    device=device,
                )
                stage_metrics.update(cuda_stage_end("planner", planner_stage_state))

                execute_stage_state = cuda_stage_begin(profile_cuda)
                # Build concept-only prefix for Stage 2.
                (
                    prefix_embeds,
                    prefix_mask,
                    prefix_pos,
                    _prefix_token_ids,
                    _prefix_type_ids,
                ) = build_executor_prefix(
                    model,
                    planner_out=planner_out,
                    metas=metas,
                    bos_id=bos_id,
                    device=device,
                )

                decoder_in, decoder_mask, labels, _ = build_decoder_tensors(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bos_id=bos_id,
                    eos_id=eos_id,
                    device=device,
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
                logits = model.forward_head(hidden)
                # Executor should not produce planner-only tokens.
                logits = logits.masked_fill(mask_cache.executor_block_bool.view(1, 1, -1), -1e4)

                loss_rec = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
                loss_quota, quota_bar = compute_planner_quota_loss(
                    planner_out=planner_out,
                    quota_controller=model.planner_quota,
                )

                loss = (
                    LAMBDA_REC * loss_rec
                    + LAMBDA_COMMIT * loss_commit
                    + LAMBDA_UNIF * loss_unif
                    + LAMBDA_EOS * loss_eos
                    + loss_quota
                ).float()
                stage_metrics.update(cuda_stage_end("execute", execute_stage_state))

            if not torch.isfinite(loss):
                logging.warning("[WARN] non-finite loss, skip batch")
                optimizer.zero_grad(set_to_none=True)
                batch = prefetcher.next()
                continue

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
            if not do_step:
                batch = prefetcher.next()
                continue

            optim_stage_state = cuda_stage_begin(profile_cuda)
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)

            trainable_with_grad = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not trainable_with_grad:
                optimizer.zero_grad(set_to_none=True)
                tokens_since_last_step = 0
                step_timer_start = time.perf_counter()
                batch = prefetcher.next()
                continue

            for p in trainable_with_grad:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=1e4, neginf=-1e4)
                p.grad.clamp_(-5.0, 5.0)
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
            stage_metrics.update(cuda_stage_end("optimizer", optim_stage_state))

            optimizer.zero_grad(set_to_none=True)

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
                    avg_lens = []
                    for out_type in planner_out.per_type:
                        avg_lens.append(out_type.actual_lengths.float().mean().item())

                    logging.info(
                        f"[Epoch {epoch + 1}/{EPOCHS}] "
                        f"Step {global_step} | "
                        f"Loss {float(loss.detach().cpu()):.4f} (avg {avg_loss:.4f}) | "
                        f"Rec {float(loss_rec.detach().cpu()):.4f} | "
                        f"Commit {float(loss_commit.detach().cpu()):.4f} | "
                        f"Unif {float(loss_unif.detach().cpu()):.4f} | "
                        f"EOS {float(loss_eos.detach().cpu()):.4f} | "
                        f"Len {float(loss_len.detach().cpu()):.4f} | "
                        f"Quota {float(loss_quota.detach().cpu()):.4f} | "
                        f"QuotaBar {float(quota_bar.detach().cpu()):.4f} | "
                        f"QuotaLam {float(model.planner_quota.lambda_value.detach().cpu()):.4f} | "
                        f"TypeLens {','.join([f'{x:.2f}' for x in avg_lens])} | "
                        f"Tau {tau:.4f} | "
                        f"LR {scheduler.get_last_lr()[0]:.2e} | "
                        f"Tok/s {step_tokens / max(step_wall_time, 1e-9):.1f}"
                    )

                    log_wandb_step_metrics(
                        wandb_run=wandb_run,
                        global_step=global_step,
                        epoch=epoch,
                        metas=metas,
                        avg_lens=avg_lens,
                        avg_loss=avg_loss,
                        loss=loss,
                        loss_rec=loss_rec,
                        loss_commit=loss_commit,
                        loss_unif=loss_unif,
                        loss_eos=loss_eos,
                        loss_len=loss_len,
                        loss_quota=loss_quota,
                        quota_bar=quota_bar,
                        quota_lambda=model.planner_quota.lambda_value,
                        grad_norm=grad_norm,
                        tau=tau,
                        lr=scheduler.get_last_lr()[0],
                        step_wall_time=step_wall_time,
                        step_tokens=step_tokens,
                        scaler=scaler,
                        stage_metrics=stage_metrics,
                    )

                if eval_every > 0 and global_step % eval_every == 0:
                    run_periodic_eval(
                        model=model,
                        tokenizer=tokenizer,
                        metas=metas,
                        mask_cache=mask_cache,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        plan_token_id=plan_token_id,
                        bos_id=bos_id,
                        eos_id=eos_id,
                        device=device,
                        global_step=global_step,
                        wandb_run=wandb_run,
                    )

                if global_step % SAVE_STEPS == 0:
                    save_checkpoint(
                        model,
                        tokenizer,
                        metas,
                        global_step,
                        OUTPUT_DIR,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                    )
            else:
                tokens_since_last_step = 0
                step_timer_start = time.perf_counter()

            progress_bar.update(1)
            elapsed = time.time() - start_time
            done = progress_bar.n
            if done > 0:
                tpb = elapsed / done
                remain = total_batches - done
                eta = timedelta(seconds=int(max(0, remain) * tpb))
                progress_bar.set_postfix({"epoch": f"{epoch + 1}/{EPOCHS}", "eta": str(eta)})

            batch = prefetcher.next()

    total_seconds = int(time.time() - start_time)
    total_time = timedelta(seconds=total_seconds)
    logging.info(f"[DONE] training complete, total time: {total_time}")
    save_checkpoint(
        model,
        tokenizer,
        metas,
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
