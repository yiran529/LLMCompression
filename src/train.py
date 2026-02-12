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

from src.config import *
from src.model import *

def setup_logging(output_dir: str) -> str:
    """configure file+stdout logging and return the log file path."""
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return log_file

class ParquetSentenceDataset(Dataset):
    def __init__(self, parquet_path: str, max_samples: int = None):
        """load `sentence` column from parquet and optionally truncate."""
        df = pd.read_parquet(parquet_path, engine="pyarrow")
        assert "sentence" in df.columns, "Parquet must include a 'sentence' column."
        self.sentences = df["sentence"].astype(str).tolist()
        if max_samples is not None:
            self.sentences = self.sentences[:max_samples]
        logging.info(f"[INFO] loaded samples: {len(self.sentences)}")

    def __len__(self) -> int:
        """return number of usable training examples."""
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        """return one raw text sample as {'sentence': str}."""
        return {"sentence": self.sentences[idx]}

@dataclass
class Collator:
    tokenizer: AutoTokenizer
    max_len: int

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """tokenize a mini-batch into padded `input_ids` and `attention_mask`."""
        texts = [ex["sentence"] for ex in batch]
        tok = self.tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            padding=True,
        )
        return {
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
        }

def get_adaptive_tau(global_step: int, total_steps: int, tau_init: float, tau_min: float) -> float:
    """anneal Gumbel-Softmax temperature from `tau_init` to `tau_min`."""
    ratio = global_step / max(1, total_steps)
    if ratio < 0.6:
        local = ratio / 0.6
        return tau_init - (tau_init - 0.4) * local
    local = (ratio - 0.6) / 0.4
    return 0.4 - (0.4 - tau_min) * local

class CUDAPrefetcher:
    def __init__(self, loader: DataLoader, device: str = "cuda"):
        """overlap host->device transfer with compute using a side CUDA stream."""
        self.loader_iter = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream(device=device) if torch.cuda.is_available() else None
        self.next_batch = None
        self._preload()

    def _preload(self):
        """asynchronously stage the next batch to target device."""
        try:
            batch = next(self.loader_iter)
        except StopIteration:
            self.next_batch = None
            return
        if self.stream is None:
            self.next_batch = batch
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = {
                "input_ids": batch["input_ids"].to(self.device, non_blocking=True),
                "attention_mask": batch["attention_mask"].to(self.device, non_blocking=True),
            }

    def next(self):
        """return current prepared batch and trigger preload of the following batch."""
        if self.next_batch is None:
            return None
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self._preload()
        return batch

def save_checkpoint(
    model: SharedBackboneUnifiedHead,
    tokenizer: AutoTokenizer,
    metas: List[ConceptTypeMeta],
    step: int,
    output_dir: str,
    optimizer=None,
    scheduler=None,
    scaler=None,
):
    """persist backbone/tokenizer and lightweight planner metadata."""
    save_name = f"checkpoint-{step}" if isinstance(step, int) else str(step)
    save_dir = os.path.join(output_dir, save_name)
    backbone_dir = os.path.join(save_dir, "backbone")
    os.makedirs(backbone_dir, exist_ok=True)

    model.base_model.save_pretrained(backbone_dir)
    tokenizer.save_pretrained(backbone_dir)

    meta_dump = [
        {
            "name": m.name,
            "type_id": m.type_id,
            "eos_id": m.eos_id,
            "concept_ids": m.concept_ids.tolist(),
            "max_steps": m.max_steps,
            "target_ratio": m.target_ratio,
        }
        for m in metas
    ]

    planner_quota_state = None
    if hasattr(model, "planner_quota") and model.planner_quota is not None:
        planner_quota_state = model.planner_quota.state_dict()

    torch.save(
        {
            "output_head": model.output_head.state_dict(),
            "type_embed": model.type_embed.state_dict(),
            "concept_metas": meta_dump,
            "planner_quota": planner_quota_state,
            "step": step,
            "tokenizer_size": len(tokenizer),
        },
        os.path.join(save_dir, "two_stage_heads.pt"),
    )
    if optimizer is not None and scheduler is not None and isinstance(step, int):
        trainer_state = {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        if scaler is not None and scaler.is_enabled():
            trainer_state["scaler"] = scaler.state_dict()
        torch.save(trainer_state, os.path.join(save_dir, "trainer_state.pt"))
    logging.info(f"[SAVE] checkpoint saved to: {save_dir}")

def train():
    """run two-stage concept-first training with a shared backbone and one output head."""
    log_file = setup_logging(OUTPUT_DIR)
    logging.info(f"[INFO] log file: {log_file}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"[INFO] device: {device}")
    resume_ckpt_dir = RESUME_CHECKPOINT_DIR.strip()
    resume_backbone_dir = os.path.join(resume_ckpt_dir, "backbone") if resume_ckpt_dir else ""
    resume_heads_path = os.path.join(resume_ckpt_dir, "two_stage_heads.pt") if resume_ckpt_dir else ""
    resume_trainer_state_path = os.path.join(resume_ckpt_dir, "trainer_state.pt") if resume_ckpt_dir else ""
    resume_step = 0
    if RESUME_ENABLED:
        assert os.path.isdir(resume_backbone_dir), f"resume backbone dir not found: {resume_backbone_dir}"
        assert os.path.isfile(resume_heads_path), f"resume heads file not found: {resume_heads_path}"
        assert os.path.isfile(resume_trainer_state_path), f"resume trainer_state file not found: {resume_trainer_state_path}"
        logging.info(f"[INFO] resume enabled from: {resume_ckpt_dir}")

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

    model_kwargs = {
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }
    if torch.cuda.is_available():
        # Optional acceleration when current GPU/runtime supports FA.
        model_kwargs["attn_implementation"] = "sdpa"
    model_base = AutoModelForCausalLM.from_pretrained(BASE_DIR, **model_kwargs).to(device)
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
        device=device,
    )
    if RESUME_ENABLED:
        head_state = torch.load(resume_heads_path, map_location=device)
        if "output_head" in head_state:
            model.output_head.load_state_dict(head_state["output_head"])
        else:
            raise RuntimeError("No output head state found in checkpoint.")
        model.type_embed.load_state_dict(head_state["type_embed"])
        if "planner_quota" in head_state and head_state["planner_quota"] is not None:
            model.planner_quota.load_state_dict(head_state["planner_quota"], device=device)
        saved_step = head_state.get("step", 0)
        if isinstance(saved_step, int):
            resume_step = saved_step

    if USE_COMPILE:
        try:
            compiled = torch.compile(model, mode=COMPILE_MODE, fullgraph=False)
            compiled.planner_quota = model.planner_quota
            model = compiled
            logging.info(f"[INFO] torch.compile enabled ({COMPILE_MODE})")
        except Exception as e:
            logging.warning(f"[WARN] torch.compile failed: {e}")

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

    dataset = ParquetSentenceDataset(PARQUET_PATH, max_samples=100000)
    collate_fn = Collator(tokenizer=tokenizer, max_len=MAX_INPUT_TOKENS)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        pin_memory_device="cuda", # Supported only in PyTorch 2.0+ and when CUDA is available
        collate_fn=collate_fn,
        drop_last=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    optim_params = [p for p in model.parameters() if p.requires_grad]
    if not optim_params:
        raise RuntimeError("No trainable parameters found. Check LoRA/parameter-freeze setup.")
    output_head_param = model.output_head.weight
    base_params = [p for p in optim_params if id(p) != id(output_head_param)]
    optimizer_groups = []
    if base_params:
        optimizer_groups.append(
            {"params": base_params, "lr": LR, "betas": (0.9, 0.95), "weight_decay": 0.01}
        )
    optimizer_groups.append(
        {"params": [output_head_param], "lr": LR, "betas": (0.9, 0.95), "weight_decay": 0.0}
    )
    optimizer = torch.optim.AdamW(optimizer_groups)
    total_steps = math.ceil(len(dataloader) / GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if torch.cuda.is_available() else None
    if RESUME_ENABLED and os.path.isfile(resume_trainer_state_path):
        trainer_state = torch.load(resume_trainer_state_path, map_location="cpu")
        if "optimizer" in trainer_state:
            optimizer.load_state_dict(trainer_state["optimizer"])
        if "scheduler" in trainer_state:
            scheduler.load_state_dict(trainer_state["scheduler"])
        if scaler is not None and scaler.is_enabled() and "scaler" in trainer_state:
            scaler.load_state_dict(trainer_state["scaler"])
        resume_step = int(trainer_state.get("step", resume_step))
    elif RESUME_ENABLED:
        logging.warning(
            f"[WARN] trainer_state.pt not found at {resume_trainer_state_path}, "
            "resume without optimizer/scheduler/scaler state"
        )

    logging.info(f"[INFO] vocab size: {vocab_size}")
    logging.info(f"[INFO] output_head frozen rows: [0, {base_vocab_size})")
    new_rows = max(0, vocab_size - base_vocab_size)
    effective_head_params = new_rows * model.output_head.weight.shape[1]
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

    model.train()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_batches = (len(dataloader) * EPOCHS) / GRAD_ACCUM
    progress_bar = tqdm(total=total_batches, desc="training", unit="step")
    start_time = time.time()

    global_step = resume_step
    for epoch in range(EPOCHS):
        epoch_losses: List[float] = []
        optimizer.zero_grad(set_to_none=True)

        micro_step = 0
        prefetcher = CUDAPrefetcher(dataloader, device=device)
        batch = prefetcher.next()

        while batch is not None:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            tau = get_adaptive_tau(global_step, total_steps, TAU_INIT, TAU_MIN)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                # Stage 1 (Planner): source text -> typed concept sequences.
                planner_out = plan_concepts(
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

                decoder_in, decoder_mask, labels, src_lengths = build_decoder_tensors(
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
                logits = model.output_head(hidden)
                # Executor should not produce planner-only tokens.
                logits = logits.masked_fill(mask_cache.executor_block_bool.view(1, 1, -1), -1e4)

                loss_rec = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
                loss_commit, loss_unif, loss_eos, loss_len = compute_planner_losses(
                    planner_out=planner_out,
                    metas=metas,
                    src_lengths=src_lengths,
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
                    + LAMBDA_LEN * loss_len
                    + loss_quota
                ).float()

            if not torch.isfinite(loss):
                logging.warning("[WARN] non-finite loss, skip batch")
                optimizer.zero_grad(set_to_none=True)
                batch = prefetcher.next()
                continue

            loss_scaled = loss / GRAD_ACCUM
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            micro_step += 1
            is_last_in_epoch = prefetcher.next_batch is None
            do_step = (micro_step % GRAD_ACCUM == 0) or is_last_in_epoch
            if not do_step:
                batch = prefetcher.next()
                continue

            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)

            trainable_with_grad = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not trainable_with_grad:
                optimizer.zero_grad(set_to_none=True)
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

            optimizer.zero_grad(set_to_none=True)

            if did_step:
                scheduler.step()
                global_step += 1
                epoch_losses.append(float(loss.detach().cpu()))

                if global_step % LOG_STEPS == 0:
                    recent = epoch_losses[-min(LOG_STEPS, len(epoch_losses)) :]
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
                        f"LR {scheduler.get_last_lr()[0]:.2e}"
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

            progress_bar.update(1)
            elapsed = time.time() - start_time
            done = progress_bar.n
            if done > 0:
                tpb = elapsed / done
                remain = total_batches - done
                eta = timedelta(seconds=int(max(0, remain) * tpb))
                progress_bar.set_postfix({"epoch": f"{epoch + 1}/{EPOCHS}", "eta": str(eta)})

            batch = prefetcher.next()

    progress_bar.close()
    total_time = timedelta(seconds=int(time.time() - start_time))
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

if __name__ == "__main__":
    train()
