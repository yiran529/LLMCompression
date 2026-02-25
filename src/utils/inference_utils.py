# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass
from typing import Iterable, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config.inference_config import *
from src.model import *
@dataclass
class InferenceRuntime:
    model: SharedBackboneUnifiedHead
    tokenizer: AutoTokenizer
    metas: List[ConceptTypeMeta]
    mask_cache: ConceptMaskCache
    plan_token_id: int
    bos_id: int
    eos_id: int
    device: str


def resolve_dtype(dtype_name: str, device: str) -> torch.dtype:
    if not str(device).startswith("cuda"):
        return torch.float32
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    return torch.float32


def load_texts_from_parquet(parquet_path: str, text_column: str, max_samples: int) -> List[str]:
    import pandas as pd

    df = pd.read_parquet(parquet_path, engine="pyarrow")
    assert text_column in df.columns, f"Parquet file missing column `{text_column}`."
    texts = df[text_column].astype(str).tolist()
    if max_samples > 0:
        texts = texts[:max_samples]
    return texts


def load_texts_from_file(text_file: str, max_samples: int) -> List[str]:
    with open(text_file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    if max_samples > 0:
        texts = texts[:max_samples]
    return texts


def iter_batches(items: List[str], batch_size: int) -> Iterable[List[str]]:
    assert batch_size > 0, "batch_size must be > 0."
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _load_backbone_model(
    *,
    backbone_dir: str,
    base_dir: str,
    tokenizer_size: int,
    device: str,
    model_dtype: torch.dtype,
) -> AutoModelForCausalLM:
    model_kwargs = {
        "torch_dtype": model_dtype if str(device).startswith("cuda") else torch.float32,
    }
    if str(device).startswith("cuda") and ATTENTION_IMPL:
        model_kwargs["attn_implementation"] = ATTENTION_IMPL

    adapter_config_path = os.path.join(backbone_dir, "adapter_config.json")
    if os.path.isfile(adapter_config_path):
        assert base_dir, "Checkpoint backbone is LoRA adapter, but INFER_BASE_DIR is empty."
        model_base = AutoModelForCausalLM.from_pretrained(base_dir, **model_kwargs).to(device)
        model_base.resize_token_embeddings(tokenizer_size, mean_resizing=False)
        return PeftModel.from_pretrained(model_base, backbone_dir, is_trainable=False).to(device)

    return AutoModelForCausalLM.from_pretrained(backbone_dir, **model_kwargs).to(device)


def _build_metas_from_state(head_state: dict, device: str) -> List[ConceptTypeMeta]:
    metas_state = head_state["concept_metas"]
    metas: List[ConceptTypeMeta] = []
    for item in metas_state:
        concept_ids = torch.tensor(item["concept_ids"], device=device, dtype=torch.long)
        eos_id = int(item["eos_id"])
        concept_ids_with_eos = torch.cat(
            [concept_ids, torch.tensor([eos_id], device=device, dtype=torch.long)], dim=0
        )
        metas.append(
            ConceptTypeMeta(
                name=item["name"],
                type_id=int(item["type_id"]),
                eos_id=eos_id,
                concept_ids=concept_ids,
                concept_ids_with_eos=concept_ids_with_eos,
                max_steps=int(item["max_steps"]),
                target_ratio=float(item["target_ratio"]),
            )
        )
    return metas


def load_runtime() -> InferenceRuntime:
    ckpt_dir = INFER_CKPT_DIR
    backbone_dir = os.path.join(ckpt_dir, "backbone")
    head_path = os.path.join(ckpt_dir, "two_stage_heads.pt")
    assert os.path.isdir(backbone_dir), f"Backbone dir not found: {backbone_dir}"
    assert os.path.isfile(head_path), f"Heads checkpoint not found: {head_path}"

    device = INFER_DEVICE
    model_dtype = resolve_dtype(INFER_DTYPE, device=device)

    tokenizer = AutoTokenizer.from_pretrained(backbone_dir, use_fast=True)
    head_state = torch.load(head_path, map_location=device)

    tokenizer_size = int(head_state.get("tokenizer_size", len(tokenizer)))
    assert len(tokenizer) == tokenizer_size, (
        f"Tokenizer size mismatch: tokenizer={len(tokenizer)} vs checkpoint={tokenizer_size}."
    )
    metas = _build_metas_from_state(head_state, device=device)
    num_type_embeddings = 1 + len(metas)
    base_vocab_size = int(head_state["output_head_base"]["weight"].shape[0])

    model_base = _load_backbone_model(
        backbone_dir=backbone_dir,
        base_dir=INFER_BASE_DIR,
        tokenizer_size=len(tokenizer),
        device=device,
        model_dtype=model_dtype,
    )

    model = SharedBackboneUnifiedHead(
        model_base,
        num_type_embeddings=num_type_embeddings,
        frozen_output_head_prefix_rows=base_vocab_size,
    ).to(device)

    model.output_head_base.load_state_dict(head_state["output_head_base"])
    if model.output_head_new is not None:
        model.output_head_new.load_state_dict(head_state["output_head_new"])
    model.type_embed.load_state_dict(head_state["type_embed"])
    token_embed_new = head_state.get("token_embed_new")
    if token_embed_new is not None and token_embed_new.numel() > 0:
        expected_new_rows = model.vocab_size - model.base_vocab_size
        assert token_embed_new.shape[0] == expected_new_rows, (
            f"token_embed_new rows mismatch: ckpt={token_embed_new.shape[0]} vs model={expected_new_rows}"
        )
        assert model.token_embed_new is not None, "Checkpoint has token_embed_new but model has no new-token embedding rows."
        model.token_embed_new.weight.data.copy_(token_embed_new.to(
            device=device,
            dtype=model.token_embed_new.weight.dtype,
        ))
    else:
        print("[WARN] `token_embed_new` not found in two_stage_heads.pt; new token embeddings may be random.")

    plan_token_id = tokenizer.convert_tokens_to_ids("<PLAN>")
    assert plan_token_id is not None and plan_token_id >= 0, "Tokenizer does not contain `<PLAN>` token."

    bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id or tokenizer.bos_token_id
    assert bos_id is not None and eos_id is not None, "Tokenizer must have BOS/EOS token id."

    blocked_ids = build_executor_blocklist(metas, plan_token_id=plan_token_id)
    vocab_size = model.vocab_size
    mask_cache = ConceptMaskCache(
        metas=metas,
        vocab_size=vocab_size,
        base_vocab_size=model.base_vocab_size,
        blocked_for_executor=blocked_ids,
        device=device,
    )
    model.eval()

    return InferenceRuntime(
        model=model,
        tokenizer=tokenizer,
        metas=metas,
        mask_cache=mask_cache,
        plan_token_id=plan_token_id,
        bos_id=int(bos_id),
        eos_id=int(eos_id),
        device=device,
    )


def get_input_texts() -> List[str]:
    assert INFER_INPUT_MODE in {"default_dataset", "parquet", "text_file"}, (
        "INFER_INPUT_MODE must be one of: default_dataset, parquet, text_file"
    )

    if INFER_INPUT_MODE == "default_dataset":
        texts = load_texts_from_parquet(INFER_PARQUET_PATH, INFER_TEXT_COLUMN, INFER_MAX_SAMPLES)
    elif INFER_INPUT_MODE == "parquet":
        texts = load_texts_from_parquet(INFER_PARQUET_PATH, INFER_TEXT_COLUMN, INFER_MAX_SAMPLES)
    else:
        texts = load_texts_from_file(INFER_TEXT_FILE, INFER_MAX_SAMPLES)

    assert texts, "No input texts found."
    return texts


def trim_to_first_eos(token_ids: List[int], eos_id: int) -> List[int]:
    if eos_id in token_ids:
        eos_pos = token_ids.index(eos_id)
        return token_ids[:eos_pos]
    return token_ids


def format_concepts(
    tokenizer: AutoTokenizer,
    metas: List[ConceptTypeMeta],
    planner_out,
    sample_idx: int,
) -> List[str]:
    lines: List[str] = []
    for meta, type_out in zip(metas, planner_out.per_type):
        valid = type_out.valid_mask[sample_idx].to(torch.bool)
        ids = type_out.token_ids[sample_idx][valid].tolist()
        toks = tokenizer.convert_ids_to_tokens(ids)
        lines.append(f"{meta.name}: ids={ids} tokens={' '.join(toks) if toks else '(empty)'}")
    return lines

