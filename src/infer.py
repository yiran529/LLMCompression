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
        )

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
