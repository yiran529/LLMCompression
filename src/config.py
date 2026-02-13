# -*- coding: utf-8 -*-
import os
import random
from dataclasses import dataclass

import torch
import torch.backends.cudnn as cudnn


USE_COMPILE = True
COMPILE_MODE = "reduce-overhead"

# A800-friendly defaults
# MODEL_DTYPE: "bf16" (recommended), "fp16", "fp32"
MODEL_DTYPE = "bf16"
# ATTENTION_IMPL: "flash_attention_2", "sdpa", "eager", or "" to let HF decide
ATTENTION_IMPL = "flash_attention_2"
# FP32_TRAINABLE: "none", "lora_only", "all"
FP32_TRAINABLE = "none"

BASE_DIR = "./qwen3-0.6b"
PARQUET_PATH = "./data/wikipedia_512.parquet"
OUTPUT_DIR = "./outputs"

BATCH_SIZE = 12
GRAD_ACCUM = 8
EPOCHS = 3
LR = 5e-4
WARMUP_RATIO = 0.1
MAX_INPUT_TOKENS = 128
SEED = 42
SAVE_STEPS = 200
LOG_STEPS = 5

TAU_INIT = 0.8
TAU_MIN = 0.2
MIN_CONCEPT_STEPS = 1

LAMBDA_REC = 1.0
LAMBDA_COMMIT = 0.5
LAMBDA_UNIF = 0.2
LAMBDA_EOS = 0.2
LAMBDA_LEN = 0.2
BETA_COMMIT = 0.5
EPS = 1e-8
QUOTA_TAU = 0.15
QUOTA_ETA = 0.01
QUOTA_LAMBDA_INIT = 0.0

# 0 is reserved for "normal text / neutral type".
TYPE_ID_TEXT = 0

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
LORA_MODULES_TO_SAVE = []

# Resume
RESUME_ENABLED = False
RESUME_CHECKPOINT_DIR = ""


@dataclass
class ConceptTypeConfig:
    name: str
    size: int
    max_steps: int
    target_ratio: float


CONCEPT_TYPE_CONFIGS = [
    ConceptTypeConfig(name="bottom", size=2048, max_steps=64, target_ratio=0.23),
    ConceptTypeConfig(name="mid", size=256, max_steps=20, target_ratio=0.04),
    ConceptTypeConfig(name="top", size=32, max_steps=8, target_ratio=0.02),
]


def apply_runtime_settings() -> None:
    """apply runtime perf switches and deterministic seeds."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    cudnn.benchmark = True
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_cache")

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


apply_runtime_settings()
