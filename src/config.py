# -*- coding: utf-8 -*-
import os
import random
from dataclasses import dataclass

import torch
import torch.backends.cudnn as cudnn


USE_COMPILE = False
COMPILE_MODE = "max-autotune"

BASE_DIR = "/root/lm_merge/qwen3_0.6b_z4096"
PARQUET_PATH = "/root/data/wiki_en_sentences_flat.parquet"
OUTPUT_DIR = "/root/lm_merge/train_runs/concept_first_v1"

BATCH_SIZE = 48
GRAD_ACCUM = 4
EPOCHS = 3
LR = 3e-4
WARMUP_RATIO = 0.1
MAX_INPUT_TOKENS = 96
SEED = 42
SAVE_STEPS = 1000
LOG_STEPS = 100

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

# 0 is reserved for "normal text / neutral type".
TYPE_ID_TEXT = 0


@dataclass
class ConceptTypeConfig:
    name: str
    size: int
    max_steps: int
    target_ratio: float


CONCEPT_TYPE_CONFIGS = [
    ConceptTypeConfig(name="bottom", size=2048, max_steps=1000, target_ratio=0.45),
    ConceptTypeConfig(name="mid", size=256, max_steps=500, target_ratio=0.05),
    ConceptTypeConfig(name="top", size=32, max_steps=500, target_ratio=0.01),
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
