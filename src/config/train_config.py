# -*- coding: utf-8 -*-
import os
import random
from dataclasses import dataclass

import torch
import torch.backends.cudnn as cudnn


USE_COMPILE = False
COMPILE_MODE = "reduce-overhead"

# A800-friendly defaults
# MODEL_DTYPE: "bf16" (recommended), "fp16", "fp32"
MODEL_DTYPE = "fp16"
# ATTENTION_IMPL: "flash_attention_2", "sdpa", "eager", or "" to let HF decide
ATTENTION_IMPL = "sdpa"
# FP32_TRAINABLE: "none", "lora_only", "all"
FP32_TRAINABLE = "lora_only"

# Gradient Checkpointing (Saves VRAM but slower training)
USE_GRADIENT_CHECKPOINTING = False

BASE_DIR = "/root/models/qwen3-0.6b"
PARQUET_PATH = "./data/wikipedia_512.parquet"
OUTPUT_DIR = "./outputs"

BATCH_SIZE = 40
GRAD_ACCUM = 8
EPOCHS = 3
LR = 1e-4
WARMUP_RATIO = 0.1
MAX_INPUT_TOKENS = 32
SEED = 42
SAVE_STEPS = 200
LOG_STEPS = 5
EVAL_STEPS = 100
EVAL_NUM_SAMPLES = 4
EVAL_MAX_NEW_TOKENS = 32
EVAL_PLANNER_TAU = 0.2
EVAL_SKIP_SPECIAL_TOKENS = True

# Stage-2 teacher-forcing input masking.
# Schedule:
# - first 10% steps: keep ratio at MIN
# - 10%~80% steps: linearly increase MIN -> MAX
# - last 20% steps: keep ratio at MAX
ENABLE_STAGE2_TF_MASKING = True
STAGE2_TF_MASKING_MAX_RATIO = 0.15
STAGE2_TF_MASKING_MIN_RATIO = 0.0

TAU_INIT = 1.0
TAU_MIN = 0.2
MIN_CONCEPT_STEPS = 4
# Planner sampling mode during training: "gumbel" | "greedy" | "mix".
TRAIN_PLANNER_SAMPLING_MODE = "mix"
# For TRAIN_PLANNER_SAMPLING_MODE == "mix", greedy row ratio linearly increases MIN -> MAX.
TRAIN_PLANNER_MIX_GREEDY_RATIO_MIN = 0.0
TRAIN_PLANNER_MIX_GREEDY_RATIO_MAX = 0.9
# 是否允许 Planner 输出 base tokens（原词表 tokens）
ALLOW_PLANNER_BASE_TOKENS = False

LAMBDA_REC = 1.0
LAMBDA_COMMIT = 0.1
LAMBDA_UNIF = 1.0
LAMBDA_EOS = 0.2
LAMBDA_LEN = 0.2
BETA_COMMIT = 0.5
EPS = 1e-8

# 0 is reserved for "normal text / neutral type".
TYPE_ID_TEXT = 0
TYPE_ID_CONCEPT = 1

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
class ConceptConfig:
    size: int
    max_steps: int
    target_ratio: float


CONCEPT_CONFIG = ConceptConfig(
    size=1024,
    max_steps=15,
    target_ratio=0.25,
)


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

############# wandb config #############
WANDB_PROJECT = "token-compression"
WANDB_RUN_NAME = "b99778c"
WANDB_ENTITY = ""
