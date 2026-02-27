from src.config.train_config import *

# Input mode: "default_dataset" | "parquet" | "text_file"
INFER_INPUT_MODE = "text_file"

# Checkpoint from training, e.g. "./outputs/checkpoint-200" or "./outputs/final"
INFER_CKPT_DIR = "./outputs/final"

# Used only when checkpoint backbone is a LoRA adapter
INFER_BASE_DIR = BASE_DIR
INFER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INFER_DTYPE = MODEL_DTYPE  # "bf16" | "fp16" | "fp32"

INFER_BATCH_SIZE = 4
INFER_MAX_INPUT_TOKENS = MAX_INPUT_TOKENS
INFER_MAX_NEW_TOKENS = 64
INFER_PLANNER_TAU = 0.2
INFER_MIN_CONCEPT_STEPS = MIN_CONCEPT_STEPS
INFER_PLANNER_DETERMINISTIC = True
INFER_MAX_SAMPLES = 20
INFER_TEXT_COLUMN = "text"
INFER_SKIP_SPECIAL_TOKENS = True

# Input sources
INFER_PARQUET_PATH = PARQUET_PATH
INFER_TEXT_FILE = "./data/infer.txt"