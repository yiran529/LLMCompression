from dataclasses import dataclass, field
from typing import List


@dataclass
class PathConfig:
    base_dir: str = "/root/lm_merge/qwen3_0.6b_z4096"
    parquet_path: str = "/root/data/wiki_en_sentences_flat.parquet"
    output_dir: str = "/root/lm_merge/train_runs/qwen_online_concept_v2"


@dataclass
class TrainConfig:
    max_samples: int = 100000
    max_input_tokens: int = 64
    batch_size: int = 8
    grad_accum: int = 2
    epochs: int = 1
    lr: float = 2e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    log_steps: int = 20
    save_steps: int = 500
    seed: int = 42
    tau_init: float = 1.0
    tau_min: float = 0.5
    num_workers: int = 4


@dataclass
class ModelConfig:
    concept_vocab_size: int = 4096
    shallow_layers: int = 8
    middle_layers: int = 8
    tail_mlp_hidden_ratio: int = 2
    very_neg: float = -1e4
    eps: float = 1e-8
    compression_ratio: float = 0.3
    beta_commit: float = 0.5
    lambda_rec: float = 1.0
    lambda_commit: float = 1.0
    lambda_unif: float = 2.0
    lambda_len: float = 0.1


@dataclass
class AdapterConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_candidates: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class ExperimentConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)

