# -*- coding: utf-8 -*-
import os
import math
import random
import time
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import PeftModel

# =========================
# 日志配置
# =========================
def setup_logging(output_dir):
    """配置日志，同时输出到控制台和文件"""
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_file

# =========================
# 全局性能设置
# =========================
import torch.backends.cudnn as cudnn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
cudnn.benchmark = True
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_cache")

USE_COMPILE = False
COMPILE_MODE = "max-autotune"

# =========================
# 用户需按需修改的路径/超参
# =========================
BASE_DIR = "/root/autodl-tmp/qwen3_0.6b_z4096"
COMPRESS_ADAPTER_DIR = "/root/lm_merge/lora_compress/qwen3-0.6b_r128"
DECOMPRESS_ADAPTER_DIR = "/root/lm_merge/lora_decompress/qwen3-0.6b_r128"
PARQUET_PATH = "/root/autodl-tmp/wikipedia/wikipedia_512.parquet"
OUTPUT_DIR = "/root/lm_merge/train_runs/z128_d100_0.6b_4"
RESUME_CHECKPOINT = ""
# Z 词表设置
# Z_ID_START = 151669
# Z_ID_END   = 151796
# EOZ_ID     = 151797
Z_ID_START = 151669
Z_ID_END   = 155764
EOZ_ID     = 155765
assert Z_ID_START < Z_ID_END < EOZ_ID, "Z / EOZ ID 范围异常，请核对"
ORIG_VOCAB_SIZE = Z_ID_START      # 151669

# 训练超参
BATCH_SIZE =12
GRAD_ACCUM = 1
EPOCHS = 20
LR = 5e-5
WARMUP_RATIO = 0.1
MAX_INPUT_TOKENS = 512
MAX_Z_STEPS = 128
TAU_INIT = 0.7
TAU_MIN = 0.2
LAMBDA_REC = 1.0
LAMBDA_Z_RECON = 1.0   # Z-only重建(截断on-policy)损失权重
LAMBDA_DIVERSITY = 1.5
LAMBDA_SPREAD = 1.0
SEED = 42
SAVE_STEPS = 160
LOG_STEPS = 10
BETA_COMMIT = 0.5
LAMBDA_COMMIT = 1.0
LAMBDA_UNIF = 2.0
LAMBDA_MARGIN = 2.0
EPS = 1e-8

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =========================
# 数据集
# =========================
class ParquetSentenceDataset(Dataset):
    def __init__(self, parquet_path: str, max_samples: int = None):
        df = pd.read_parquet(parquet_path, engine="pyarrow")
        assert "text" in df.columns, "Parquet 里必须有 'sentence' 列"
        self.sentences = df["text"].astype(str).tolist()
        if max_samples:
            self.sentences = self.sentences[:max_samples]
        logging.info(f"[INFO] 加载了 {len(self.sentences)} 个句子")

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {"sentence": self.sentences[idx]}

@dataclass
class Collator:
    tokenizer: AutoTokenizer
    max_len: int

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
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

# =========================
# 工具函数
# =========================
def gumbel_softmax_sample(logits: torch.Tensor, tau: float, hard: bool = False) -> torch.Tensor:
    try:
        u = torch.empty_like(logits).uniform_(1e-8, 1-1e-8)
        g = -torch.log(-torch.log(u + 1e-10) + 1e-10)
        y = (logits + g) / max(tau, 1e-4)
        y = y - y.max(dim=-1, keepdim=True).values
        y = torch.clamp(y, min=-50, max=50)
        y_soft = F.softmax(y, dim=-1)
        y_soft = torch.nan_to_num(y_soft, nan=0.0, posinf=0.0, neginf=0.0)
        y_soft = y_soft / (y_soft.sum(dim=-1, keepdim=True) + 1e-8)
        if hard:
            idx = y_soft.argmax(dim=-1, keepdim=True)
            y_hard = torch.zeros_like(y_soft).scatter_(-1, idx, 1.0)
            return y_hard - y_soft.detach() + y_soft
        return y_soft
    except Exception as e:
        logging.warning(f"Gumbel softmax error: {e}")
        return torch.ones_like(logits) / logits.size(-1)

class MaskCache:
    def __init__(self, vocab_size, z_start, z_end, eoz_id, device):
        self.vocab_size = vocab_size
        self.device = device
        very_neg = -1e4
        self.z_mask = torch.full((vocab_size,), very_neg, device=device, dtype=torch.float16)
        self.z_mask[z_start:z_end+1] = 0.0
        self.z_mask[eoz_id] = 0.0
        self.non_z_mask_bool = torch.zeros(vocab_size, device=device, dtype=torch.bool)
        self.non_z_mask_bool[z_start:eoz_id+1] = True
        self.z_indices = torch.arange(z_start, z_end+1, device=device, dtype=torch.long)
        self.z_indices_with_eoz = torch.cat([self.z_indices, torch.tensor([eoz_id], device=device, dtype=torch.long)])
        self.eoz_id = eoz_id
        self.z_start = z_start
        self.z_end = z_end
        self.num_z_tokens = z_end - z_start + 1

def commitment_loss(e_soft: torch.Tensor, e_hard: torch.Tensor, beta: float) -> torch.Tensor:
    loss1 = (e_soft.detach() - e_hard).pow(2).mean()
    loss2 = (e_soft - e_hard.detach()).pow(2).mean()
    return loss1 + beta * loss2

def usage_kl_to_uniform(z_hist: torch.Tensor) -> torch.Tensor:
    Z = z_hist.numel()
    u = torch.full_like(z_hist, 1.0 / max(1, Z))
    kl = torch.sum(z_hist * (torch.log(z_hist + EPS) - torch.log(u + EPS)))
    return kl

def codebook_spread_loss(Ez):
    z = F.normalize(Ez.float(), dim=1)
    G = z @ z.t()
    off = G - torch.eye(G.size(0), device=G.device)
    return off.pow(2).mean()

def get_adaptive_tau(global_step, total_steps, tau_init=0.8, tau_min=0.2):
    ratio = global_step / max(1, total_steps)
    if ratio < 0.6:
        local_ratio = ratio / 0.6
        return tau_init - (tau_init - 0.4) * local_ratio
    else:
        local_ratio = (ratio - 0.6) / 0.4
        return 0.4 - (0.4 - tau_min) * local_ratio

# [CHANGED] 1) 让 scheduled sampling 真正生效：逐步从 TF 过渡到 on-policy
def get_scheduled_sampling_prob(global_step, total_steps):
    """
    前50%步数：纯Teacher Forcing（自回归概率0%）
    后50%步数：从0%线性提升至40%自回归概率
    """
    r = global_step / max(1, total_steps)  # 全局进度比例（0~1）
    
    if r < 1:
        # 前50%步数：完全使用Teacher Forcing
        return 0.0
    # else:
    #     # 后50%步数：线性从0%升到40%
    #     # 先将r映射到0~1区间（仅针对后50%）
    #     normalized_r = (r - 0.5) / 0.5  # 当r=0.5时为0，r=1.0时为1
    #     return 0.4 * normalized_r  # 乘以目标最大值40%（0.4）

class SlotPooler(nn.Module):
    """
    一次前向：用 K 个可学习 query 去“读”整段上下文的最后层隐藏态，得到 K 个槽位向量。
    然后投影到 Z 词表（含 EOZ）logits，再用 Gumbel-Softmax 得到离散 one-hot，查表拿到 Z-embeds。
    """
    def __init__(self, hidden_size: int, num_slots: int, z_vocab_size: int):
        super().__init__()
        self.num_slots = num_slots
        self.q = nn.Parameter(torch.randn(num_slots, hidden_size) * 0.02)  # K×H 可学习 query
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=max(1, hidden_size // 64), batch_first=True)
        self.proj = nn.Linear(hidden_size, z_vocab_size)  # 映射到 |Z|（含EOZ）个类

        # 让 query 初始更稳定
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, ctx_states: torch.Tensor, attn_mask: torch.Tensor, tau: float):
        """
        ctx_states: [B, L, H]  (compress适配器得到的最后层隐藏态；建议去掉<COMP>/<BOS>前缀再喂进来)
        attn_mask : [B, L]     (1=有效, 0=pad)
        return:
          z_logits:   [B, K, |Z|]
          z_probs_h:  [B, K, |Z|]  (hard Gumbel one-hot + straight-through)
          slots:      [B, K, H]
        """
        B, L, H = ctx_states.shape
        # 扩展 K 个 query 到 batch 维
        q = self.q.unsqueeze(0).expand(B, -1, -1)  # [B, K, H]

        # MultiheadAttention 需要 key_padding_mask: True为padding
        # 这里让 query 只看上下文，不互看：用 cross-attn (q vs ctx_states)
        key_padding_mask = (attn_mask == 0)  # [B, L], True=pad
        slots, _ = self.attn(q, ctx_states, ctx_states, key_padding_mask=key_padding_mask)  # [B, K, H]

        z_logits = self.proj(slots)  # [B, K, |Z|]
        # 外部再做 mask（限制到 Z∪EOZ），这里不做
        return z_logits, slots
    
# =========================
# Scheduled Sampling 解码（主重建分支）
# =========================
def scheduled_sampling_decode(
    model,
    z_input,
    z_mask,
    targets,
    base_embed,
    sampling_prob,
    vocab_size,
    mask_cache,
    device
):
    B, T = targets.shape
    if sampling_prob < 1e-6:
        tf_inputs = targets[:, :-1]
        tf_input_embeds = base_embed(tf_inputs)
        full_embeds = torch.cat([z_input, tf_input_embeds], dim=1)
        full_mask = torch.cat([z_mask, torch.ones_like(tf_inputs)], dim=1)
        return full_embeds, full_mask

    generated_embeds = []
    current_embeds = z_input
    current_mask = z_mask

    for t in range(T - 1):
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=True):
            out = model(
                inputs_embeds=current_embeds,
                attention_mask=current_mask,
                use_cache=False
            )
        next_token_logits = out.logits[:, -1, :]
        use_model_pred = torch.rand(B, device=device) < sampling_prob
        pred_tokens = torch.argmax(next_token_logits, dim=-1)
        true_tokens = targets[:, t]
        next_tokens = torch.where(use_model_pred, pred_tokens, true_tokens)
        next_embeds = base_embed(next_tokens.unsqueeze(1))
        generated_embeds.append(next_embeds)
        current_embeds = torch.cat([current_embeds, next_embeds], dim=1)
        current_mask = torch.cat([current_mask, torch.ones((B,1), device=device, dtype=torch.long)], dim=1)

    all_gen_embeds = torch.cat(generated_embeds, dim=1)
    full_embeds = torch.cat([z_input, all_gen_embeds], dim=1)
    full_mask = torch.cat([z_mask, torch.ones((B, T-1), device=device, dtype=torch.long)], dim=1)
    return full_embeds, full_mask

# =========================
# 截断 on-policy：从 Z 前缀 roll-in 真值到随机 t0，再做 K 步自回归并计算 CE
# （这是你原先的 compute_z_only_recon_truncated，直接用于“重建 on-policy 辅助”）
# =========================
def compute_z_only_recon_truncated(
    model, z_embeddings, input_ids, attention_mask, base_embed, mask_cache,
    bos_id, decomp_id, eos_id, device,
    k_steps: int = 8,
    rollin_min: float = 0.3,
    rollin_max: float = 0.9,
    forbid_z_out: bool = True,
    candidate_topk: int | None = None
):
    B = z_embeddings.size(0)
    lengths = attention_mask.sum(dim=1).to(torch.long)
    Lmax = int(lengths.max().item())

    decomp_embed = base_embed(torch.full((B,1), decomp_id, device=device, dtype=torch.long))
    bos_embed    = base_embed(torch.full((B,1), bos_id,    device=device, dtype=torch.long))
    prefix_embeds = torch.cat([decomp_embed, bos_embed, z_embeddings], dim=1)

    out0 = model(inputs_embeds=prefix_embeds, use_cache=True)
    past_kv = out0.past_key_values
    logits_t = out0.logits[:, -1, :]

    t0_list = []
    for b in range(B):
        Lb = int(lengths[b].item())
        lo = int(Lb * rollin_min)
        hi = max(lo+1, int(Lb * rollin_max))
        t0 = random.randint(lo, max(lo, hi-1))
        t0_list.append(t0)
    t0 = torch.tensor(t0_list, device=device, dtype=torch.long)

    if Lmax > 0:
        max_t0 = int(t0.max().item())
        if max_t0 > 0:
            rollin_tokens = []
            for b in range(B):
                Lb = int(lengths[b].item())
                tb = int(min(t0[b].item(), Lb))
                seg = input_ids[b, :tb]
                pad = torch.full((max_t0 - tb,), eos_id, device=device, dtype=torch.long)
                rollin_tokens.append(torch.cat([seg, pad], dim=0))
            rollin_tokens = torch.stack(rollin_tokens, dim=0)
            rollin_embeds = base_embed(rollin_tokens)
            out_ri = model(inputs_embeds=rollin_embeds, past_key_values=past_kv, use_cache=True)
            past_kv = out_ri.past_key_values
            logits_t = out_ri.logits[:, -1, :]

    loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    valid_cnt = 0
    for step in range(k_steps):
        if forbid_z_out:
            logits_t = logits_t.masked_fill(mask_cache.non_z_mask_bool.view(1, -1), -1e4)
        tgt = torch.full((B,), -100, device=device, dtype=torch.long)
        for b in range(B):
            pos = int(t0[b].item()) + step
            if pos < lengths[b]:
                tgt[b] = input_ids[b, pos]
            elif pos == lengths[b]:
                tgt[b] = eos_id
        loss_t = F.cross_entropy(logits_t.float(), tgt, ignore_index=-100, reduction='sum')
        loss_sum += loss_t
        valid_cnt += int((tgt != -100).sum().item())
        next_tok = torch.argmax(logits_t, dim=-1)
        next_emb = base_embed(next_tok.unsqueeze(1))
        out_next = model(inputs_embeds=next_emb, past_key_values=past_kv, use_cache=True)
        past_kv = out_next.past_key_values
        logits_t = out_next.logits[:, -1, :]

    return (loss_sum / max(1, valid_cnt))


# =========================
# CUDA 异步预取器
# =========================
class CUDAPrefetcher:
    def __init__(self, loader, device="cuda"):
        self.loader_iter = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream(device=device) if torch.cuda.is_available() else None
        self.next_batch = None
        self._preload()

    def _preload(self):
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
        if self.next_batch is None:
            return None
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self._preload()
        return batch

# =========================
# 训练主体
# =========================
def train():
    log_file = setup_logging(OUTPUT_DIR)
    logging.info(f".[INFO] 日志文件已保存至: {log_file}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"[INFO] 使用设备: {device}")

    logging.info("[INFO] 加载模型和tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=True)

    # 注册模式标签
    added = tokenizer.add_special_tokens({
        "additional_special_tokens": ["<COMP>", "<DECOMP>"]
    })
    if added:
        logging.info(f"[INFO] 新增特殊token数: {added}")
    COMP_ID = tokenizer.convert_tokens_to_ids("<COMP>")
    DECOMP_ID = tokenizer.convert_tokens_to_ids("<DECOMP>")

    model_base = AutoModelForCausalLM.from_pretrained(
        BASE_DIR,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    model_base.resize_token_embeddings(len(tokenizer))

    # 新增：加载之前训练的扩展权重（嵌入层和LM Head的扩展部分）
    if RESUME_CHECKPOINT:
        new_weights_path = os.path.join(RESUME_CHECKPOINT + "_new_weights.pt")
        assert os.path.exists(new_weights_path), f"未找到权重文件：{new_weights_path}"
        new_weights = torch.load(new_weights_path, map_location=device)
        # 覆盖嵌入层的扩展部分（ORIG_VOCAB_SIZE之后的行）
        embed_weight = model_base.get_input_embeddings().weight
        embed_weight.data[ORIG_VOCAB_SIZE:] = new_weights["new_embeddings"].to(embed_weight.dtype).to(device)
        # 覆盖LM Head的扩展部分（ORIG_VOCAB_SIZE之后的行）
        lm_head_weight = model_base.lm_head.weight
        lm_head_weight.data[:] = new_weights["new_lm_head"].to(lm_head_weight.dtype).to(device)
        logging.info(f"[INFO] 已加载扩展权重：{new_weights_path}，orig_vocab_size={new_weights['orig_vocab_size']}")

    OLD_VOCAB_SIZE = 15668
    NEW_VOCAB_SIZE = len(tokenizer)
    logging.info(f"[INFO] 词表大小: 原始 {OLD_VOCAB_SIZE}, 新 {NEW_VOCAB_SIZE}, COMP={COMP_ID}, DECOMP={DECOMP_ID}")

    embed_weight = model_base.get_input_embeddings().weight
    lm_head_weight = model_base.lm_head.weight
    embed_weight.requires_grad = True
    lm_head_weight.requires_grad = True

    def create_embed_hook(param_shape, device):
        vocab_size = param_shape[0]
        def hook(grad):
            if grad is None:
                return None
            grad = torch.clamp(grad, min=-10.0, max=10.0)
            mask_1d = torch.zeros(vocab_size, device=device, dtype=grad.dtype)
            mask_1d[ORIG_VOCAB_SIZE:] = 1.0  # 仅扩展行(含Z/EOZ/<COMP>/<DECOMP>)
            return grad * mask_1d.unsqueeze(1) if grad.dim()==2 else grad * mask_1d
        return hook

    embed_weight.register_hook(create_embed_hook(embed_weight.shape, device))

    logging.info("[INFO] 加载 LoRA adapters...")
    model = PeftModel.from_pretrained(
        model_base,
        COMPRESS_ADAPTER_DIR,
        adapter_name="compress",
        is_trainable=True
    )
    model.load_adapter(
        DECOMPRESS_ADAPTER_DIR,
        adapter_name="decompress",
        is_trainable=True
    )
    model.to(device)
    trainable_para = 0
    for name, param in model.named_parameters():
        if "lora" not in name and "embed_tokens" not in name and "lm_head" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            trainable_para += 1
    logging.info(f"all trainable_para is {trainable_para}")

    if USE_COMPILE:
        try:
            model = torch.compile(model, mode=COMPILE_MODE, fullgraph=False)
            logging.info(f"[INFO] 已启用 torch.compile ({COMPILE_MODE})")
        except Exception as e:
            logging.warning(f"[WARN] torch.compile 不可用/失败: {e}")

    logging.info("[INFO] 准备数据集...")
    dataset = ParquetSentenceDataset(PARQUET_PATH, max_samples=None)

    export_path = os.path.join(OUTPUT_DIR, "train_sentences.txt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(export_path, "w", encoding="utf-8") as f:
        cnt = 0
        for s in dataset.sentences:
            s = str(s).replace("\r", " ").replace("\n", " ").strip()
            if not s:
                continue
            f.write(s + "\n")
            cnt += 1
    logging.info(f"[INFO] 已导出 {cnt} 条训练句子到: {export_path}")

    collate_fn = Collator(tokenizer, MAX_INPUT_TOKENS)

    try:
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
            pin_memory_device="cuda",
            collate_fn=collate_fn,
            drop_last=True,
            prefetch_factor=4,
            persistent_workers=True
        )
        logging.info(f"[INFO] DataLoader: num_workers=12, prefetch_factor=4, persistent_workers=True, pin_memory_device=cuda")
    except TypeError:
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True,
            prefetch_factor=4,
            persistent_workers=True
        )
        logging.info(f"[INFO] DataLoader: num_workers=12, prefetch_factor=4, persistent_workers=True")

    logging.info("[INFO] 设置优化器...")
    base_model = model.get_base_model()

    hidden_size = base_model.get_input_embeddings().weight.shape[1]
    z_vocab_size = (Z_ID_END - Z_ID_START + 1) + 1   # Z 区间 + EOZ

    pooler = SlotPooler(hidden_size, num_slots=MAX_Z_STEPS, z_vocab_size=z_vocab_size).to(device)

    # 优化器参数组里把 pooler 参数也加上（学习率可与 LoRA 组一致）
    lora_params = [p for p in model.parameters() if p.requires_grad and id(p) not in {id(embed_weight), id(lm_head_weight)}]

    embed_weight = base_model.get_input_embeddings().weight
    lm_head_weight = base_model.lm_head.weight

    vocab_size = base_model.lm_head.out_features

    special_params = {id(embed_weight), id(lm_head_weight)}
    lora_params = [p for p in model.parameters() if p.requires_grad and id(p) not in special_params]

    optimizer_groups = [
        {"params": [embed_weight, lm_head_weight], "lr": LR, "weight_decay": 0.0, "betas": (0.9, 0.95)},
        {"params": lora_params, "lr": LR, "weight_decay": 0.01, "betas": (0.9, 0.95)},
        {"params": pooler.parameters(),            "lr": LR, "weight_decay": 0.00, "betas": (0.9, 0.95)},
    ]

    try:
        optimizer = torch.optim.AdamW(optimizer_groups, fused=True)
        logging.info("[INFO] 使用 AdamW(fused=True)")
    except Exception:
        try:
            optimizer = torch.optim.AdamW(optimizer_groups, foreach=True)
            logging.info("[INFO] 使用 AdamW(foreach=True)")
        except Exception:
            optimizer = torch.optim.AdamW(optimizer_groups)
            logging.info("[INFO] 使用 AdamW(标准实现)")

    total_steps = math.ceil(len(dataloader) / GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    cuda_trainable = [p for g in optimizer.param_groups for p in g["params"]
                      if p.requires_grad and p.is_cuda and p.dtype.is_floating_point]
    use_amp = (len(cuda_trainable) > 0) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=False)

    bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id or tokenizer.bos_token_id

    logging.info(f"[INFO] BOS: {bos_id}, EOS: {eos_id}, Vocab Size: {vocab_size}")
    logging.info(f"[INFO] Z tokens: {Z_ID_START}-{Z_ID_END}, EOZ: {EOZ_ID}")

    mask_cache = MaskCache(vocab_size, Z_ID_START, Z_ID_END, EOZ_ID, device)
    mask_cache.z_embed_matrix = base_model.get_input_embeddings().weight[Z_ID_START:Z_ID_END+1]

    base_embed = base_model.get_input_embeddings()

    global_step = 0
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.info(f"[INFO] 开始训练: {EPOCHS} epochs, {len(dataloader)} batches/epoch")
    logging.info(f"[INFO] 总步数: {total_steps}, Warmup: {warmup_steps}")
    logging.info(f"[INFO] 使用策略: Scheduled Sampling + Z-only Reconstruction + ZBias + Attention Assist")

    model.train()

    bos_tokens_full = torch.full((BATCH_SIZE, 1), bos_id, device=device, dtype=torch.long)
    eos_tokens_full = torch.full((BATCH_SIZE, 1), eos_id, device=device, dtype=torch.long)
    comp_tokens_full = torch.full((BATCH_SIZE, 1), COMP_ID, device=device, dtype=torch.long)
    decomp_tokens_full = torch.full((BATCH_SIZE, 1), DECOMP_ID, device=device, dtype=torch.long)
    ones_col_cache = None

    total_batches = (len(dataloader) * EPOCHS) / GRAD_ACCUM
    progress_bar = tqdm(total=total_batches, desc="训练进度", unit="batch")
    start_time = time.time()

    z_usage_tracker = torch.zeros(mask_cache.num_z_tokens, device=device)

    for epoch in range(EPOCHS):
        epoch_losses = []
        epoch_div_losses = []
        optimizer.zero_grad(set_to_none=True)

        micro_step = 0
        prefetcher = CUDAPrefetcher(dataloader, device=device)
        batch = prefetcher.next()

        while batch is not None:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            B, L = input_ids.shape

            if ones_col_cache is None or ones_col_cache.shape[0] != B:
                ones_col_cache = torch.ones((B, 1), device=device, dtype=torch.long)

            tau = get_adaptive_tau(global_step, total_steps, TAU_INIT, TAU_MIN)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                z_embed_matrix = base_embed.weight.index_select(0, mask_cache.z_indices_with_eoz)

                # ===== 压缩器（一次前向并行） =====
                model.set_adapter("compress")

                bos_tokens = bos_tokens_full[:B]
                comp_tokens = comp_tokens_full[:B]

                compress_input = torch.cat([comp_tokens, bos_tokens, input_ids], dim=1)       # [B, 2+L]
                compress_mask  = torch.cat([ones_col_cache, ones_col_cache, attention_mask], dim=1)

                # 取最后层隐藏态（一次前向，无需 use_cache）
                out_enc = model(
                    input_ids=compress_input,
                    attention_mask=compress_mask,
                    use_cache=False,
                    output_hidden_states=True
                )

                # ctx_states 只取真实上下文部分（去掉前缀 <COMP>, <BOS>）
                ctx_states = out_enc.hidden_states[-1][:, 2:, :]        # [B, L, H]
                ctx_mask   = attention_mask                              # [B, L]

                # 一次前向得到 K 个槽位的 logits 与槽位向量
                z_logits_rel, slot_hidden = pooler(ctx_states, ctx_mask, tau=tau)  # [B, K, |Z|], [B, K, H]

                # 把“相对 Z 词表”的 logits 映射回“全词表”的位置上，以沿用你已有的 mask/查表逻辑
                # 先做一个全词表的巨型 -inf，再把 Z∪EOZ 段写回 logits
                very_neg = -1e4
                z_logits_full = torch.full((B, MAX_Z_STEPS, vocab_size), very_neg, device=device, dtype=slot_hidden.dtype)
                z_slice_idx = torch.cat([mask_cache.z_indices, torch.tensor([EOZ_ID], device=device, dtype=torch.long)])  # [|Z|]
                z_logits_full[:, :, z_slice_idx] = z_logits_rel

                # 经过 soft/hard Gumbel（hard + straight-through），得到 one-hot
                # 注意：这里对“相对 Z 词表”做 softmax 更稳定
                z_probs_rel = gumbel_softmax_sample(z_logits_rel, tau=tau, hard=True)   # [B, K, |Z|]

                # 取 codebook embedding（只对 Z∪EOZ 区间）
                z_embed_matrix = base_embed.weight.index_select(0, z_slice_idx)         # [|Z|, H]
                z_embeddings_soft = torch.matmul(z_probs_rel.to(z_embed_matrix.dtype), z_embed_matrix)  # [B, K, H]

                # 也构建“硬”的 e_hard（用 argmax id + 查 E）
                hard_idx_rel = torch.argmax(z_logits_rel, dim=-1)                       # [B, K]  相对索引
                hard_ids_abs = z_slice_idx[hard_idx_rel]                                 # [B, K]  绝对词表 id
                E = base_embed.weight
                e_hard = E[hard_ids_abs]                                                 # [B, K, H]

                # 直通技巧：硬 + 软的梯度
                z_embeddings = e_hard + (z_embeddings_soft - z_embeddings_soft.detach())

                # 统计/正则（和你原来一致）
                # 均匀使用正则：用相对 logits 的 softmax 分布统计（更干净）
                with torch.no_grad():
                    z_hist = torch.softmax(z_logits_rel.float(), dim=-1).sum(dim=(0, 1))   # [|Z|]
                    z_hist = z_hist / (z_hist.sum() + EPS)
                loss_unif = usage_kl_to_uniform(z_hist)

                # commitment
                loss_commit = commitment_loss(z_embeddings_soft, e_hard, beta=BETA_COMMIT)

                # ========== 解压缩器 ==========
                model.set_adapter("decompress")

                lengths = attention_mask.sum(dim=1)
                max_len = int(lengths.max().item())

                targets = torch.full((B, max_len + 1), eos_id, device=device, dtype=torch.long)
                for b in range(B):
                    Lb = int(lengths[b].item())
                    targets[b, :Lb] = input_ids[b, :Lb]
                    targets[b, Lb] = eos_id

                bos_embed = base_embed(bos_tokens_full[:B])
                decomp_embed = base_embed(decomp_tokens_full[:B])

                z_input = torch.cat([decomp_embed, bos_embed, z_embeddings], dim=1)
                z_mask = torch.ones(z_input.shape[:2], device=device, dtype=torch.long)

                # [CHANGED] 3) 截断 on-policy CE（重建支路的辅助损失）
                loss_z_recon = compute_z_only_recon_truncated(
                    model, z_embeddings, input_ids, attention_mask,
                    base_embed, mask_cache,
                    bos_id, DECOMP_ID, eos_id, device,
                    k_steps=8, rollin_min=0.2, rollin_max=0.8, forbid_z_out=True,
                )

                # Scheduled Sampling（主重建分支）
                sampling_prob = get_scheduled_sampling_prob(global_step, total_steps)
                full_embeds, full_mask = scheduled_sampling_decode(
                    model, z_input, z_mask, targets,
                    base_embed, sampling_prob, vocab_size,
                    mask_cache, device
                )

                out = model(
                    inputs_embeds=full_embeds,
                    attention_mask=full_mask,
                    use_cache=False,
                    output_attentions=False
                )
                logits_full = out.logits

                tf_inputs = targets[:, :-1]
                tf_labels = targets[:, 1:]
                T = tf_inputs.size(1)

                label_mask = torch.zeros_like(tf_labels, dtype=torch.bool)
                for b in range(B):
                    Lb = int(lengths[b].item())
                    valid_len = min(Lb, T)
                    if valid_len > 0:
                        label_mask[b, :valid_len] = True

                tf_labels_masked = torch.where(
                    label_mask, tf_labels, torch.full_like(tf_labels, -100)
                )

                z_len = z_input.size(1)
                logits_tf = logits_full[:, z_len:, :]


                # 屏蔽 Z∪EOZ 输出
                logits_tf = logits_tf.masked_fill(mask_cache.non_z_mask_bool.view(1, 1, -1), -1e4)

                loss_rec = F.cross_entropy(
                    logits_tf.reshape(-1, vocab_size),
                    tf_labels_masked.reshape(-1),
                    ignore_index=-100
                )

                # 总损失
                loss = (LAMBDA_REC * loss_rec
                        + LAMBDA_Z_RECON * loss_z_recon
                        # + LAMBDA_DIVERSITY * loss_div
                        + LAMBDA_COMMIT * loss_commit
                        + LAMBDA_UNIF * loss_unif
                        # + LAMBDA_MARGIN * loss_margin
                        )
                loss = loss.float()

                if not torch.isfinite(loss):
                    logging.warning(f"[WARNING] 计算得到非有限损失值: {loss.item()}")
                    optimizer.zero_grad(set_to_none=True)
                    batch = prefetcher.next()
                    continue

            # ===== 反向 =====
            loss_scaled = loss / GRAD_ACCUM
            if not torch.isfinite(loss_scaled):
                logging.warning(f"[WARNING] 缩放后损失非有限值，跳过")
                optimizer.zero_grad(set_to_none=True)
                batch = prefetcher.next()
                continue

            loss_scaled.backward()

            micro_step += 1
            is_last_in_epoch = (prefetcher.next_batch is None)
            do_step = ((micro_step % GRAD_ACCUM) == 0) or is_last_in_epoch
            if not do_step:
                batch = prefetcher.next()
                continue

            trainable_with_grad = [
                p for g in optimizer.param_groups for p in g["params"]
                if p.requires_grad and (p.grad is not None)
            ]
            if len(trainable_with_grad) == 0:
                optimizer.zero_grad(set_to_none=True)
                logging.info("len(trainable_with_grad)=0")
                batch = prefetcher.next()
                continue

            for p in trainable_with_grad:
                if p.grad is not None:
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=1e4, neginf=-1e4)
                    p.grad.clamp_(-5.0, 5.0)

            has_half = scaler.is_enabled() and any(
                (p.grad is not None) and p.grad.is_cuda and p.grad.dtype in (torch.float16, torch.bfloat16)
                for p in trainable_with_grad
            )

            did_optimizer_step = False
            if has_half:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_with_grad, max_norm=0.5)
                if torch.isfinite(grad_norm):
                    scaler.step(optimizer)
                    did_optimizer_step = True
                else:
                    logging.warning(f"[WARNING] 梯度范数非有限值，跳过")
            else:
                all_trainable_params = [p for p in model.parameters() if p.requires_grad]
                grad_norm = torch.nn.utils.clip_grad_norm_(all_trainable_params, max_norm=0.5)
                if torch.isfinite(grad_norm):
                    optimizer.step()
                    did_optimizer_step = True
                else:
                    logging.warning(f"[WARNING] 梯度范数非有限值，跳过")

            if has_half and did_optimizer_step:
                scaler.update()

            optimizer.zero_grad(set_to_none=True)

            if did_optimizer_step:
                scheduler.step()
                global_step += 1
                epoch_losses.append(float(loss.detach()))
                # epoch_div_losses.append(float(loss_div.detach()))

                if global_step % LOG_STEPS == 0:
                    l_val = float(loss.detach().cpu())
                    rec_val = float(loss_rec.detach().cpu())
                    z_recon_val = float(loss_z_recon.detach().cpu())
                    # div_val = float(loss_div.detach().cpu())
                    lr_val = scheduler.get_last_lr()[0]
                    commit_val = float(loss_commit.detach().cpu())
                    unif_val = float(loss_unif.detach().cpu())

                    recent_losses = epoch_losses[-min(LOG_STEPS, len(epoch_losses)):]
                    avg_loss = sum(recent_losses) / max(1, len(recent_losses))

                    recent_div_losses = epoch_div_losses[-min(LOG_STEPS, len(epoch_div_losses)):]
                    # avg_div_loss = sum(recent_div_losses) / max(1, len(recent_div_losses))

                    sampling_prob_current = get_scheduled_sampling_prob(global_step, total_steps)
                    if global_step % SAVE_STEPS == 0:
                        save_checkpoint(model, tokenizer, global_step, OUTPUT_DIR, pooler)
                    logging.info(f"[Epoch {epoch+1}/{EPOCHS}] Step {global_step} | "
                                 f"Loss: {l_val:.4f} (avg: {avg_loss:.4f}) | "
                                 f"Rec: {rec_val:.4f} | "
                                 f"Z-Recon(OnPolicy@K): {z_recon_val:.4f} | "
                                #  f"Div: {div_val:.4f} (avg: {avg_div_loss:.4f}) | "
                                 f"commit: {commit_val:.4f} | "
                                 f"unif: {unif_val:.4f} | "
                                 f"SampProb: {sampling_prob_current:.2%} | "
                                 f"Tau: {tau:.4f} | "
                                 f"LR: {lr_val:.2e}")

                    z_usage_tracker.zero_()

            progress_bar.update(1)
            elapsed_time = time.time() - start_time
            batches_processed = progress_bar.n
            if batches_processed > 0:
                time_per_batch = elapsed_time / batches_processed
                remaining_batches = total_batches - batches_processed
                remaining_time = timedelta(seconds=int(remaining_batches * time_per_batch))
                progress_bar.set_postfix({
                    "epoch": f"{epoch+1}/{EPOCHS}",
                    "剩余时间": str(remaining_time),
                    "当前损失": f"{loss.item():.4f}",
                    "采样率": f"{get_scheduled_sampling_prob(global_step, total_steps):.2%}"
                })

            batch = prefetcher.next()

    progress_bar.close()
    total_time = timedelta(seconds=int(time.time() - start_time))
    logging.info(f"✅ 训练完成！总训练时间: {total_time}")

def save_checkpoint(model, tokenizer, step, output_dir, pooler=None):
    """保存模型checkpoint"""
    save_name = f"checkpoint-{step}" if isinstance(step, int) else step
    save_path_compress = os.path.join(output_dir, f"{save_name}_compress")
    save_path_decompress = os.path.join(output_dir, f"{save_name}_decompress")
    os.makedirs(save_path_compress, exist_ok=True)
    os.makedirs(save_path_decompress, exist_ok=True)
    
    # 保存 LoRA adapters
    model.save_pretrained(save_path_compress, selected_adapters=["compress"])
    model.save_pretrained(save_path_decompress, selected_adapters=["decompress"])
    tokenizer.save_pretrained(save_path_compress)
    tokenizer.save_pretrained(save_path_decompress)
    
    # 保存基础模型的扩展权重
    base_model = model.get_base_model()
    E  = base_model.get_input_embeddings().weight.detach().cpu()
    LM = base_model.lm_head.weight.detach().cpu()

    new_embeds  = E[ORIG_VOCAB_SIZE:]
    new_lm_head = LM[:]
    
    # 保存数据
    save_data = {
        "new_embeddings": new_embeds,
        "new_lm_head": new_lm_head,
        "step": step,
        "orig_vocab_size": ORIG_VOCAB_SIZE,
        "tokenizer_size": len(tokenizer),
    }
    
    # 保存 pooler 参数（如果提供了）
    if pooler is not None:
        save_data["pooler_state_dict"] = pooler.state_dict()
        save_data["pooler_config"] = {
            "hidden_size": pooler.attn.embed_dim,
            "num_slots": pooler.num_slots,
            "z_vocab_size": pooler.proj.out_features
        }
    
    torch.save(save_data, os.path.join(output_dir, f"{save_name}_new_weights.pt"))
    
    # 也单独保存一份 pooler 参数以便单独加载
    if pooler is not None:
        torch.save({
            "pooler_state_dict": pooler.state_dict(),
            "pooler_config": save_data["pooler_config"],
            "step": step
        }, os.path.join(output_dir, f"{save_name}_pooler.pt"))
    
    logging.info(f"[SAVE] rows_saved={new_embeds.shape[0]} | "
                 f"orig={ORIG_VOCAB_SIZE} -> new={len(tokenizer)}")
    if pooler is not None:
        logging.info(f"[SAVE] Pooler参数已保存 | slots={pooler.num_slots}")
    logging.info(f"[SAVE] 已保存 checkpoint 到 {output_dir}/{save_name}")


if __name__ == "__main__":
    train()