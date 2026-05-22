"""
GPT 因果语言模型 — 训练脚本

训练任务：next-token prediction（用前 N-1 个 token 预测后 N-1 个 token）

用法：
  python train.py                          # 从头训练
  python train.py --resume checkpoints/epoch_10.pt   # 从 checkpoint 恢复

运行前请确保已执行：
  python preprocess.py                     # 生成 token 缓存
"""
import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime

from config import *  # 必须在 transformers 之前，确保 HF_HOME 等环境变量先生效

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from tqdm import tqdm

from gpt_model import GptConfig, MyGpt


# ══════════════════════════════════════════════════════════════════
# 日志
# ══════════════════════════════════════════════════════════════════

def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"gpt_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = logging.getLogger("gpt_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    logger.info(f"日志文件: {log_file}")
    return logger


# ══════════════════════════════════════════════════════════════════
# 数据集（轻量 wrapper，核心逻辑在 preprocess.py）
# ══════════════════════════════════════════════════════════════════

class GptDataset(Dataset):
    """从缓存 .pt 文件加载预处理好的 token chunk"""
    def __init__(self, cache_path):
        self.chunks = torch.load(cache_path, weights_only=True, mmap=True, map_location="cpu")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return chunk, chunk.clone()  # input_ids, labels（模型内部做 shift）


# ══════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════

def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    elif m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


@torch.no_grad()
def generate_sample(model, tokenizer, prompt, device):
    """生成文本样本，用于观察训练进展"""
    model.eval()
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    if input_ids.size(1) == 0:
        return "(空 prompt)"
    output_ids = model.generate(
        input_ids,
        max_new_tokens=GEN_LENGTH,
        temperature=GEN_TEMPERATURE,
        top_k=GEN_TOP_K,
        top_p=GEN_TOP_P,
        min_p=GEN_MIN_P,
        repetition_penalty=GEN_REPETITION_PENALTY,
        eos_token_id=tokenizer.sep_token_id,
    )
    generated = tokenizer.decode(output_ids[0].cpu().tolist(), skip_special_tokens=True).replace(" ", "")
    model.train()
    return generated


def get_perplexity(loss):
    return torch.exp(torch.tensor(loss)).item()


# ══════════════════════════════════════════════════════════════════
# 验证
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(model, val_loader, device, use_tqdm=True):
    model.eval()
    total_loss = 0.0
    batches = 0
    loader = tqdm(val_loader, desc="验证", leave=False, ncols=100) if use_tqdm else val_loader
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        outputs = model(input_ids, labels=labels)
        total_loss += outputs["loss"].item()
        batches += 1
        if use_tqdm:
            loader.set_postfix(loss=f"{total_loss / batches:.3f}")
    avg_loss = total_loss / batches
    ppl = get_perplexity(avg_loss)
    model.train()
    return avg_loss, ppl


# ══════════════════════════════════════════════════════════════════
# 主训练函数
# ══════════════════════════════════════════════════════════════════

def train(config: GptConfig, resume_from: str = None, logger: logging.Logger = None,
          use_combined: bool = False):
    if logger is None:
        logger = logging.getLogger("gpt_train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ── Tokenizer ──
    logger.info(f"\n加载 tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # ── 数据 ──
    train_cache = COMBINED_CACHE if use_combined else TRAIN_CACHE
    logger.info("\n加载预处理数据...")
    if use_combined:
        logger.info(f"  使用联合语料: cnews + SogouC + WikiText-zh")
    for cache_path, desc in [(train_cache, "训练集缓存"), (VAL_CACHE, "验证集缓存")]:
        if not os.path.exists(cache_path):
            logger.error(f"{desc}不存在: {cache_path}")
            logger.error("请先运行: python preprocess.py" +
                         (" --combine --wiki" if use_combined else ""))
            sys.exit(1)

    train_ds = GptDataset(train_cache)
    val_ds = GptDataset(VAL_CACHE)
    logger.info(f"  训练样本: {len(train_ds):,}, 验证样本: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )
    logger.info(f"  训练 batch: {len(train_loader)}, 验证 batch: {len(val_loader)}")

    # ── 全局精度 ──
    torch.set_float32_matmul_precision(MATMUL_PRECISION)

    # ── 模型 ──
    logger.info("\n构建模型...")
    model = MyGpt(config).to(device)
    # 保存新模型的位置编码尺寸（compile 后 state_dict 可能不可访问）
    _new_pos_shape = model.embeddings.position_embeddings.weight.shape
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  总参数: {total_params:,}")
    logger.info(f"  可训练: {trainable:,}")

    # ── 优化器 & 调度器（compile 前创建，参数引用在 compile 后仍有效）──
    adam_kwargs = dict(lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    if USE_FUSED_ADAMW:
        adam_kwargs["fused"] = True
    optimizer = AdamW(model.parameters(), **adam_kwargs)
    steps_per_epoch = len(train_loader) // GRADIENT_ACCUMULATION
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = steps_per_epoch * WARMUP_EPOCHS
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)

    amp_dtype = torch.bfloat16 if AMP_DTYPE == "bf16" else torch.float16
    if USE_AMP and AMP_DTYPE == "bf16" and device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            logger.warning("  GPU 不支持 bf16，回退到 fp16")
            amp_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda") if (USE_AMP and amp_dtype == torch.float16) else None

    # ── 恢复训练（必须在 compile 前加载，否则 OptimizedModule 的 key 带 _orig_mod. 前缀会不匹配）──
    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_ppl": []}

    if resume_from:
        logger.info(f"\n从 checkpoint 恢复: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)

        # 兼容含 _orig_mod. 前缀的 checkpoint（由 compiled 模型保存时自动 strip）
        old_pos = ckpt["model_state_dict"]["embeddings.position_embeddings.weight"]
        old_len, new_len = old_pos.shape[0], _new_pos_shape[0]
        if old_len != new_len:
            logger.info(f"  Position Embedding 不匹配: {old_len} → {new_len}")
            if old_len < new_len:
                interpolated = torch.nn.functional.interpolate(
                    old_pos.unsqueeze(0).unsqueeze(0),
                    size=(new_len, old_pos.shape[1]),
                    mode="bilinear",
                ).squeeze(0).squeeze(0)
                interpolated[:old_len] = old_pos
                ckpt["model_state_dict"]["embeddings.position_embeddings.weight"] = interpolated

                opt_state = ckpt["optimizer_state_dict"]["state"]
                for param_id, state in opt_state.items():
                    exp_avg = state.get("exp_avg")
                    if exp_avg is not None and exp_avg.ndim == 2 \
                       and exp_avg.shape[0] == old_len:
                        for key in ["exp_avg", "exp_avg_sq"]:
                            t = state[key]
                            ti = torch.nn.functional.interpolate(
                                t.unsqueeze(0).unsqueeze(0),
                                size=(new_len, t.shape[1]),
                                mode="bilinear",
                            ).squeeze(0).squeeze(0)
                            ti[:old_len] = t
                            state[key] = ti
                        break
                logger.info(f"  → 权重 + Adam 状态均已完成插值")
            else:
                ckpt["model_state_dict"]["embeddings.position_embeddings.weight"] = old_pos[:new_len]
                logger.info(f"  → 截断到 {new_len} 位置")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scaler and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        history = ckpt.get("history", history)
        logger.info(f"  恢复到 Epoch {ckpt['epoch']}, 最佳 val loss: {best_val_loss:.4f}")

    # torch.compile（在 resume 之后，避免 OptimizedModule 前缀不匹配）
    if USE_COMPILE:
        if device.type == "cuda" and COMPILE_CACHE_LIMIT:
            torch._dynamo.config.cache_size_limit = COMPILE_CACHE_LIMIT
        logger.info(f"  启用 torch.compile (mode={COMPILE_MODE})...")
        model = torch.compile(model, mode=COMPILE_MODE)

    features = []
    if USE_AMP: features.append(f"AMP({AMP_DTYPE})" if amp_dtype == (torch.bfloat16 if AMP_DTYPE=="bf16" else torch.float16) else f"AMP({AMP_DTYPE}→fp16)")
    if USE_COMPILE: features.append(f"compile({COMPILE_MODE})")
    if USE_GRADIENT_CHECKPOINT: features.append("grad-ckpt")
    if USE_FUSED_ADAMW: features.append("fused-AdamW")
    logger.info(f"  总步数: {total_steps:,}, Warmup: {warmup_steps:,}")
    if features:
        logger.info(f"  加速特性: {', '.join(features)}")

    # ── 训练循环 ──
    logger.info("\n" + "=" * 60)
    logger.info(f"训练配置:")
    logger.info(f"  BATCH={BATCH_SIZE}x{GRADIENT_ACCUMULATION} "
                f"(有效={BATCH_SIZE * GRADIENT_ACCUMULATION})")
    logger.info(f"  LR={LEARNING_RATE} | EPOCHS={EPOCHS} | "
                f"SEQ_LEN={MAX_SEQ_LEN} | WARMUP={WARMUP_EPOCHS}轮")
    logger.info("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)
    global_step = 0

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        batch_count = 0
        optimizer.zero_grad()

        # ── tqdm 进度条 ──
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}",
                     ncols=120, leave=True, file=sys.stdout)
        for batch_idx, (input_ids, labels) in enumerate(pbar):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            with torch.amp.autocast("cuda", enabled=USE_AMP, dtype=amp_dtype):
                torch.compiler.cudagraph_mark_step_begin()
                outputs = model(input_ids.clone(), labels=labels)
            loss = outputs["loss"] / GRADIENT_ACCUMULATION
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            batch_count += 1
            # 用 detach 累加避免 CUDA sync；仅在累积步调用 .item()
            _loss_detached = loss.detach()
            train_loss += _loss_detached * GRADIENT_ACCUMULATION

            if batch_count % GRADIENT_ACCUMULATION == 0:
                global_step += 1

                # Warmup: 线性增加 LR
                if global_step <= warmup_steps and WARMUP_EPOCHS > 0:
                    lr_scale = global_step / max(warmup_steps, 1)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = LEARNING_RATE * lr_scale

                if scaler:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if global_step > warmup_steps:
                    scheduler.step()
                optimizer.zero_grad()

                # 仅在累积步更新进度条（减少 CUDA sync）
                displayed_loss = _loss_detached.float().item() * GRADIENT_ACCUMULATION
                avg_loss = train_loss.float().item() / batch_count
                current_lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(
                    loss=f"{displayed_loss:.3f}",
                    avg=f"{avg_loss:.3f}",
                    lr=f"{current_lr:.1e}",
                )

        avg_train_loss = train_loss.float().item() / batch_count
        history["train_loss"].append(avg_train_loss)

        # ── 验证 ──
        val_loss, ppl = validate(model, val_loader, device)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(ppl)

        epoch_time = time.time() - epoch_start

        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{EPOCHS} 完成 | 用时: {format_time(epoch_time)}")
        logger.info(f"  训练 Loss: {avg_train_loss:.4f}")
        logger.info(f"  验证 Loss: {val_loss:.4f}  |  PPL: {ppl:.2f}")
        logger.info(f"{'='*60}")

        # ── 生成样本 ──
        gen_text = generate_sample(model, tokenizer, GEN_PROMPT, device)
        logger.info(f"\n  生成样本 (prompt='{GEN_PROMPT}'):")
        logger.info(f"  {gen_text[:200]}\n")

        # ── 构建 checkpoint ──
        raw_model = getattr(model, '_orig_mod', model)  # compile 后取原始模型
        ckpt = {
            "epoch": epoch,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss, "val_ppl": ppl,
            "best_val_loss": best_val_loss,
            "config": {k: v for k, v in config.__dict__.items()},
            "history": history,
        }
        if scaler:
            ckpt["scaler_state_dict"] = scaler.state_dict()

        # ── 保存最佳模型 ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(ckpt, os.path.join(MODEL_DIR, "best_model.pt"))
            logger.info(f"  >> 最佳模型已保存 (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            logger.info(
                f"  Val loss 未下降 ({patience_counter}/{EARLY_STOP_PATIENCE}), "
                f"最佳: {best_val_loss:.4f}"
            )

        # ── 定期保存 checkpoint ──
        if epoch % SAVE_EPOCHS == 0:
            torch.save(ckpt, os.path.join(MODEL_DIR, f"epoch_{epoch}.pt"))
            logger.info(f"  Checkpoint 已保存: epoch_{epoch}.pt")

        # ── 早停 ──
        if patience_counter >= EARLY_STOP_PATIENCE:
            logger.info(f"\n早停！Val loss 连续 {EARLY_STOP_PATIENCE} 轮未下降。")
            logger.info(f"最佳 val_loss: {best_val_loss:.4f} @ epoch {epoch - patience_counter}")
            break

    logger.info(f"\n训练完成！最佳验证 Loss: {best_val_loss:.4f}")

    # 保存训练历史
    history_path = os.path.join(MODEL_DIR, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    logger.info(f"训练历史已保存: {history_path}")

    return model, history


# ══════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GPT 因果语言模型训练")
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 恢复训练")
    parser.add_argument("--combined", action="store_true",
                        help="使用联合语料训练 (cnews + SogouC + WikiText-zh)")
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("GPT 因果语言模型 — 训练")
    logger.info("=" * 60)

    config = GptConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        hidden_dropout_prob=HIDDEN_DROPOUT_PROB,
        attention_probs_dropout_prob=ATTENTION_PROBS_DROPOUT_PROB,
        layer_norm_eps=LAYER_NORM_EPS,
        pad_token_id=PAD_TOKEN_ID,
        use_rms_norm=USE_RMS_NORM,
        use_flash_attn=USE_FLASH_ATTN,
        use_gradient_checkpoint=USE_GRADIENT_CHECKPOINT,
    )

    model, history = train(config, resume_from=args.resume, logger=logger,
                           use_combined=args.combined)

    # ── 最终生成测试 ──
    logger.info("\n" + "=" * 60)
    logger.info("加载最佳模型，进行最终生成测试")
    logger.info("=" * 60)

    device = next(model.parameters()).device
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    best_ckpt = torch.load(os.path.join(MODEL_DIR, "best_model.pt"),
                           map_location=device, weights_only=False)
    raw_model = getattr(model, '_orig_mod', model)
    raw_model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    test_prompts = ["今天", "中国", "人工智能", "体育", "经济"]
    for prompt in test_prompts:
        text = generate_sample(model, tokenizer, prompt, device)
        logger.info(f"\n  [{prompt}] {text[:200]}")

    logger.info("\n全部完成。")


if __name__ == "__main__":
    main()
