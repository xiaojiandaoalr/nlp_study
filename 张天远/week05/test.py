"""
GPT 因果语言模型 — 测试脚本：评估 + 多样性评测 + 交互式生成

用法：
  python test.py                    # 完整评测 + 交互式生成
  python test.py --prompt "...      # 单次生成
  python test.py --eval-only        # 仅评测，跳过交互
"""
import os
import sys
import argparse
import logging
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from config import *
from gpt_model import GptConfig, MyGpt


# ══════════════════════════════════════════════════════════════════
# 日志
# ══════════════════════════════════════════════════════════════════

def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, "gpt_test_result.log")
    logger = logging.getLogger("gpt_test")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    return logger


# ══════════════════════════════════════════════════════════════════
# 模型加载
# ══════════════════════════════════════════════════════════════════

def load_model(device):
    ckpt_path = os.path.join(MODEL_DIR, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"未找到训练好的模型: {ckpt_path}\n请先运行: python train.py"
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    saved_cfg = ckpt.get("config", {})
    config = GptConfig(**saved_cfg)

    model = MyGpt(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, config, ckpt


# ══════════════════════════════════════════════════════════════════
# 评估
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()
    total_loss = 0.0
    batches = 0
    for input_ids, labels in test_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        outputs = model(input_ids, labels=labels)
        total_loss += outputs["loss"].item()
        batches += 1
    avg_loss = total_loss / batches
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, ppl


# ══════════════════════════════════════════════════════════════════
# 生成
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate(model, tokenizer, prompt, device, max_tokens=100,
             temperature=0.8, top_k=40, top_p=0.95,
             min_p=0.05, repetition_penalty=1.1):
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    if input_ids.size(1) == 0:
        return "(无法 tokenize 该 prompt)"

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.sep_token_id,
    )
    return tokenizer.decode(output_ids[0].cpu().tolist(), skip_special_tokens=True).replace(" ", "")


# ══════════════════════════════════════════════════════════════════
# 多样性评测
# ══════════════════════════════════════════════════════════════════

def compute_diversity(texts):
    """distinct-1, distinct-2, repetition rate"""
    all_unigrams = []
    all_bigrams = []
    repeat_count = 0
    total_tokens = 0

    for text in texts:
        tokens = list(text)  # 字符级分词，中文天然
        all_unigrams.extend(tokens)
        all_bigrams.extend(["".join(tokens[i:i+2]) for i in range(len(tokens)-1)])
        total_tokens += len(tokens)
        # 连续重复：相同 token 连续出现计一次
        for i in range(1, len(tokens)):
            if tokens[i] == tokens[i-1]:
                repeat_count += 1

    unigram_count = len(all_unigrams)
    bigram_count = len(all_bigrams)

    distinct1 = len(set(all_unigrams)) / unigram_count if unigram_count > 0 else 0
    distinct2 = len(set(all_bigrams)) / bigram_count if bigram_count > 0 else 0
    rep_rate = repeat_count / total_tokens if total_tokens > 0 else 0

    return {
        "distinct-1": round(distinct1, 4),
        "distinct-2": round(distinct2, 4),
        "rep_rate": round(rep_rate, 4),
        "total_tokens": total_tokens,
    }


# 多主题 prompt，覆盖不同领域
EVAL_PROMPTS = {
    "新闻": ["据新华社报道", "当地时间", "记者从", "在发布会上"],
    "科技": ["人工智能的发展", "随着5G技术的", "科学家发现", "最新研究表明"],
    "体育": ["在本赛季", "这场比赛中", "凭借出色的", "最终以"],
    "日常": ["今天天气", "周末和朋友", "回家的路上", "昨天我去了"],
    "文学": ["在那遥远的地方", "夜色渐深", "他站在窗前", "春风拂过"],
}


# ══════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GPT 模型测试与生成")
    parser.add_argument("--prompt", type=str, default=None,
                        help="单次生成模式")
    parser.add_argument("--eval-only", action="store_true",
                        help="仅评测，跳过交互式生成")
    parser.add_argument("--max_tokens", type=int, default=GEN_LENGTH,
                        help="生成的最大 token 数")
    parser.add_argument("--temperature", type=float, default=GEN_TEMPERATURE,
                        help="采样温度")
    args = parser.parse_args()

    logger = setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("GPT 因果语言模型 — 测试")
    logger.info("=" * 60)
    logger.info(f"设备: {device}")

    # ── 加载模型 ──
    logger.info("\n加载模型...")
    model, config, ckpt = load_model(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  模型参数: {total_params:,}")
    logger.info(f"  训练轮数: Epoch {ckpt['epoch']}")
    if isinstance(ckpt.get('val_loss'), (int, float)):
        logger.info(f"  验证 Loss: {ckpt['val_loss']:.4f}")
    if isinstance(ckpt.get('val_ppl'), (int, float)):
        logger.info(f"  验证 PPL: {ckpt['val_ppl']:.2f}")

    # ── Tokenizer ──
    logger.info(f"\n加载 tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # ══════════════════════════════════════════════════════════════
    # 1. 测试集 PPL 评测
    # ══════════════════════════════════════════════════════════════
    if os.path.exists(TEST_CACHE):
        logger.info("\n" + "=" * 60)
        logger.info("1. 测试集困惑度 (PPL)")
        logger.info("=" * 60)

        class GptDataset(Dataset):
            def __init__(self, cache_path):
                self.chunks = torch.load(cache_path, weights_only=True, mmap=True, map_location="cpu")
            def __len__(self):
                return len(self.chunks)
            def __getitem__(self, idx):
                c = self.chunks[idx]
                return c, c.clone()

        test_ds = GptDataset(TEST_CACHE)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        logger.info(f"  测试样本: {len(test_ds):,}")

        test_loss, test_ppl = evaluate(model, test_loader, device)
        logger.info(f"  测试 Loss: {test_loss:.4f}")
        logger.info(f"  测试 PPL: {test_ppl:.2f}  (随机 ≈ {config.vocab_size})")
    else:
        logger.info(f"\n跳过测试集评估（缓存不存在: {TEST_CACHE}）")

    # ══════════════════════════════════════════════════════════════
    # 2. 多样性评测
    # ══════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("2. 多样性评测 (Distinct-N & 重复率)")
    logger.info("=" * 60)
    logger.info("  5 类别 × 4 prompt × 1 采样, max_tokens=100")
    logger.info("")

    all_texts = []
    cat_results = {}

    for category, prompts in EVAL_PROMPTS.items():
        cat_texts = []
        for prompt in prompts:
            text = generate(model, tokenizer, prompt, device,
                          max_tokens=100, temperature=0.8)
            cat_texts.append(text)
            all_texts.append(text)
        div = compute_diversity(cat_texts)
        cat_results[category] = div
        logger.info(f"  [{category}] distinct-1={div['distinct-1']:.3f}  "
                     f"distinct-2={div['distinct-2']:.3f}  "
                     f"重复率={div['rep_rate']:.3f}  token数={div['total_tokens']}")

    overall = compute_diversity(all_texts)
    logger.info(f"  {'─' * 48}")
    logger.info(f"  [总体] distinct-1={overall['distinct-1']:.3f}  "
                 f"distinct-2={overall['distinct-2']:.3f}  "
                 f"重复率={overall['rep_rate']:.3f}  token数={overall['total_tokens']}")

    # ── 生成样本展示 ──
    logger.info(f"\n  ── 生成样本 ──")
    for category, prompts in EVAL_PROMPTS.items():
        prompt = prompts[0]
        text = generate(model, tokenizer, prompt, device,
                       max_tokens=80, temperature=0.8)
        logger.info(f"  [{category}] {prompt}{text}")

    # ══════════════════════════════════════════════════════════════
    # 3. 单次生成 / 交互式生成
    # ══════════════════════════════════════════════════════════════
    if args.prompt:
        logger.info(f"\n{'='*60}")
        logger.info(f"生成: {args.prompt}")
        logger.info(f"{'='*60}")
        text = generate(model, tokenizer, args.prompt, device,
                        max_tokens=args.max_tokens, temperature=args.temperature)
        print(f"\n{text}\n")
        return

    if args.eval_only:
        logger.info("\n评测完成（--eval-only）。")
        return

    print("\n" + "=" * 60)
    print("GPT 交互式生成模式")
    print(f"配置: {config.hidden_size}维 × {config.num_hidden_layers}层")
    print(f"温度={args.temperature}  Top-k=40  Top-p=0.95")
    print("输入 'quit' / 'exit' 退出, 'temp=0.5' 设置温度")
    print("=" * 60)

    temperature = args.temperature
    max_tokens = args.max_tokens

    while True:
        try:
            user_input = input("\n请输入 prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("退出。")
            break

        if user_input.startswith("temp="):
            try:
                temperature = float(user_input.split("=")[1])
                print(f"温度已设为: {temperature}")
            except ValueError:
                print(f"无效温度值")
            continue
        if user_input.startswith("max_tokens="):
            try:
                max_tokens = int(user_input.split("=")[1])
                print(f"最大 token 数已设为: {max_tokens}")
            except ValueError:
                print(f"无效值")
            continue

        text = generate(model, tokenizer, user_input, device,
                        max_tokens=max_tokens, temperature=temperature)
        print(f"\n{text}")

    logger.info("\n测试完成。")


if __name__ == "__main__":
    main()
