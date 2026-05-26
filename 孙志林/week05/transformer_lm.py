"""
作业：训练基于Transformer的单向语言模型并完成文本生成

1. MiniGPT模型（Decoder-only架构）
2. 数据准备（文本→token序列→切块）
3. 训练流程（AdamW+余弦学习率）
4. 文本生成（4种解码策略）

使用方式：
    python transformer_lm.py --mode all              # 训练+生成
    python transformer_lm.py --mode train             # 仅训练
    python transformer_lm.py --mode generate          # 仅生成
    python transformer_lm.py --mode prepare           # 仅准备数据
"""

import os
import sys
import math
import json
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import BertTokenizerFast

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
LOG_PATH = OUTPUT_DIR / "training_log.jsonl"
PRETRAIN_MODEL_DIR = Path(__file__).parent / "pretrain_models" / "bert-base-chinese"

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


class CausalSelfAttention(nn.Module):
    """因果多头自注意力

    核心机制：
    1. Q/K/V投影：将输入分别映射为Query、Key、Value
    2. 缩放点积注意力：score = (Q @ K^T) / sqrt(d_head)
    3. 因果掩码：上三角置-inf，防止看到未来信息
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / self.scale

        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))


class FeedForward(nn.Module):
    """前馈神经网络：Linear → GELU → Linear"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer解码器块：Pre-LN结构

    Pre-LN：LayerNorm在残差分支内部，训练更稳定
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """Decoder-only GPT语言模型

    架构：
    1. Token Embedding + Positional Embedding（可学习）
    2. N个Transformer解码器块
    3. LM Head：映射到词表概率分布

    默认配置（~25M参数）：
        vocab_size=21128, seq_len=256, d_model=384, n_heads=6, n_layers=6, d_ff=1536
    """

    def __init__(
        self,
        vocab_size: int = 21128,
        seq_len: int = 256,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 6,
        d_ff: int = 1536,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        assert T <= self.seq_len, f"输入长度 {T} 超过最大序列长度 {self.seq_len}"

        positions = torch.arange(T, device=input_ids.device)
        x = self.emb_dropout(self.token_emb(input_ids) + self.pos_emb(positions))

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        return self.lm_head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(vocab_size: int = 21128, seq_len: int = 256) -> MiniGPT:
    return MiniGPT(vocab_size=vocab_size, seq_len=seq_len)


class TokenDataset(Dataset):
    """Token序列数据集

    自回归训练：
    - input: tokens[0..T-1]
    - target: tokens[1..T]（右移一位）
    """

    def __init__(self, pt_path: Path):
        ckpt = torch.load(pt_path, weights_only=True)
        self.data = ckpt["data"]
        self.vocab_size = ckpt["vocab_size"]
        self.seq_len = ckpt["seq_len"]
        logger.info(f"加载数据集：{pt_path.name}，共 {len(self.data):,} 个样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        return chunk[:-1], chunk[1:]


def get_tokenizer():
    local_path = PRETRAIN_MODEL_DIR
    if local_path.exists():
        return BertTokenizerFast.from_pretrained(str(local_path))
    return BertTokenizerFast.from_pretrained("bert-base-chinese")


def prepare_dataset(seq_len: int = 256, max_tokens: int = None):
    """数据准备：文本 → token id → 连续序列 → 切块"""
    jsonl_path = DATA_DIR / "wiki_zh.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"未找到数据文件 {jsonl_path}，请先运行demo/download_data.py或准备数据")

    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size
    logger.info(f"Tokenizer vocab size: {vocab_size}")

    logger.info("开始 tokenize 并拼接所有文章...")
    all_ids = []
    n_articles = 0
    SEP_TOKEN_ID = 102

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_ids.extend(ids)
            all_ids.append(SEP_TOKEN_ID)
            n_articles += 1

            if max_tokens and len(all_ids) >= max_tokens:
                break

            if n_articles % 5000 == 0:
                logger.info(f"  已处理 {n_articles} 篇，当前 token 数：{len(all_ids):,}")

    total_tokens = len(all_ids)
    logger.info(f"总 token 数：{total_tokens:,}（来自 {n_articles} 篇文章）")

    n_chunks = (total_tokens - 1) // seq_len
    logger.info(f"切块数：{n_chunks}（seq_len={seq_len}）")

    ids_tensor = torch.tensor(all_ids[:n_chunks * seq_len + 1], dtype=torch.long)

    val_size = max(1, int(n_chunks * 0.05))
    train_size = n_chunks - val_size
    logger.info(f"训练样本：{train_size:,}，验证样本：{val_size:,}")

    train_data = ids_tensor[: train_size * seq_len + 1].unfold(0, seq_len + 1, seq_len)[:-1]
    val_data = ids_tensor[train_size * seq_len: (train_size + val_size) * seq_len + 1].unfold(0, seq_len + 1, seq_len)[:-1]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"data": train_data, "vocab_size": vocab_size, "seq_len": seq_len},
               DATA_DIR / f"train_seq{seq_len}.pt")
    torch.save({"data": val_data, "vocab_size": vocab_size, "seq_len": seq_len},
               DATA_DIR / f"val_seq{seq_len}.pt")
    logger.info(f"数据已保存到 {DATA_DIR}")


def compute_ppl(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """计算验证集困惑度 PPL = exp(avg_loss)

    PPL 越低，模型语言建模能力越强
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for input_ids, targets in loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            logits = model(input_ids)
            B, T, V = logits.shape
            loss = loss_fn(logits.view(B * T, V), targets.view(B * T))
            total_loss += loss.item()
            total_tokens += B * T

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    model.train()
    return ppl


def train(
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    seq_len: int = 256,
    num_workers: int = 0,
):
    """训练循环"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备：{device}")

    train_path = DATA_DIR / f"train_seq{seq_len}.pt"
    val_path = DATA_DIR / f"val_seq{seq_len}.pt"
    if not train_path.exists():
        raise FileNotFoundError(f"未找到 {train_path}，请先运行 --mode prepare 准备数据")

    train_ds = TokenDataset(train_path)
    val_ds = TokenDataset(val_path)
    vocab_size = train_ds.vocab_size

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=(device.type == "cuda"))

    model = build_model(vocab_size=vocab_size, seq_len=seq_len).to(device)
    logger.info(f"模型参数量：{model.count_parameters() / 1e6:.1f}M")

    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = AdamW([
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=lr, betas=(0.9, 0.95))

    total_steps = len(train_loader) * epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.1)
    loss_fn = nn.CrossEntropyLoss()

    logger.info("计算训练前基线 PPL（随机初始化）...")
    baseline_ppl = compute_ppl(model, val_loader, device)
    logger.info(f"基线 val PPL：{baseline_ppl:.1f}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_val_ppl = float("inf")
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for step, (input_ids, targets) in enumerate(train_loader, 1):
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            logits = model(input_ids)
            B, T, V = logits.shape
            loss = loss_fn(logits.view(B * T, V), targets.view(B * T))

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if step % 200 == 0:
                avg_loss = epoch_loss / n_batches
                ppl = math.exp(avg_loss)
                cur_lr = scheduler.get_last_lr()[0]
                logger.info(f"Epoch {epoch}/{epochs} Step {step}/{len(train_loader)} | "
                           f"loss={avg_loss:.4f} PPL={ppl:.1f} lr={cur_lr:.2e}")

        val_ppl = compute_ppl(model, val_loader, device)
        train_ppl = math.exp(epoch_loss / n_batches)
        cur_lr = scheduler.get_last_lr()[0]

        logger.info(f"\n{'='*60}\nEpoch {epoch} 完成 | train PPL={train_ppl:.1f} | val PPL={val_ppl:.1f}\n{'='*60}")

        log_entry = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": epoch_loss / n_batches,
            "train_ppl": train_ppl,
            "val_ppl": val_ppl,
            "lr": cur_lr,
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        ckpt_path = CKPT_DIR / f"epoch{epoch}_ppl{val_ppl:.1f}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_ppl": val_ppl,
            "vocab_size": vocab_size,
            "seq_len": seq_len,
        }, ckpt_path)
        logger.info(f"Checkpoint 已保存：{ckpt_path.name}")

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_ppl": val_ppl,
                "vocab_size": vocab_size,
                "seq_len": seq_len,
            }, CKPT_DIR / "best_model.pt")
            logger.info(f"最优模型已更新 → val PPL={best_val_ppl:.1f}")

    logger.info(f"\n训练完成！最优 val PPL = {best_val_ppl:.1f}")


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 80,
    strategy: str = "greedy",
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
) -> torch.Tensor:
    """自回归文本生成

    四种解码策略：
    1. greedy: 贪心取最高概率token
    2. temperature: 温度采样，T<1更保守
    3. top_k: Top-K采样
    4. top_p: Top-P(Nucleus)采样
    """
    device = input_ids.device
    seq_len = model.seq_len
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        context = generated[:, -seq_len:]
        logits = model(context)
        next_logits = logits[0, -1, :]

        if strategy == "greedy":
            next_token = next_logits.argmax(dim=-1, keepdim=True)

        else:
            next_logits = next_logits / max(temperature, 1e-8)

            if strategy == "top_k":
                values, _ = torch.topk(next_logits, top_k)
                threshold = values[-1]
                next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

            elif strategy == "top_p":
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumprobs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

    return generated[0]


def decode_text(tokenizer, ids: torch.Tensor) -> str:
    return tokenizer.decode(ids.tolist(), skip_special_tokens=True)


def compare_strategies(model, tokenizer, prompt: str, max_new_tokens: int, device: torch.device):
    """四种解码策略并排对比"""
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]

    strategies = [
        ("Greedy", dict(strategy="greedy")),
        ("Temperature(T=0.8)", dict(strategy="temperature", temperature=0.8)),
        ("Top-K(K=50)", dict(strategy="top_k", temperature=0.8, top_k=50)),
        ("Top-P(p=0.9)", dict(strategy="top_p", temperature=0.8, top_p=0.9)),
    ]

    print(f"\n{'='*70}")
    print(f"Prompt：{prompt}")
    print(f"{'='*70}")

    for name, kwargs in strategies:
        out_ids = generate(model, input_ids, max_new_tokens=max_new_tokens, **kwargs)
        new_ids = out_ids[prompt_len:]
        generated_text = decode_text(tokenizer, new_ids)
        print(f"\n【{name}】")
        print(f"{prompt}{generated_text}")
        print("-" * 50)


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = build_model(vocab_size=ckpt["vocab_size"], seq_len=ckpt["seq_len"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["seq_len"]


def main():
    parser = argparse.ArgumentParser(description="Transformer单向语言模型训练与生成")
    parser.add_argument("--mode", type=str, default="all",
                       choices=["all", "train", "generate", "prepare"],
                       help="运行模式：all(训练+生成), train(仅训练), generate(仅生成), prepare(仅准备数据)")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--seq_len", type=int, default=256, help="序列长度")
    parser.add_argument("--max_tokens", type=int, default=None, help="最大token数（快速验证）")
    parser.add_argument("--checkpoint", type=str, default=None, help="指定checkpoint路径")
    parser.add_argument("--prompt", type=str, default="中国的首都是", help="生成起始文本")
    parser.add_argument("--max_new_tokens", type=int, default=80, help="最大生成token数")
    parser.add_argument("--strategy", type=str, default="top_p",
                       choices=["greedy", "temperature", "top_k", "top_p"])
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--compare", action="store_true", help="对比四种解码策略")
    args = parser.parse_args()

    if args.mode == "prepare":
        prepare_dataset(seq_len=args.seq_len, max_tokens=args.max_tokens)
        return

    if args.mode in ["all", "train"]:
        train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seq_len=args.seq_len)

    if args.mode in ["all", "generate"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = get_tokenizer()

        if args.checkpoint:
            ckpt_path = Path(args.checkpoint)
        else:
            ckpt_path = CKPT_DIR / "best_model.pt"

        if not ckpt_path.exists():
            logger.warning(f"未找到checkpoint：{ckpt_path}")
            logger.info("使用随机初始化模型进行演示...")
            model = build_model(vocab_size=21128, seq_len=256).to(device)
        else:
            model, seq_len = load_model(ckpt_path, device)
            logger.info(f"模型加载完毕：{ckpt_path}")

        if args.compare:
            compare_strategies(model, tokenizer, args.prompt, args.max_new_tokens, device)
        else:
            input_ids = tokenizer.encode(args.prompt, add_special_tokens=False, return_tensors="pt").to(device)
            out_ids = generate(
                model, input_ids,
                max_new_tokens=args.max_new_tokens,
                strategy=args.strategy,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
            output_text = decode_text(tokenizer, out_ids[input_ids.shape[1]:])
            print(f"\n{'='*60}")
            print(f"策略：{args.strategy}")
            print(f"{'='*60}")
            print(f"{args.prompt}{output_text}")


if __name__ == "__main__":
    main()