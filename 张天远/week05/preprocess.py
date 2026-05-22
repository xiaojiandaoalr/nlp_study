"""
GPT 数据预处理：读取文本 → tokenize → 拼接 → 切成定长 chunk → 保存 .pt

支持三种数据源：
  1. cnews 单文件格式（标签\t文本）
  2. SogouC 目录格式（多级目录下 GBK/UTF-8 txt 文件）
  3. WikiText-zh 纯文本格式（一行一个段落，无标签）

下载 WikiText-zh：
  huggingface-cli download pleisto/wikipedia-cn-20230720-filtered --local-dir wiki_zh
  # 然后将 wiki_zh 下的文本文件放到 CN_Corpus/wiki_zh.txt

用法：
  python preprocess.py              # 处理 cnews 训练集 + 验证集
  python preprocess.py --all        # 同时处理测试集
  python preprocess.py --combine    # 合并 cnews + SogouC，生成联合训练集
  python preprocess.py --combine --wiki  # 合并 cnews + SogouC + WikiText-zh
"""
import os
import sys
import argparse
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from config import *


class GptPreprocessor:
    """GPT 数据预处理器"""
    def __init__(self, tokenizer_name=TOKENIZER_NAME, max_seq_len=MAX_SEQ_LEN,
                 min_seq_len=MIN_SEQ_LEN):
        print(f"加载 tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len

    # ── 文本读取 ──

    def _read_file_lines(self, file_path, encoding="utf-8"):
        """逐行读取 cnews 格式文件（标签\t文本），返回文本列表"""
        texts = []
        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                text = parts[1] if len(parts) == 2 else line
                if len(text) >= self.min_seq_len:
                    texts.append(text)
        return texts

    def _read_plain_file(self, file_path, encoding="utf-8"):
        """逐行读取纯文本文件（一行一个段落），返回文本列表"""
        texts = []
        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            for line in f:
                line = line.strip()
                if line and len(line) >= self.min_seq_len:
                    texts.append(line)
        return texts

    def _read_directory(self, dir_path):
        """递归读取目录下所有 txt 文件（自动检测GBK/UTF-8编码），返回文本列表"""
        # 收集所有 txt 文件
        all_files = []
        for root, _dirs, files in os.walk(dir_path):
            for fname in files:
                if fname.endswith(".txt"):
                    all_files.append(os.path.join(root, fname))

        print(f"  发现 {len(all_files):,} 个 txt 文件")

        # 先探测编码：随机抽几个文件，用 chardet 或逐编码试
        encodings_to_try = ["gb18030", "gbk", "utf-8", "utf-8-sig"]
        detected_encoding = "utf-8"
        sample_files = all_files[:min(5, len(all_files))]
        for enc in encodings_to_try:
            ok = True
            for fp in sample_files:
                try:
                    with open(fp, "r", encoding=enc) as fh:
                        fh.read(100)
                except (UnicodeDecodeError, UnicodeError):
                    ok = False
                    break
            if ok:
                detected_encoding = enc
                break
        print(f"  检测到编码: {detected_encoding}")

        texts = []
        for fp in tqdm(all_files, desc="  读取文件", ncols=100):
            try:
                with open(fp, "r", encoding=detected_encoding, errors="replace") as fh:
                    content = fh.read()
            except Exception:
                continue
            # 去掉首尾空白，过滤太短的
            content = content.strip()
            if len(content) >= self.min_seq_len:
                texts.append(content)

        return texts

    # ── tokenize & chunk ──

    def _tokenize_and_chunk(self, texts, cache_path, desc=""):
        """将文本列表 tokenize、拼接、切 chunk 并保存"""
        all_ids = []
        sep_id = self.tokenizer.sep_token_id
        total_chars = 0

        for text in tqdm(texts, desc="  tokenize", ncols=100):
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) >= self.min_seq_len:
                all_ids.extend(ids)
                all_ids.append(sep_id)
                total_chars += len(text)

        all_ids = torch.tensor(all_ids, dtype=torch.long)
        total_tokens = len(all_ids)
        print(f"  总字符数: {total_chars:,}")
        print(f"  总 token 数: {total_tokens:,}")
        print(f"  平均每字符 token 数: {total_tokens / max(total_chars, 1):.2f}")

        num_chunks = total_tokens // self.max_seq_len
        usable = num_chunks * self.max_seq_len
        chunks = all_ids[:usable].view(num_chunks, self.max_seq_len)
        discarded = total_tokens - usable
        print(f"  生成 {num_chunks:,} 个训练样本 ({self.max_seq_len} tokens/样本)")
        print(f"  丢弃尾数: {discarded} tokens ({discarded / max(total_tokens, 1) * 100:.1f}%)")

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(chunks, cache_path)
        file_size = os.path.getsize(cache_path) / 1024**2
        print(f"  已保存: {cache_path} ({file_size:.1f} MB)")
        return chunks

    # ── 公开接口 ──

    def process_file(self, file_path, cache_path, desc=""):
        """处理单个 cnews 格式文件"""
        print(f"\n{'='*60}")
        print(f"处理 {desc}: {file_path}")
        print(f"{'='*60}")
        texts = self._read_file_lines(file_path)
        print(f"  有效文本: {len(texts):,} 条 (≥{self.min_seq_len} 字符)")
        return self._tokenize_and_chunk(texts, cache_path, desc)

    def process_directory(self, dir_path, cache_path, desc=""):
        """处理 SogouC 目录格式"""
        print(f"\n{'='*60}")
        print(f"处理 {desc}: {dir_path}")
        print(f"{'='*60}")
        texts = self._read_directory(dir_path)
        print(f"  有效文本: {len(texts):,} 条 (≥{self.min_seq_len} 字符)")
        return self._tokenize_and_chunk(texts, cache_path, desc)

    def process_combined(self, sources, cache_path, desc=""):
        """合并多个数据源（file=带标签, plain=纯文本, dir=目录）"""
        print(f"\n{'='*60}")
        print(f"处理 {desc}")
        print(f"{'='*60}")

        all_texts = []
        for src_path, src_type in sources:
            print(f"  读取: {src_path}")
            if src_type == "file":
                texts = self._read_file_lines(src_path)
            elif src_type == "plain":
                texts = self._read_plain_file(src_path)
            else:
                texts = self._read_directory(src_path)
            print(f"    → {len(texts):,} 条文本")
            all_texts.extend(texts)

        print(f"  合并后总计: {len(all_texts):,} 条文本 (≥{self.min_seq_len} 字符)")
        return self._tokenize_and_chunk(all_texts, cache_path, desc)


def main():
    parser = argparse.ArgumentParser(description="GPT 数据预处理")
    parser.add_argument("--all", action="store_true",
                        help="同时处理测试集（默认只处理训练集+验证集）")
    parser.add_argument("--combine", action="store_true",
                        help="合并 cnews 训练集 + SogouC 语料，生成联合训练集")
    parser.add_argument("--wiki", action="store_true",
                        help="同时合并 WikiText-zh 纯文本语料（需先下载）")
    args = parser.parse_args()

    proc = GptPreprocessor()

    if args.combine:
        # 合并模式：cnews train + SogouC (+ WikiText-zh) → combined_train_tokens.pt
        sources = []
        if os.path.exists(TRAIN_FILE):
            sources.append((TRAIN_FILE, "file"))
        else:
            print(f"警告: cnews 训练集不存在: {TRAIN_FILE}")

        if os.path.exists(CN_CORPUS_DIR):
            sources.append((CN_CORPUS_DIR, "dir"))
        else:
            print(f"警告: SogouC 语料目录不存在: {CN_CORPUS_DIR}")

        if args.wiki:
            if os.path.exists(WIKI_FILE):
                sources.append((WIKI_FILE, "plain"))
            else:
                print(f"警告: WikiText-zh 文件不存在: {WIKI_FILE}")
                print(f"  下载: huggingface-cli download pleisto/wikipedia-cn-20230720-filtered")
                print(f"  然后将文本文件移动/重命名为: {WIKI_FILE}")

        if not sources:
            print("错误: 没有任何可用的数据源")
            sys.exit(1)

        proc.process_combined(sources, COMBINED_CACHE, "合并训练集 (cnews + SogouC)")

        # 合并模式也处理验证集，方便直接训练
        if os.path.exists(VAL_FILE):
            proc.process_file(VAL_FILE, VAL_CACHE, "验证集")
    else:
        # 标准模式：cnews 训练集 + 验证集
        for fpath, desc in [(TRAIN_FILE, "训练集"), (VAL_FILE, "验证集")]:
            if not os.path.exists(fpath):
                print(f"错误: {desc} 文件不存在: {fpath}")
                sys.exit(1)
        proc.process_file(TRAIN_FILE, TRAIN_CACHE, "训练集")
        proc.process_file(VAL_FILE, VAL_CACHE, "验证集")

    # 测试集（可选）
    if args.all:
        if os.path.exists(TEST_FILE):
            proc.process_file(TEST_FILE, TEST_CACHE, "测试集")
        else:
            print(f"\n警告: 测试集文件不存在: {TEST_FILE}")

    print(f"\n{'='*60}")
    print("预处理完成！")
    if args.combine:
        print(f"  联合训练缓存: {COMBINED_CACHE}")
    else:
        print(f"  训练缓存: {TRAIN_CACHE}")
    print(f"  验证缓存: {VAL_CACHE}")
    if args.all:
        print(f"  测试缓存: {TEST_CACHE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
