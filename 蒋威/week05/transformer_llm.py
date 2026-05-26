#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
金融语料 MLM 继续预训练 —— 纯 PyTorch 实现
基于 bert-base-chinese，不依赖 datasets / Trainer
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 国内镜像
import re
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, pipeline


# ==================== 配置 ====================
CORPUS_PATH = r'e:\BaiduNetdiskDownload\week3 深度学习常用组件\corpus.txt'  # 改成你的实际路径
OUTPUT_DIR = r'e:\BaiduNetdiskDownload\week3 深度学习常用组件\finbert_output'
MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 10
LR = 2e-5
WARMUP_STEPS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 特殊 token ID（bert-base-chinese 标准值）
MASK_TOKEN_ID = 103
CLS_TOKEN_ID = 101
SEP_TOKEN_ID = 102
PAD_TOKEN_ID = 0


# ==================== 1. 语料清洗 ====================

def clean_financial_corpus(text: str) -> str:
    text = re.sub(r'</?document[^>]*>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'已有_COUNT_条评论\s*我要评论', '', text)
    text = re.sub(r'欢迎发表评论\s*我要评论', '', text)
    text = re.sub(
        r'^[□⊙○◆■▲▼●\s]*(?:本报记者|记者|编辑|作者|理财周报记者'
        r'|每经记者|证券时报记者|早报记者|文/表|文/图|图/记者)[^\n]*$',
        '',
        text,
        flags=re.MULTILINE,
    )
    lines = text.split('\n')
    filtered = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        floats = re.findall(r'\d+\.\d+', line)
        if (
            len(floats) >= 3
            and len(line) <= 80
            and re.match(r'^[\u4e00-\u9fa5]', line)
        ):
            continue
        if re.match(r'^[\d\s\.\-]+$', line):
            continue
        filtered.append(line)
    
    text = '\n'.join(filtered)
    lines = text.split('\n')
    deduped = []
    prev = None
    for line in lines:
        stripped = line.strip()
        if stripped and stripped != prev:
            deduped.append(line)
            prev = stripped
    text = '\n'.join(deduped)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ==================== 2. 构建样本 ====================

def build_samples(text: str, tokenizer, max_length: int = 256):
    parts = re.split(r'([。！？；\n])', text)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        if i + 1 < len(parts):
            sentences.append(parts[i] + parts[i + 1])
        else:
            sentences.append(parts[i])
    
    samples = []
    current_sents = []
    current_len = 0
    
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        sent_len = len(tokenizer.encode(sent, add_special_tokens=False))
        if sent_len > max_length - 2:
            samples.append(sent)
            continue
        if current_len + sent_len > max_length - 2:
            if current_sents:
                samples.append(''.join(current_sents))
            current_sents = [sent]
            current_len = sent_len
        else:
            current_sents.append(sent)
            current_len += sent_len
    if current_sents:
        samples.append(''.join(current_sents))
    return samples


# ==================== 3. PyTorch Dataset ====================

class MLMDataset(Dataset):
    """
    纯 PyTorch Dataset，不依赖 HuggingFace datasets。
    在 __getitem__ 中做 tokenize，在 collate_fn 中做动态 MLM 掩码。
    """
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),      # [T]
            'attention_mask': encoding['attention_mask'].squeeze(0),  # [T]
        }


def collate_fn(batch, tokenizer, mlm_prob=0.15):
    """
    动态 MLM 掩码：每个 batch 实时随机 mask。
    """
    input_ids = torch.stack([b['input_ids'] for b in batch])           # [B, T]
    attention_mask = torch.stack([b['attention_mask'] for b in batch]) # [B, T]
    
    labels = input_ids.clone()
    labels_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    
    # 逐条 mask
    for b in range(input_ids.size(0)):
        seq = input_ids[b]
        valid_pos = [
            i for i in range(1, seq.size(0) - 1)  # 跳过 [CLS] 和 [SEP]
            if seq[i] != PAD_TOKEN_ID
            and seq[i] not in (CLS_TOKEN_ID, SEP_TOKEN_ID)
        ]
        if not valid_pos:
            continue
        
        n_mask = max(1, int(len(valid_pos) * mlm_prob))
        mask_pos = random.sample(valid_pos, n_mask)
        
        for pos in mask_pos:
            labels_mask[b, pos] = True
            original_id = seq[pos].item()
            prob = random.random()
            
            if prob < 0.8:
                input_ids[b, pos] = MASK_TOKEN_ID
            elif prob < 0.9:
                # 随机替换为词表中的某个词（排除特殊 token）
                input_ids[b, pos] = random.randint(1000, tokenizer.vocab_size - 1)
            # else: 10% 保持不变，只改 labels
    
    # labels：只在 mask 位置保留原 token id，其余设为 -100（loss 忽略）
    labels = torch.where(labels_mask, labels, torch.tensor(-100))
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }


# ==================== 4. 训练 ====================

def get_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / (warmup_steps + 1)
    return base_lr * 0.5 * (1 + torch.cos(torch.tensor(3.14159 * (step - warmup_steps) / (total_steps - warmup_steps))))

def train():
    print(f'使用设备: {DEVICE}')
    
    # 1. 加载 tokenizer & 模型
    print('加载 tokenizer 与模型...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertForMaskedLM.from_pretrained('bert-base-chinese')
    model.to(DEVICE)
    
    # 2. 处理语料
    print('读取并清洗语料...')
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    cleaned = clean_financial_corpus(raw_text)
    samples = build_samples(cleaned, tokenizer, max_length=MAX_LENGTH)
    print(f'构建样本数: {len(samples)}')
    
    # 3. 构建 DataLoader
    dataset = MLMDataset(samples, tokenizer, max_length=MAX_LENGTH)
    
    def _collate(batch):
        return collate_fn(batch, tokenizer, mlm_prob=0.15)
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=_collate,
        num_workers=0,  # Windows 下建议 0
    )
    
    # 4. 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    total_steps = len(dataloader) * EPOCHS
    global_step = 0
    model.train()
    
    print('开始训练...')
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            # 学习率调度
            lr = get_lr(global_step, total_steps, WARMUP_STEPS, LR)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            if global_step % 50 == 0:
                print(f'Epoch {epoch+1}/{EPOCHS} | Step {global_step} | Loss: {loss.item():.4f} | LR: {lr:.2e}')
        
        avg_loss = epoch_loss / len(dataloader)
        print(f'>>> Epoch {epoch+1} 完成，平均 Loss: {avg_loss:.4f}')
    
    # 5. 保存
    print(f'保存模型到 {OUTPUT_DIR}')
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print('完成！')


if __name__ == '__main__':
    # train()
    MODEL_PATH = r'e:\BaiduNetdiskDownload\week3 深度学习常用组件\finbert_output'

    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForMaskedLM.from_pretrained(MODEL_PATH)

    fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

    tests = [
        '黄金由于货币属性强势回归，仍将受到[MASK]资金的青睐。',
        '美联储可能采取新的措施来提振[MASK]经济。',
        '基金裕泽作为一只封闭式基金，开始进入了[MASK]阶段。',
        '近期，国内[MASK]市场出现较大震荡，投资者需保持谨慎。',
        '央行近期采取量化宽松政策，导致[MASK]价格大幅上涨。',
    ]

    for sent in tests:
        print(f'\n输入: {sent}')
        preds = fill_mask(sent)
        for p in preds[:3]:
            print(f"  -> {p['token_str']} (score: {p['score']:.4f})")