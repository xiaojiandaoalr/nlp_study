"""
GPT 模型快速自测

运行方式：
  python diy-gpt.py              # 使用默认配置测试
  python diy-gpt.py --small      # 使用小模型测试（更快）
  python diy-gpt.py --large      # 使用大模型测试（需要更多显存）

此文件是 gpt_model.py 的轻量入口，保留用于快速验证模型结构。
完整训练请分别运行：preprocess.py → train.py → test.py
"""
import sys
import argparse
import torch
from gpt_model import GptConfig, MyGpt, build_gpt


def run_tests(config: GptConfig):
    model, _ = build_gpt(config)

    print("\n" + "=" * 60)
    print("开始模型结构测试")
    print("=" * 60)

    # ── 测试 1: 前向传播 ──
    print("\n[1] 训练模式前向传播")
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    model.train()
    outputs = model(input_ids, labels=labels)
    print(f"  输入:        {input_ids.shape}")
    print(f"  Logits:      {outputs['logits'].shape}")
    print(f"  Loss:        {outputs['loss'].item():.4f}")

    # ── 测试 2: padding mask ──
    print("\n[2] 带 padding mask 的训练")
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[:, -5:] = 0
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    print(f"  Logits:      {outputs['logits'].shape}")
    print(f"  Loss:        {outputs['loss'].item():.4f}")

    # ── 测试 3: 生成策略 ──
    print("\n[3] 生成策略")
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    print(f"  Prompt shape: {prompt.shape}")

    greedy = model.generate(prompt, max_new_tokens=10, temperature=0)
    print(f"  贪心 (T=0):     {greedy.shape}")

    sampled = model.generate(prompt, max_new_tokens=10, temperature=0.8)
    print(f"  温度 (T=0.8):   {sampled.shape}")

    topk = model.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=40, top_p=1.0)
    print(f"  Top-k (k=40):   {topk.shape}")

    nucleus = model.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=50, top_p=0.95)
    print(f"  Top-p (p=0.95): {nucleus.shape}")

    # ── 测试 4: 因果注意力 ──
    print("\n[4] 因果注意力验证")
    print("  修改位置 2 的输入 → 位置 0/1 的输出应不变")
    x1 = torch.randint(0, config.vocab_size, (1, 4))
    x2 = x1.clone()
    x2[0, 2] = (x2[0, 2] + 1) % config.vocab_size
    out1, out2 = model(x1)["logits"], model(x2)["logits"]
    d0 = (out1[0, 0] - out2[0, 0]).abs().max().item()
    d1 = (out1[0, 1] - out2[0, 1]).abs().max().item()
    d2 = (out1[0, 2] - out2[0, 2]).abs().max().item()
    d3 = (out1[0, 3] - out2[0, 3]).abs().max().item()
    print(f"  位置0: {d0:.6f} (期望 0)    位置1: {d1:.6f} (期望 0)")
    print(f"  位置2: {d2:.6f} (期望>0)   位置3: {d3:.6f} (期望>0)")
    ok = d0 < 1e-6 and d1 < 1e-6 and d2 > 0 and d3 > 0
    print(f"  因果注意力: {'✓ 通过' if ok else '✗ 失败'}")

    print("\n" + "=" * 60)
    print("所有测试完成。")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT 模型快速自测")
    parser.add_argument("--small", action="store_true", help="使用小模型测试")
    parser.add_argument("--large", action="store_true", help="使用大模型测试")
    args = parser.parse_args()

    if args.small:
        config = GptConfig(hidden_size=256, intermediate_size=1024,
                           num_hidden_layers=4, num_attention_heads=4,
                           max_position_embeddings=128)
    elif args.large:
        config = GptConfig(hidden_size=768, intermediate_size=3072,
                           num_hidden_layers=12, num_attention_heads=12,
                           max_position_embeddings=512)
    else:
        config = GptConfig()  # 默认 ~36M

    run_tests(config)
