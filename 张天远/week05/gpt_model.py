"""
GPT 因果语言模型 — 基于 PyTorch 从零搭建

架构：Pre-Norm Decoder-Only Transformer
  Embedding → [ DecoderLayer × L ] → LayerNorm → LM Head → 词表概率分布

与 BERT 的核心区别：
  1. 因果注意力（前向 mask），每个 token 只能看到自身及之前的 token
  2. 无 token_type_embeddings
  3. Pre-Norm 代替 Post-Norm，训练更稳定
  4. LM Head 代替 Pooler，输出词表概率分布
  5. LM Head 与 Token Embedding 共享权重（weight tying）

参考：GPT-2 / GPT-3 架构（可选 RMSNorm 代替 LayerNorm，参考 LLaMA）
"""
import math
import torch


# ══════════════════════════════════════════════════════════════════
# RMSNorm（Root Mean Square Normalization）
# ══════════════════════════════════════════════════════════════════

class GptRMSNorm(torch.nn.Module):
    """
    RMSNorm：只做缩放归一化，不做均值中心化。

    与 LayerNorm 的区别：
      LayerNorm:  y = (x - μ) / σ × γ + β      (2H 参数)
      RMSNorm:    y = x / RMS(x) × γ             (H 参数，无 β)

    优势：更快（少一次 reduce）、参数更少、LLaMA/Mistral 等验证效果相当。

    参数量 (H=512): H = 512（仅为 LayerNorm 的一半）
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        # RMS = sqrt(mean(x²) + ε)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class GptConfig:
    """
    GPT 模型可配置参数。

    默认值针对 1080Ti 11GB 设计，模型约 36M 参数。
    """
    def __init__(
        self,
        vocab_size: int = 21128,
        hidden_size: int = 512,
        intermediate_size: int = 2048,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        max_position_embeddings: int = 256,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-6,
        pad_token_id: int = 0,
        use_rms_norm: bool = True,       # True=RMSNorm, False=LayerNorm
        use_flash_attn: bool = True,     # True=FlashAttention, False=手工注意力
        use_gradient_checkpoint: bool = False,  # 梯度检查点（省显存）
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.use_rms_norm = use_rms_norm
        self.use_flash_attn = use_flash_attn
        self.use_gradient_checkpoint = use_gradient_checkpoint


# ══════════════════════════════════════════════════════════════════
# Embedding 层
# ══════════════════════════════════════════════════════════════════

class GptEmbedding(torch.nn.Module):
    """
    GPT Embedding：token + position → Dropout

    GPT 不使用 token_type_embeddings，也不在 embedding 后接 LayerNorm。

    参数量 (默认配置 H=512, P=256, V=21128):
      token_embeddings:     V × H  = 21128 × 512 = 10,817,536
      position_embeddings:  P × H  =   256 × 512 =    131,072
      ─────────────────────────────────────────────────────────
      合计:                                        10,948,608  (~10.9M)
    """
    def __init__(self, config: GptConfig):
        super().__init__()
        self.token_embeddings = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = torch.nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).unsqueeze(0),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor):
        seq_length = input_ids.size(1)
        position_ids = self.position_ids[:, :seq_length]
        token_emb = self.token_embeddings(input_ids)
        pos_emb = self.position_embeddings(position_ids)
        embeddings = token_emb + pos_emb
        embeddings = self.dropout(embeddings)
        return embeddings


# ══════════════════════════════════════════════════════════════════
# 因果自注意力（核心差异）
# ══════════════════════════════════════════════════════════════════

class GptCausalSelfAttention(torch.nn.Module):
    """
    因果自注意力：每个 token 只能看到自身及之前的 token（下三角 mask）。

    Q/K/V 投影 + 输出投影 + 因果 mask 都在此类中完成。

    参数量 (默认 H=512):
      Q:  H × H + H = 512² + 512 = 262,656
      K:  H × H + H              = 262,656
      V:  H × H + H              = 262,656
      输出投影: H × H + H        = 262,656
      ─────────────────────────────────────────
      合计: 4 × (H² + H)        = 1,050,624  (~1.05M)
    """
    def __init__(self, config: GptConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size {config.hidden_size} 必须能被 "
                f"num_attention_heads {config.num_attention_heads} 整除"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.use_flash_attn = config.use_flash_attn

        self.query = torch.nn.Linear(config.hidden_size, self.all_head_size)
        self.key = torch.nn.Linear(config.hidden_size, self.all_head_size)
        self.value = torch.nn.Linear(config.hidden_size, self.all_head_size)
        self.output = torch.nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout = torch.nn.Dropout(config.attention_probs_dropout_prob)

    def _split_heads(self, x: torch.Tensor):
        """将 hidden_size 拆分为 (num_heads × head_size)，返回 (batch, heads, seq, head_size)"""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor):
        """合并多头：(batch, heads, seq, head_size) → (batch, seq, hidden_size)"""
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (self.all_head_size,)
        return x.view(*new_shape)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
        query_layer = self._split_heads(self.query(hidden_states))
        key_layer = self._split_heads(self.key(hidden_states))
        value_layer = self._split_heads(self.value(hidden_states))

        if self.use_flash_attn:
            # ── FlashAttention ──
            # is_causal 自动处理前向 mask；attn_mask 处理 padding
            if attention_mask is not None:
                pad_mask = attention_mask[:, None, None, :].to(torch.bool)
            else:
                pad_mask = None

            context_layer = torch.nn.functional.scaled_dot_product_attention(
                query_layer, key_layer, value_layer,
                attn_mask=pad_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
                scale=1.0 / math.sqrt(self.attention_head_size),
            )
        else:
            # ── 手工注意力（与原始实现一致，用于对比）──
            seq_len = hidden_states.size(1)
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=hidden_states.device), diagonal=1
            ) * -10000.0
            attention_scores = attention_scores + causal_mask

            if attention_mask is not None:
                extended_mask = (1.0 - attention_mask[:, None, None, :].to(torch.float32)) * -10000.0
                attention_scores = attention_scores + extended_mask

            attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)
            context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = self._merge_heads(context_layer)

        # 输出投影
        context_layer = self.output(context_layer)
        return context_layer


# ══════════════════════════════════════════════════════════════════
# 前馈网络
# ══════════════════════════════════════════════════════════════════

class GptFeedForward(torch.nn.Module):
    """
    FFN：Linear(H → I) → GELU → Linear(I → H) → Dropout

    参数量 (默认 H=512, I=2048):
      dense_in:   H × I + I = 512 × 2048 + 2048 = 1,050,624
      dense_out:  I × H + H = 2048 × 512 + 512  = 1,049,088
      ─────────────────────────────────────────────────────
      合计:                                        2,099,712  (~2.10M)
    """
    def __init__(self, config: GptConfig):
        super().__init__()
        self.dense_in = torch.nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_out = torch.nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dense_in(hidden_states)
        hidden_states = torch.nn.functional.gelu(hidden_states)
        hidden_states = self.dense_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# ══════════════════════════════════════════════════════════════════
# Decoder 层
# ══════════════════════════════════════════════════════════════════

class GptDecoderLayer(torch.nn.Module):
    """
    单层 Decoder（Pre-Norm 架构）:

        x ─→ LayerNorm → CausalSelfAttention ─→ + ─→ LayerNorm → FFN ─→ + ─→ 输出
        └──────────────────────────────────────┘   └─────────────────────┘

    参数量 (默认配置, RMSNorm):
      GptCausalSelfAttention:  4 × (H² + H) = 1,050,624
      GptFeedForward:          2,099,712
      RMSNorm × 2:             2 × H       =     1,024
      ─────────────────────────────────────────────────
      合计:                                  3,151,360  (~3.15M)
    """
    def __init__(self, config: GptConfig):
        super().__init__()
        norm_cls = GptRMSNorm if config.use_rms_norm else torch.nn.LayerNorm
        norm_kwargs = {"hidden_size": config.hidden_size, "eps": config.layer_norm_eps} \
            if config.use_rms_norm else \
            {"normalized_shape": config.hidden_size, "eps": config.layer_norm_eps}
        self.ln_1 = norm_cls(**norm_kwargs)
        self.attention = GptCausalSelfAttention(config)
        self.ln_2 = norm_cls(**norm_kwargs)
        self.feed_forward = GptFeedForward(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
        # 注意力子层（Pre-Norm）
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = hidden_states + residual

        # FFN 子层（Pre-Norm）
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states


# ══════════════════════════════════════════════════════════════════
# Decoder 堆叠
# ══════════════════════════════════════════════════════════════════

class GptDecoder(torch.nn.Module):
    """
    N 层 GptDecoderLayer 顺序堆叠。

    参数量 (默认 L=8):
      L × 3,152,384 = 8 × 3,152,384 = 25,219,072  (~25.2M)
    """
    def __init__(self, config: GptConfig):
        super().__init__()
        self.use_checkpoint = config.use_gradient_checkpoint
        self.layers = torch.nn.ModuleList(
            [GptDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, attention_mask, use_reentrant=False
                )
            else:
                hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


# ══════════════════════════════════════════════════════════════════
# LM Head
# ══════════════════════════════════════════════════════════════════

class GptLMHead(torch.nn.Module):
    """
    LM Head：Linear(H → V)，将最后一层 hidden state 映射到词表概率。

    通过 weight tying 与 token_embeddings 共享权重，实际不额外占用参数。
    """
    def __init__(self, config: GptConfig):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        return self.dense(hidden_states)


# ══════════════════════════════════════════════════════════════════
# 完整 GPT 模型
# ══════════════════════════════════════════════════════════════════

class MyGpt(torch.nn.Module):
    """
    完整 GPT 模型

    架构:
      Embedding → Decoder × L → 最终 Norm → LM Head → logits

    训练任务:
      因果语言模型 (Causal LM)：用位置 0..i 的 token 预测位置 i+1 的 token

    总参数量 (默认配置, RMSNorm, 权重共享):
      GptEmbedding:           10,948,608  (30.3%)
      GptDecoder × 8:        25,210,880  (69.7%)
      最终 Norm:                    512  (<0.1%)
      LM Head (权重共享):              0
      ─────────────────────────────────────────
      合计:                  36,160,000  ≈ 36.2M

    训练显存估算 (fp32, batch=8, seq=256):
      模型权重:  36M × 4B   =  145 MB
      Adam 状态: 36M × 8B   =  290 MB
      激活值:                ≈ 1000 MB
      ─────────────────────────────────────
      总计:                  ≈ 1.5 GB  (1080Ti 11GB 充足)
    """
    def __init__(self, config: GptConfig):
        super().__init__()
        self.config = config

        self.embeddings = GptEmbedding(config)
        self.decoder = GptDecoder(config)

        norm_cls = GptRMSNorm if config.use_rms_norm else torch.nn.LayerNorm
        norm_kwargs = {"hidden_size": config.hidden_size, "eps": config.layer_norm_eps} \
            if config.use_rms_norm else \
            {"normalized_shape": config.hidden_size, "eps": config.layer_norm_eps}
        self.ln_f = norm_cls(**norm_kwargs)

        self.lm_head = GptLMHead(config)

        # Weight tying: LM Head 与 Token Embedding 共享权重
        self.lm_head.dense.weight = self.embeddings.token_embeddings.weight

        # 参数初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """GPT-2 风格的权重初始化"""
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, (torch.nn.LayerNorm, GptRMSNorm)):
            torch.nn.init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        """
        前向传播。

        Args:
            input_ids:     (batch, seq_len)  输入 token ID
            attention_mask:(batch, seq_len)  1=有效token, 0=padding
            labels:        (batch, seq_len)  目标 token ID（训练时传入）

        Returns:
            dict: {"logits": ..., "loss": ...}

        训练时 loss 的计算方式 (next-token prediction)：
          位置 i 的 hidden state → 预测位置 i+1 的 token
        """
        hidden_states = self.embeddings(input_ids)
        hidden_states = self.decoder(hidden_states, attention_mask)
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )

        return {"logits": logits, "loss": loss}

    # ── 生成方法 ──

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        min_p: float = 0.0,              # Min-P 阈值，0 不启用，推荐 0.05
        repetition_penalty: float = 1.0, # 重复惩罚，1.0 不启用，推荐 1.1
        eos_token_id: int = None,
    ):
        """
        自回归文本生成。

        采样流程：temperature → repetition_penalty → top-k → top-p → min-p → 多项式采样

        Args:
            input_ids:           (batch, seq_len)  初始序列
            max_new_tokens:      最大生成 token 数
            temperature:         温度系数，<1 更确定，=0 贪心解码
            top_k:               top-k 过滤，0 不启用
            top_p:               nucleus 过滤，1.0 不启用
            min_p:               Min-P 过滤，保留概率 ≥ min_p × max_prob 的 token
            repetition_penalty:  重复惩罚，>1 惩罚已出现过的 token
            eos_token_id:        终止 token ID

        Returns:
            (batch, seq_len + 实际生成长度) 完整序列
        """
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            if generated.size(1) > self.config.max_position_embeddings:
                generated = generated[:, -self.config.max_position_embeddings:]

            outputs = self.forward(generated)
            next_logits = outputs["logits"][:, -1, :]

            # ── Temperature ──
            if temperature > 0:
                next_logits = next_logits / temperature
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=-1)
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
                continue

            # ── Repetition Penalty ──
            # 对已出现的 token 施加惩罚：logit>0 时除以 penalty，logit<0 时乘以 penalty
            if repetition_penalty != 1.0:
                for b in range(generated.size(0)):
                    seen = set(generated[b].tolist())
                    for token_id in seen:
                        if next_logits[b, token_id] > 0:
                            next_logits[b, token_id] /= repetition_penalty
                        else:
                            next_logits[b, token_id] *= repetition_penalty

            # ── Top-k ──
            if top_k > 0:
                top_k_val = min(top_k, next_logits.size(-1))
                top_k_values, _ = torch.topk(next_logits, top_k_val, dim=-1)
                min_top_k = top_k_values[:, -1].unsqueeze(-1)
                next_logits[next_logits < min_top_k] = -float("inf")

            # ── Top-p (nucleus) ──
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(
                    torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = -float("inf")

            # ── Min-P ──
            # 保留概率 ≥ min_p × max_prob 的 token，简单有效
            if min_p > 0:
                probs_check = torch.nn.functional.softmax(next_logits, dim=-1)
                max_prob = probs_check.max(dim=-1, keepdim=True).values
                next_logits[probs_check < min_p * max_prob] = -float("inf")

            # ── 采样 ──
            probs = torch.nn.functional.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated


# ══════════════════════════════════════════════════════════════════
# 构建函数
# ══════════════════════════════════════════════════════════════════

def build_gpt(config: GptConfig = None):
    """构建 GPT 模型并打印参数量。"""
    if config is None:
        config = GptConfig()
    model = MyGpt(config)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"=== GPT 模型参数 ===")
    print(f"  hidden_size:             {config.hidden_size}")
    print(f"  intermediate_size:       {config.intermediate_size}")
    print(f"  num_hidden_layers:       {config.num_hidden_layers}")
    print(f"  num_attention_heads:     {config.num_attention_heads}")
    print(f"  max_position_embeddings: {config.max_position_embeddings}")
    print(f"  vocab_size:              {config.vocab_size}")
    print(f"  总参数量:                {total:,}")
    print(f"  可训练参数量:             {trainable:,}")
    return model, config


# ══════════════════════════════════════════════════════════════════
# 自测（直接运行 python gpt_model.py）
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("测试 GPT 模型结构...\n")
    model, config = build_gpt()

    # 前向传播
    print("\n[1] 训练模式前向传播")
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    model.train()
    outputs = model(input_ids, labels=labels)
    print(f"  输入:  {input_ids.shape}")
    print(f"  Logits: {outputs['logits'].shape}")
    print(f"  Loss:   {outputs['loss'].item():.4f}")

    # 因果注意力验证
    print("\n[2] 因果注意力验证")
    print("  修改位置 2 的输入 → 位置 0/1 的输出应不变")
    x1 = torch.randint(0, config.vocab_size, (1, 4))
    x2 = x1.clone()
    x2[0, 2] = (x2[0, 2] + 1) % config.vocab_size
    out1, out2 = model(x1)["logits"], model(x2)["logits"]
    d0 = (out1[0, 0] - out2[0, 0]).abs().max().item()
    d1 = (out1[0, 1] - out2[0, 1]).abs().max().item()
    d2 = (out1[0, 2] - out2[0, 2]).abs().max().item()
    d3 = (out1[0, 3] - out2[0, 3]).abs().max().item()
    print(f"  位置0: {d0:.6f} (期望 0)  位置1: {d1:.6f} (期望 0)")
    print(f"  位置2: {d2:.6f} (期望>0)  位置3: {d3:.6f} (期望>0)")
    ok = d0 < 1e-6 and d1 < 1e-6 and d2 > 0 and d3 > 0
    print(f"  结果: {'✓ 通过' if ok else '✗ 失败'}")

    # 生成
    print("\n[3] 生成测试")
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    gen = model.generate(prompt, max_new_tokens=10, temperature=0.8)
    print(f"  Prompt:  {prompt.shape} → 生成后: {gen.shape}")
    print("\n模型结构测试完成。")
