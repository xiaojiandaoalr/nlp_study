"""
GPT 因果语言模型 — 配置文件

集中管理：路径、模型参数、训练超参数、生成参数
各模块（preprocess / train / test）统一从此处读取配置。

所有路径均相对于目录树自动推导，本地和云端无需手动改路径：
  NPL_ROOT/
  ├── hub-kzSW/          ← 代码仓库 (REPO_ROOT)
  │   └── 张天远/week05/ ← 本项目 (BASE_DIR)
  └── CN_Corpus/         ← 外部语料 (SogouC / WikiText-zh)
"""
import os

# HuggingFace 缓存路径（必须在 import transformers 之前设置）
# HF 缓存：优先数据盘（云平台）→ 盘符（Windows）→ 默认 ~/.cache
if os.name == "nt":
    for _drive in ["M:", "D:"]:
        if os.path.exists(_drive + "\\"):
            os.environ["HF_HOME"] = _drive + "\\huggingface_cache"
            break
else:
    for _path in ["/root/autodl-tmp/huggingface_cache", "/data/huggingface_cache"]:
        if os.path.exists(_path.rsplit("/", 1)[0]):
            os.environ["HF_HOME"] = _path
            break
        # 云平台直连 HF 大概率超时，用镜像
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# 自动离线：缓存已存在才离线，首次下载不阻拦
if os.path.exists(os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")):
    os.environ["HF_HUB_OFFLINE"] = "1"

# ── 目录层级：自动推导，跨平台通用 ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))               # .../hub-kzSW/张天远/week05
REPO_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))               # .../hub-kzSW
NPL_ROOT = os.path.dirname(REPO_ROOT)                                # .../npl (语料所在层)

CACHE_DIR = os.path.join(BASE_DIR, "cache")
MODEL_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# ---- 数据文件 ----
CNEWS_DIR = os.path.join(REPO_ROOT, "张天远", "week04", "bert_news_classify", "cnews")
CN_CORPUS_DIR = os.path.join(NPL_ROOT, "CN_Corpus", "SogouC.reduced", "Reduced")
WIKI_FILE = os.path.join(NPL_ROOT, "CN_Corpus", "wiki_zh.txt")

TRAIN_FILE = os.path.join(CNEWS_DIR, "cnews.train.txt")
VAL_FILE = os.path.join(CNEWS_DIR, "cnews.val.txt")
TEST_FILE = os.path.join(CNEWS_DIR, "cnews.test.txt")

# ---- 预处理缓存 ----
TRAIN_CACHE = os.path.join(CACHE_DIR, "train_tokens.pt")
VAL_CACHE = os.path.join(CACHE_DIR, "val_tokens.pt")
TEST_CACHE = os.path.join(CACHE_DIR, "test_tokens.pt")
COMBINED_CACHE = os.path.join(CACHE_DIR, "combined_train_tokens.pt")  # cnews + sogou 合并

# ---- Tokenizer ----
TOKENIZER_NAME = "bert-base-chinese"

# ---- GPT 模型参数 ----
# 默认规模 (~36M 参数)，可根据需要调整：
#   想省显存：减小 HIDDEN_SIZE / NUM_HIDDEN_LAYERS
#   想更强效果：增大 HIDDEN_SIZE / NUM_HIDDEN_LAYERS
VOCAB_SIZE = 21128                    # 词表大小（与 bert-base-chinese 一致）
HIDDEN_SIZE = 768                     # 隐藏层维度 H
INTERMEDIATE_SIZE = 3072              # FFN 中间层维度（通常 4×H）
NUM_HIDDEN_LAYERS = 12                 # Decoder 层数 L
NUM_ATTENTION_HEADS = 12               # 注意力头数 A
MAX_POSITION_EMBEDDINGS = 512         # 最大序列长度 P
HIDDEN_DROPOUT_PROB = 0.05            # 隐藏层 dropout（小模型不宜太高）
ATTENTION_PROBS_DROPOUT_PROB = 0.05   # 注意力 dropout
LAYER_NORM_EPS = 1e-6                  # LayerNorm / RMSNorm epsilon
PAD_TOKEN_ID = 0                      # padding token ID
USE_RMS_NORM = True                   # True=RMSNorm(推荐,省参数), False=LayerNorm

# ---- 训练参数 (1080Ti 11GB) ----
MAX_SEQ_LEN = 512                     # 训练序列长度
MIN_SEQ_LEN = 16                      # 过滤短文本的最小长度
BATCH_SIZE = 18                        # 单卡 batch
GRADIENT_ACCUMULATION = 4             # 梯度累积，等效 batch = 8 × 4 = 32
EPOCHS = 20                           # 训练轮数（从头训练需要更多轮次）
LEARNING_RATE = 5e-4                  # GPT 从头训练的学习率
WARMUP_EPOCHS = 1                     # warmup 轮数
WEIGHT_DECAY = 0.01                   # AdamW 权重衰减
MAX_GRAD_NORM = 1.0                   # 梯度裁剪阈值
LOG_INTERVAL = 50                     # 每 N 个 batch 打印日志
SAVE_EPOCHS = 2                       # 每 N 轮保存一次 checkpoint
EARLY_STOP_PATIENCE = 3               # 早停耐心值（验证 loss 不降 N 轮则停止）

# ---- DataLoader ----
NUM_WORKERS = 0
PIN_MEMORY = True
USE_AMP = False                      # 混合精度（1080Ti 不支持，4060Ti 设为 True）
AMP_DTYPE = "bf16"                   # 混合精度类型：fp16（1080/3090需GradScaler） / bf16（4060+ Ada推荐，无需GradScaler，兼容compile）
USE_FUSED_ADAMW = False              # Fused AdamW（CUDA 融合版，约 +5% 速度）
USE_COMPILE = False                  # torch.compile（PyTorch 2.0+，约 +20-40% 速度）
COMPILE_MODE = "default"              # compile 模式：default / reduce-overhead(CUDA Graph与embedding有兼容问题) / max-autotune
COMPILE_CACHE_LIMIT = 128            # 编译缓存上限（Ada 小 L2 建议 128，默认 64）
USE_GRADIENT_CHECKPOINT = False      # 梯度检查点（省 40-60% 激活显存，代价约 -10% 速度）
MATMUL_PRECISION = "high"            # TF32 精度：high / medium（省显存加速）/ highest
USE_FLASH_ATTN = True                # FlashAttention（PyTorch 2.0+ 推荐开启）

# ---- 生成参数 ----
GEN_PROMPT = "今天"                   # 训练中验证时用于生成测试的 prompt
GEN_LENGTH = 30                       # 生成 token 数
GEN_TEMPERATURE = 0.8                 # 温度
GEN_TOP_K = 40                        # Top-k
GEN_TOP_P = 0.95                      # Top-p
GEN_MIN_P = 0.05                      # Min-P 阈值 (0 关闭)
GEN_REPETITION_PENALTY = 1.1          # 重复惩罚 (1.0 关闭)

# 确保目录存在
for d in [CACHE_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)
