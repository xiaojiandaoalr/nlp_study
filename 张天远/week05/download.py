import os
import json
from datasets import load_dataset

# ================= 1. 配置与环境 =================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

dataset_name = "0xDing/wikipedia-cn-20230720-filtered"
output_dir = r"E:/npl/CN_Corpus"
jsonl_file = os.path.join(output_dir, "wiki_zh_raw.jsonl")
txt_file = os.path.join(output_dir, "wiki_zh.txt")

os.makedirs(output_dir, exist_ok=True)

# ================= 2. 下载阶段 =================
print(f"🚀 正在从镜像站下载数据集: {dataset_name} ...")
try:
    # 加载数据集
    dataset = load_dataset(dataset_name, split="train")
    print(f"✅ 下载完成！共加载 {len(dataset)} 条数据。")

    # 保存为 JSONL
    print(f"💾 正在保存中间文件 (JSONL) 到: {jsonl_file} ...")
    dataset.to_json(jsonl_file, force_ascii=False, lines=True)
    print("✅ JSONL 保存成功。")

except Exception as e:
    print(f"❌ 下载失败: {e}")
    exit()

# ================= 3. 转换阶段 (强力逻辑) =================
print("\n" + "="*30)
print("🔄 开始转换 JSONL 为纯文本 TXT...")
print("="*30)

count = 0
empty_count = 0

try:
    with open(jsonl_file, 'r', encoding='utf-8') as f_in, \
         open(txt_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)

                # 🔑 核心逻辑：多重备选键名 (完全复刻你的参考逻辑)
                # 优先找 completion，没有找 text，再没有找 content
                text = item.get('completion', '') or item.get('text', '') or item.get('content', '')

                text = text.strip()

                if text:
                    f_out.write(text + '\n')
                    count += 1

            except json.JSONDecodeError:
                # 跳过坏掉的 JSON 行
                pass

            # 实时打印进度
            if count % 50000 == 0 and count > 0:
                print(f"  📝 已写入有效文本: {count:,} 条...")

    print(f"🎉 转换完成！")
    print(f"📊 共写入有效文本行数: {count:,}")
    print(f"📄 最终文件路径: {txt_file}")

except FileNotFoundError:
    print(f"❌ 错误：找不到文件 {jsonl_file}，请检查下载步骤。")