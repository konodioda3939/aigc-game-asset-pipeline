"""自动打标：使用WD14 SwinV2 Tagger v3 (ONNX) 为每张图生成Danbooru风格标签，写入同名.txt"""
import os, sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
import numpy as np
from PIL import Image
import csv

# ===== HuggingFace镜像 =====
HF_MIRROR = "https://hf-mirror.com"
if HF_MIRROR:
    os.environ['HF_ENDPOINT'] = HF_MIRROR
    print(f"Using HF mirror: {HF_MIRROR}")

# 配置
IMAGE_DIR = Path(r"D:\aigc-project\data\processed")
THRESHOLD = 0.35
MODEL_NAME = "SmilingWolf/wd-swinv2-tagger-v3"   # 改用v3，社区主力模型
TARGET_SIZE = 448

print("Loading WD14 tagger v3 (ONNX)...")
print(f"Model: {MODEL_NAME}")

# 1. 下载并加载 ONNX 模型
from huggingface_hub import hf_hub_download
import onnxruntime as ort

print("Downloading model.onnx...")
onnx_path = hf_hub_download(repo_id=MODEL_NAME, filename="model.onnx", repo_type="model")

session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"ONNX input: {input_name}, shape: {input_shape} (NHWC, model has internal transpose)")

# 2. 加载标签列表
print("Loading tag list...")
tags_csv_path = hf_hub_download(repo_id=MODEL_NAME, filename="selected_tags.csv", repo_type="model")

tags = []
with open(tags_csv_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        tags.append(row[1])  # name

print(f"Loaded {len(tags)} tags")

# 3. 预处理函数
# ONNX模型内嵌了完整预处理: Transpose(NHWC→NCHW) → /127.5 → -1.0
# 因此只需: 中心裁切 → resize → 原始[0,255]像素值 → NHWC
def preprocess(img_path: Path) -> np.ndarray:
    """中心裁切正方形 -> resize -> 保持原始[0,255] -> NHWC"""
    img = Image.open(img_path).convert("RGB")

    # 中心裁切为正方形
    w, h = img.size
    if w > h:
        left = (w - h) // 2
        img = img.crop((left, 0, left + h, h))
    else:
        top = (h - w) // 2
        img = img.crop((0, top, w, top + w))

    # Resize
    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

    # 原始[0,255]像素值，NHWC格式（模型内嵌Transpose会转成NCHW）
    arr = np.array(img, dtype=np.float32)
    arr = arr[np.newaxis, ...]  # NHWC: [1, H, W, 3]
    return arr

# 4. 处理每张图片
images = sorted([f for f in IMAGE_DIR.iterdir() if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
print(f"\nProcessing {len(images)} images...\n")

for i, img_path in enumerate(images):
    try:
        inputs = preprocess(img_path)
        raw_output = session.run(None, {input_name: inputs})[0].squeeze(0)

        # 检测模型输出是否已含sigmoid
        if raw_output.min() >= 0 and raw_output.max() <= 1:
            probs = raw_output
        else:
            probs = 1.0 / (1.0 + np.exp(-raw_output))

        # 筛选高于阈值的标签
        selected_indices = np.where(probs > THRESHOLD)[0]
        selected_tags = [(tags[idx], float(probs[idx])) for idx in selected_indices]
        selected_tags.sort(key=lambda x: x[1], reverse=True)

        caption = ", ".join([tag for tag, _ in selected_tags])

        txt_path = img_path.with_suffix(".txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(caption)

        top5 = ", ".join([f"{tag}({conf:.0%})" for tag, conf in selected_tags[:5]])
        print(f"  [{i+1:2d}] {img_path.name[:45]:<45} {len(selected_tags):2d} tags  |  {top5}")

    except Exception as e:
        print(f"  [{i+1:2d}] {img_path.name[:45]}  ERROR: {e}")

print(f"\nDone! Captions saved to {IMAGE_DIR}\\*.txt")
print(f"Next: Manually review and edit the .txt files, then we train LoRA.")
