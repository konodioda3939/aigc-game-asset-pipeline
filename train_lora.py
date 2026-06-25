"""使用HuggingFace Diffusers训练SD 1.5 LoRA（风格学习）"""
import os, sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# HF镜像（必须在导入diffusers之前）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

# ==================== 配置参数 ====================
PRETRAINED_MODEL = "gsdf/Counterfeit-V2.5"  # 动漫专用基座
# "runwayml/stable-diffusion-v1-5"  # 原版SD1.5（通用但动漫差）
DATA_DIR = Path(r"D:\aigc-project\data\processed")
OUTPUT_DIR = Path(r"D:\aigc-project\lora_output")
CACHE_DIR = Path(r"D:\aigc-project\cache")

RESOLUTION = 512          # SD1.5原生分辨率
RANK = 16                 # LoRA rank (8~32)
LEARNING_RATE = 1e-4
BATCH_SIZE = 1            # 8GB显存用1
GRADIENT_ACCUMULATION = 4 # 等效batch=4
MAX_TRAIN_STEPS = 1200    # 48图 × ~100 epochs，让风格学得更充分
SAVE_EVERY = 200          # 每N步保存一次
SEED = 42
# =================================================

torch.manual_seed(SEED)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Device: {device}, dtype: {dtype}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"Output: {OUTPUT_DIR}")

# ==================== 阶段1: 缓存 VAE latents ====================
print("\n" + "="*60)
print("Phase 1/3: Caching VAE latents...")
print("="*60)

from diffusers import AutoencoderKL

vae = AutoencoderKL.from_pretrained(
    PRETRAINED_MODEL, subfolder="vae",
    torch_dtype=dtype, cache_dir=str(CACHE_DIR / "hub")
).to(device)
vae.eval()

# SD 1.5 VAE 缩放因子
vae_scale_factor = 0.18215

image_files = sorted([
    f for f in DATA_DIR.iterdir()
    if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}
])

latents_cache = {}
with torch.no_grad():
    for img_path in tqdm(image_files, desc="Encoding images"):
        img = Image.open(img_path).convert("RGB")
        img = img.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) * 2.0  # [-1, 1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, dtype=dtype)

        latent = vae.encode(tensor).latent_dist.sample() * vae_scale_factor
        latents_cache[img_path.stem] = latent.cpu()  # [1, 4, 64, 64]

# 释放VAE
del vae
torch.cuda.empty_cache()
print(f"Cached {len(latents_cache)} latents")

# ==================== 阶段2: 缓存文本嵌入 ====================
print("\n" + "="*60)
print("Phase 2/3: Caching text embeddings...")
print("="*60)

from transformers import CLIPTokenizer, CLIPTextModel

tokenizer = CLIPTokenizer.from_pretrained(
    PRETRAINED_MODEL, subfolder="tokenizer",
    cache_dir=str(CACHE_DIR / "hub")
)
text_encoder = CLIPTextModel.from_pretrained(
    PRETRAINED_MODEL, subfolder="text_encoder",
    torch_dtype=dtype, cache_dir=str(CACHE_DIR / "hub")
).to(device)
text_encoder.eval()

text_cache = {}
with torch.no_grad():
    for img_path in tqdm(image_files, desc="Encoding captions"):
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists():
            print(f"  WARNING: No caption for {img_path.name}, skipping")
            continue

        caption = txt_path.read_text(encoding='utf-8').strip()
        if not caption:
            caption = "anime style"  # fallback

        tokens = tokenizer(
            caption, max_length=tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}

        emb = text_encoder(**tokens).last_hidden_state.cpu()  # [1, 77, 768]
        text_cache[img_path.stem] = emb

# 释放文本编码器
del text_encoder
torch.cuda.empty_cache()

# 对齐：只保留同时有图片和文本的
valid_keys = sorted(set(latents_cache.keys()) & set(text_cache.keys()))
print(f"Valid pairs: {len(valid_keys)}")

# ==================== 阶段3: 训练UNet LoRA ====================
print("\n" + "="*60)
print("Phase 3/3: Training LoRA on UNet...")
print("="*60)

from diffusers import UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig, get_peft_model, TaskType

# 噪声调度器
noise_scheduler = DDPMScheduler.from_pretrained(
    PRETRAINED_MODEL, subfolder="scheduler",
    cache_dir=str(CACHE_DIR / "hub")
)

# UNet — 以fp16加载基座，LoRA层peft会自动创建为fp32
unet = UNet2DConditionModel.from_pretrained(
    PRETRAINED_MODEL, subfolder="unet",
    torch_dtype=dtype, cache_dir=str(CACHE_DIR / "hub")
)

# 冻结原始权重
unet.requires_grad_(False)

# 配置LoRA（只训attention层的q,k,v,o投影矩阵）
lora_config = LoraConfig(
    r=RANK,
    lora_alpha=RANK,         # alpha = rank 是常用设置
    target_modules=[
        "to_q", "to_k", "to_v", "to_out.0"  # attention层
    ],
    lora_dropout=0.0,
    bias="none",
)

unet = get_peft_model(unet, lora_config)
unet.enable_gradient_checkpointing()  # 省显存
# 注意：只移到device，不指定dtype——保留LoRA层为fp32
unet = unet.to(device)
unet.train()

# 统计可训练参数
trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
total = sum(p.numel() for p in unet.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
print(f"Target modules: {lora_config.target_modules}")

# 优化器
optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)

# 混合精度缩放器（LoRA是fp32，GradScaler可以正常工作）
scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

# 训练循环
global_step = 0
losses = []
optimizer.zero_grad()

# 构建数据集
dataset = [
    (latents_cache[k], text_cache[k]) for k in valid_keys
]

pbar = tqdm(total=MAX_TRAIN_STEPS, desc="Training")
while global_step < MAX_TRAIN_STEPS:
    # 随机采样一个batch
    indices = np.random.choice(len(dataset), BATCH_SIZE, replace=False)
    latents = torch.cat([dataset[i][0] for i in indices]).to(device, dtype=dtype)
    encoder_hidden_states = torch.cat([dataset[i][1] for i in indices]).to(device, dtype=dtype)

    # 随机噪声 + 随机timestep
    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps,
        (BATCH_SIZE,), device=device
    ).long()

    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # autocast: 前向用fp16省显存，反向自动fp32
    with torch.amp.autocast('cuda', enabled=(device == "cuda")):
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = F.mse_loss(noise_pred, noise)
        loss = loss / GRADIENT_ACCUMULATION

    scaler.scale(loss).backward()

    if (global_step + 1) % GRADIENT_ACCUMULATION == 0:
        # 梯度裁剪防止fp16溢出导致的尖峰
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    loss_val = loss.item() * GRADIENT_ACCUMULATION
    losses.append(loss_val)
    global_step += 1
    pbar.update(1)
    pbar.set_postfix({"loss": f"{loss_val:.4f}" if not np.isnan(loss_val) else "nan"})

    # 检测NaN，中止训练
    if np.isnan(loss_val) and global_step > 5:
        print(f"\n  ERROR: Loss became NaN at step {global_step}. Training diverged.")
        print(f"  Try lowering learning rate or checking data.")
        break

    # 保存检查点
    if global_step % SAVE_EVERY == 0 or global_step >= MAX_TRAIN_STEPS:
        save_path = OUTPUT_DIR / f"checkpoint-{global_step}"
        save_path.mkdir(exist_ok=True)
        unet.save_pretrained(save_path)
        print(f"\n  Saved LoRA to {save_path}")

pbar.close()

# ==================== 最终保存 ====================
print("\n" + "="*60)
print("Saving final LoRA weights...")
print("="*60)

unet.save_pretrained(OUTPUT_DIR / "final")
# 也保存一份方便直接加载的权重
unet.save_pretrained(OUTPUT_DIR)

print(f"\nDone! LoRA weights saved to: {OUTPUT_DIR}")
print(f"Final loss: {losses[-1]:.6f}")
print(f"Avg loss (last 100): {np.mean(losses[-100:]):.6f}")
print(f"\nNext: Run inference to compare with/without LoRA (step 1.5)")
