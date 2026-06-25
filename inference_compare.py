"""1.5 验证LoRA效果：用同一组prompt对比加载前后的生成效果"""
import os, sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel

# ===== 配置 =====
# 动漫专用基座模型（Counterfeit-V2.5 = 社区公认最好的动漫SD1.5之一）
MODEL_NAME = "gsdf/Counterfeit-V2.5"
# 备选: "Linaqruf/anything-v3.0", "hakurei/waifu-diffusion"
LORA_PATH = r"D:\aigc-project\lora_output"
COMPARISON_DIR = Path(r"D:\aigc-project\lora_output\comparison")
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

# 测试prompt — 用训练数据中的标签风格，才能触发LoRA学到的画风
TEST_PROMPTS = [
    "1girl, solo, long hair, looking at viewer, multicolored hair, portrait, colored skin",
    "1girl, raiden shogun, japanese clothes, long hair, purple hair, braid, purple eyes, flower",
    "1boy, solo, male focus, short hair, green eyes, gloves, black gloves",
    "1girl, furina_(genshin_impact), heterochromia, white hair, blue eyes, top hat, ascot",
    "1girl, hu tao, twintails, brown hair, smile, flower-shaped pupils, chinese clothes",
    "1girl, solo, nilou_(genshin_impact), long hair, blue hair, horns, veil, water, dancing",
    "1boy, kaedehara kazuha, streaked hair, white hair, red eyes, japanese clothes, autumn leaves",
    "1girl, lumine_(genshin_impact), blonde hair, dress, white dress, flower, bare shoulders",
]

NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"

SEED = 42
STEPS = 25
GUIDANCE_SCALE = 7.5
# =================

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Device: {device}, dtype: {dtype}")

# ===== 加载基础模型 =====
print("Loading SD 1.5 base model...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    safety_checker=None,
    cache_dir=r"D:\aigc-project\cache\hub",
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to(device)
pipe.enable_attention_slicing()

# ===== Step 1: 无LoRA baseline =====
print("\n" + "="*60)
print("Step 1/2: Generating WITHOUT LoRA (baseline)...")
print("="*60)

for i, prompt in enumerate(TEST_PROMPTS):
    generator = torch.Generator(device).manual_seed(SEED + i * 100)

    with torch.no_grad():
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
        ).images[0]

    fname = f"without_lora_{i+1:02d}.png"
    image.save(COMPARISON_DIR / fname)
    print(f"  [{i+1:2d}/{len(TEST_PROMPTS)}] {fname}  |  {prompt[:60]}...")

# ===== Step 2: 用PEFT加载LoRA（不是pipe.load_lora_weights） =====
print("\n" + "="*60)
print("Step 2/2: Loading LoRA via PEFT and generating...")
print("="*60)

# PEFT方式加载：直接在UNet上挂LoRA，然后merge进基础权重
unet = PeftModel.from_pretrained(pipe.unet, LORA_PATH)
pipe.unet = unet.merge_and_unload()  # 把LoRA融进UNet，变成标准UNet
pipe.unet.to(device, dtype=dtype)
print("LoRA merged into UNet")

for i, prompt in enumerate(TEST_PROMPTS):
    generator = torch.Generator(device).manual_seed(SEED + i * 100)

    with torch.no_grad():
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
        ).images[0]

    fname = f"with_lora_{i+1:02d}.png"
    image.save(COMPARISON_DIR / fname)
    print(f"  [{i+1:2d}/{len(TEST_PROMPTS)}] {fname}  |  {prompt[:60]}...")

print(f"\nDone! {len(TEST_PROMPTS)*2} images saved to: {COMPARISON_DIR}")
print(f"  without_lora_*.png  — SD 1.5 baseline")
print(f"  with_lora_*.png     — SD 1.5 + merged LoRA")
print(f"\nCompare side-by-side pairs with the same seed to see the style difference.")
