"""模型加载器：加载Stable Diffusion + LoRA权重，全局只加载一次，复用管线。"""
import os
import sys

# ==== 打印立即刷新，不做缓冲（用户能看到每一步进度，不会以为卡了）====
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ==== 必须在 import diffusers 之前设置，否则连不上 HuggingFace ====
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel


# ===== 配置 =====
# 基座模型：动漫专用 SD 1.5
MODEL_NAME = "gsdf/Counterfeit-V2.5"

# LoRA 权重路径（里程碑1训练产出）
# 用户预期文件名可能是 pytorch_lora_weights.safetensors，实际文件是 adapter_model.safetensors
LORA_DIR = Path(r"D:\aigc-project\lora_output")

# 模型缓存目录（和训练脚本共用，避免重复下载）
CACHE_DIR = Path(r"D:\aigc-project\cache\hub")


# ===== 全局单例 =====
_pipeline = None  # 加载一次，整个服务生命周期复用


def load_pipeline() -> StableDiffusionPipeline:
    """
    加载 Stable Diffusion 管线 + LoRA 权重。

    流程：
    1. 加载 Counterfeit-V2.5 基座模型（动漫专用 SD 1.5）
    2. 通过 PEFT 挂载已训练的 LoRA 权重
    3. 将 LoRA 融合进 UNet（merge_and_unload），变成标准 UNet
    4. 配置 DPMSolver 调度器（比默认 Euler 收敛更快）

    返回：
        配置好的 StableDiffusionPipeline，可直接调用生成图片

    说明：
        这个函数只在服务启动时调用一次，之后全局复用管线对象。
        LoRA 融合到 UNet 后，推理速度和不加 LoRA 一样快。
    """
    global _pipeline

    # 如果已经加载过，直接返回（服务重启前一直有效）
    if _pipeline is not None:
        print("[model_loader] 管线已加载，复用现有实例。", flush=True)
        return _pipeline

    # ---------- 检测运行设备 ----------
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        print(f"[model_loader] 使用 GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"[model_loader] 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB", flush=True)
    else:
        device = "cpu"
        dtype = torch.float32
        print("[model_loader] ⚠️ 未检测到 GPU，使用 CPU（生成会较慢）", flush=True)

    # ---------- 加载基座模型 ----------
    print(f"[model_loader] 正在加载基座模型: {MODEL_NAME} ...", flush=True)
    print(f"[model_loader] 缓存目录: {CACHE_DIR}", flush=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        safety_checker=None,          # 禁用安全检查器（动漫图不需要，也省显存）
        cache_dir=str(CACHE_DIR),
    )

    # 使用 DPMSolver 调度器 — 比默认 DDPM 快 2-3 倍，25 步就能出好图
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    print("[model_loader] 基座模型加载完成。", flush=True)

    # ---------- 加载 LoRA 权重 ----------
    lora_weights = LORA_DIR / "adapter_model.safetensors"
    if lora_weights.exists():
        print(f"[model_loader] 正在加载 LoRA 权重: {lora_weights}", flush=True)

        # PEFT 方式加载：在 UNet 上挂 LoRA 适配器，然后融合进基础权重
        # 不能用 pipe.load_lora_weights() — PEFT 保存的 key 命名格式不兼容
        unet = PeftModel.from_pretrained(pipe.unet, str(LORA_DIR))
        pipe.unet = unet.merge_and_unload()          # 融合：把 LoRA 增量融进 UNet 原始权重
        pipe.unet = pipe.unet.to(device, dtype=dtype)  # 移回设备

        print("[model_loader] LoRA 权重已融合进 UNet。", flush=True)
    else:
        print(f"[model_loader] ⚠️ 未找到 LoRA 权重文件: {lora_weights}", flush=True)
        print("[model_loader] 将使用基座模型（无风格微调）生成。", flush=True)

    # ---------- 移到设备 + 优化 ----------
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()  # 省显存：把注意力计算切成小块，8GB 卡也能跑

    print(f"[model_loader] 管线就绪，设备: {device}, 精度: {dtype}", flush=True)
    print(f"[model_loader] 默认参数: steps=25, guidance_scale=7.5, size=512×512", flush=True)

    _pipeline = pipe
    return _pipeline


def get_pipeline() -> StableDiffusionPipeline | None:
    """获取已加载的管线（如果还没加载，会自动加载）。"""
    if _pipeline is None:
        return load_pipeline()
    return _pipeline
