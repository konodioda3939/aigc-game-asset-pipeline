"""FastAPI 推理服务：接收文字 prompt，返回 AI 生成的图片。

启动方式：
    cd d:\aigc-project\inference_server
    uvicorn main:app --host 127.0.0.1 --port 8000

测试方式：
    curl -X POST http://127.0.0.1:8000/generate \
      -H "Content-Type: application/json" \
      -d '{"prompt": "a game sword icon, fantasy style, masterpiece"}' \
      -o sword.png

接口说明：
    POST /generate   → 提交生成请求，返回图片文件
    GET  /health     → 检查服务是否就绪
"""
import os
import sys

# ==== 打印立即刷新 + 编码设置 ====
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ==== 必须在 import transformers/diffusers 之前设置 ====
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import io
import time
import json
from pathlib import Path
from datetime import datetime

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, FileResponse
from pydantic import BaseModel, Field

from model_loader import load_pipeline


# ===== 常量 =====
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# 默认负面提示词 — 屏蔽常见质量问题（畸形手指、低画质、水印等）
DEFAULT_NEGATIVE = (
    "lowres, bad anatomy, bad hands, text, error, extra digit, "
    "fewer digits, cropped, worst quality, low quality, normal quality, "
    "jpeg artifacts, signature, watermark, username, blurry"
)


# ===== 请求/响应模型 =====
class GenerateRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="正向提示词（英文），描述你想要生成的画面。例如: '1girl, solo, long hair, masterpiece'",
        min_length=1,
        max_length=1000,
    )
    negative_prompt: str = Field(
        default=DEFAULT_NEGATIVE,
        description="负面提示词，描述你不想要的元素。默认屏蔽低画质、畸形等。",
        max_length=1000,
    )
    steps: int = Field(
        default=25,
        description="推理步数。20~30 是推荐范围。越多越精细但越慢。",
        ge=10,
        le=100,
    )
    guidance_scale: float = Field(
        default=7.5,
        description="提示词引导强度。5~10 是推荐范围。越高越贴近 prompt，但过高会失真。",
        ge=1.0,
        le=20.0,
    )
    seed: int | None = Field(
        default=None,
        description="随机种子。留空则每次生成不同；填数字则相同 prompt+seed 生成相同图片。",
        ge=0,
    )


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    lora_loaded: bool
    uptime_seconds: float


# ===== FastAPI 应用 =====
app = FastAPI(
    title="AIGC LoRA 推理服务",
    description="用训练好的原神风格 LoRA 生成动漫图片",
    version="0.1.0",
)

# 记录启动时间（用于 /health 的 uptime 计算）
_start_time: float | None = None


# ===== 启动事件 =====
@app.on_event("startup")
async def startup():
    """服务启动时加载模型（只加载一次，之后所有请求复用）。"""
    global _start_time

    print("=" * 50, flush=True)
    print("  正在启动 AIGC LoRA 推理服务...", flush=True)
    print("=" * 50, flush=True)

    # 加载模型（model_loader 内部会自动检测 GPU/CPU、加载 LoRA）
    load_pipeline()

    _start_time = time.time()
    print(f"\n  服务已就绪 → http://127.0.0.1:8000", flush=True)
    print(f"  API 文档 → http://127.0.0.1:8000/docs", flush=True)
    print("=" * 50, flush=True)


# ===== 接口 =====
@app.post("/generate")
async def generate(req: GenerateRequest):
    """
    生成图片。

    传入 prompt 和可选参数，返回一张 PNG 图片。

    - **prompt**: 必须，英文描述。例如 "1girl, raiden shogun, purple eyes, masterpiece"
    - **negative_prompt**: 可选，不想出现的元素
    - **steps**: 可选，默认 25（20~30 推荐）
    - **guidance_scale**: 可选，默认 7.5（5~10 推荐）
    - **seed**: 可选，固定随机种子以复现相同结果
    """
    from model_loader import get_pipeline

    pipe = get_pipeline()
    if pipe is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成，请稍后重试")

    device = pipe.device

    # 处理随机种子：用户没传就用系统时间戳，保证每次不同
    actual_seed = req.seed if req.seed is not None else int(time.time() * 1000) % (2**31)

    print(f"\n[generate] prompt: {req.prompt[:80]}...", flush=True)
    print(f"[generate] steps={req.steps}, cfg={req.guidance_scale}, seed={actual_seed}", flush=True)

    # ---------- 推理 ----------
    try:
        generator = torch.Generator(device).manual_seed(actual_seed)

        with torch.no_grad():
            result = pipe(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                num_inference_steps=req.steps,
                guidance_scale=req.guidance_scale,
                generator=generator,
            )

        image: Image.Image = result.images[0]

    except torch.cuda.OutOfMemoryError:
        raise HTTPException(
            status_code=500,
            detail="GPU 显存不足。请尝试降低 steps 参数，或重启服务释放显存。",
        )
    except Exception as e:
        print(f"[generate] 错误: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"图片生成失败: {str(e)}")

    # ---------- 存档到本地 ----------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_prompt = req.prompt[:30].replace(" ", "_").replace(",", "").replace("/", "_")
    filename = f"{timestamp}_{safe_prompt}.png"
    save_path = OUTPUT_DIR / filename
    image.save(save_path)
    print(f"[generate] 已保存: {save_path}", flush=True)

    # ---------- 返回图片 ----------
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    return Response(
        content=img_bytes.getvalue(),
        media_type="image/png",
        headers={
            "X-Seed": str(actual_seed),
            "X-Filename": filename,
        },
    )


@app.get("/health")
async def health():
    """健康检查：返回服务是否就绪、加载了哪个模型、运行了多久。"""
    from model_loader import get_pipeline

    pipe = get_pipeline()
    lora_loaded = Path(r"D:\aigc-project\lora_output\adapter_model.safetensors").exists()

    uptime = (time.time() - _start_time) if _start_time else 0

    return HealthResponse(
        status="ready" if pipe is not None else "loading",
        model="Counterfeit-V2.5 + LoRA (原神风格)",
        device="cuda" if (pipe and str(pipe.device).startswith("cuda")) else "cpu",
        lora_loaded=lora_loaded,
        uptime_seconds=round(uptime, 1),
    )


@app.get("/")
async def root():
    """根路径重定向到 API 文档"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


# ===== 关闭事件 =====
@app.on_event("shutdown")
async def shutdown():
    """服务关闭时释放 GPU 显存。"""
    from model_loader import get_pipeline

    pipe = get_pipeline()
    if pipe is not None:
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("[shutdown] 服务已关闭，显存已释放。", flush=True)


# ===== 直接双击 main.py 或 python main.py 启动 =====
if __name__ == "__main__":
    import uvicorn

    print("=" * 50, flush=True)
    print("  AIGC LoRA 推理服务 — 启动中...", flush=True)
    print("=" * 50, flush=True)
    print("", flush=True)
    print("  也可以双击 start.bat 启动（更简单）", flush=True)
    print("", flush=True)

    uvicorn.run(app, host="127.0.0.1", port=8000)
