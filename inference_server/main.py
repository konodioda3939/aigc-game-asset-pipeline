"""
FastAPI 推理服务：接收文字 prompt + 可选参考图（ControlNet），返回 AI 生成的图片。

启动方式：
    cd d:\aigc-project\inference_server
    uvicorn main:app --host 127.0.0.1 --port 8000

接口说明：
    POST /generate            → txt2img（纯文本生图，向后兼容）
    POST /generate-controlled  → ControlNet 可控生成（图片 + prompt）
    GET  /health              → 检查服务是否就绪
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
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import Response, RedirectResponse
from pydantic import BaseModel, Field

from model_loader import (
    load_pipeline,
    get_pipeline,
    get_controlnet_pipeline,
    get_available_controlnet_modes,
    get_triposr_model,
)


# ===== 常量 =====
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULT_NEGATIVE = (
    "lowres, bad anatomy, bad hands, text, error, extra digit, "
    "fewer digits, cropped, worst quality, low quality, normal quality, "
    "jpeg artifacts, signature, watermark, username, blurry"
)


# ===== 请求/响应模型 =====

class GenerateRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="正向提示词（英文）",
        min_length=1,
        max_length=1000,
    )
    negative_prompt: str = Field(
        default=DEFAULT_NEGATIVE,
        description="负面提示词",
        max_length=1000,
    )
    steps: int = Field(default=25, description="推理步数", ge=10, le=100)
    guidance_scale: float = Field(default=7.5, description="引导强度", ge=1.0, le=20.0)
    seed: int | None = Field(default=None, description="随机种子", ge=0)


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    lora_loaded: bool
    controlnet_modes: list
    triposr_loaded: bool
    uptime_seconds: float


# ===== FastAPI 应用 =====
app = FastAPI(
    title="AIGC LoRA 推理服务",
    description="用训练好的原神风格 LoRA + ControlNet 生成动漫图片",
    version="0.2.0",
)

_start_time: float | None = None


# ===== 预处理函数 =====

# SD 1.5 原生分辨率 512×512，超过 768 后显存和速度急剧恶化
# 8GB 显存参考：512=~2s/步, 768=~5s/步, 1024=~26s/步（可能触发系统内存交换）
DEFAULT_CONTROLNET_MAX_SIZE = 768
# VAE 要求尺寸为 8 的倍数
_LATENT_ALIGN = 8


def _resize_for_controlnet(image: Image.Image, max_size: int = DEFAULT_CONTROLNET_MAX_SIZE) -> Image.Image:
    """
    将输入图缩放到适合 SD 1.5 + ControlNet 处理的尺寸。

    保持宽高比，长边不超过 max_size，尺寸对齐到 8 的倍数（VAE 要求）。
    1024×1024 → 768×768（速度提升约 5 倍，显存安全）。
    """
    w, h = image.size
    longest = max(w, h)
    if longest <= max_size:
        return image  # 已经够小，不需要缩放

    scale = max_size / longest
    new_w = int(w * scale)
    new_h = int(h * scale)
    # 对齐到 8 的倍数（VAE 编码器要求）
    new_w = (new_w // _LATENT_ALIGN) * _LATENT_ALIGN
    new_h = (new_h // _LATENT_ALIGN) * _LATENT_ALIGN
    print(f"[generate-controlled] 自动缩放: {w}×{h} → {new_w}×{new_h} "
          f"（SD 1.5 原生 512×512，过大图片会极慢）", flush=True)
    return image.resize((new_w, new_h), Image.LANCZOS)

def preprocess_canny(image: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    """
    Canny 边缘检测：提取图片的结构轮廓，作为 ControlNet 的骨架输入。

    大白话：把图片变成「线稿」，AI 照着这个线稿上色。
    """
    try:
        import cv2
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="缺少 opencv-python 依赖。请运行: pip install opencv-python"
        )

    # 转灰度 → Canny 边缘检测
    img_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)

    return Image.fromarray(edges)


def preprocess_scribble(image: Image.Image) -> Image.Image:
    """
    涂鸦/草图预处理：生成类似手绘轮廓的预处理图。

    策略：用 Canny 低阈值多抓边缘 + 高斯模糊 → 模拟手绘感。
    如果 controlnet_aux 可用，则使用 HED 检测器（效果更好）。
    """
    try:
        from controlnet_aux import HEDdetector
        hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
        result = hed(image)
        if isinstance(result, Image.Image):
            return result
        return image
    except Exception:
        pass  # 回退到 OpenCV 方案

    import cv2
    img_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # 低阈值抓更多边缘 → 模拟草图
    edges = cv2.Canny(gray, 50, 150)
    # 轻微模糊，模拟手绘
    blurred = cv2.GaussianBlur(edges, (3, 3), 0)

    return Image.fromarray(blurred)


def preprocess_depth(image: Image.Image) -> Image.Image:
    """
    深度图预处理：提取空间深度信息，让 AI 保持物体的空间关系。
    适合 3D 渲染图/照片 → 保持前后遮挡关系。
    """
    try:
        from controlnet_aux import MidasDetector
        midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
        result = midas(image)
        if isinstance(result, Image.Image):
            return result
        return image
    except Exception:
        pass

    import cv2
    img_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    # 用高斯模糊模拟简单深度（远处模糊 = 近处清晰）
    depth = cv2.GaussianBlur(gray, (5, 5), 0)
    return Image.fromarray(depth)


PREPROCESSORS = {
    "canny": preprocess_canny,
    "scribble": preprocess_scribble,
    "depth": preprocess_depth,
}


# ===== 启动事件 =====

@app.on_event("startup")
async def startup():
    """服务启动时加载模型（只加载一次，之后所有请求复用）。"""
    global _start_time

    print("=" * 50, flush=True)
    print("  正在启动 AIGC 推理服务 (txt2img + ControlNet)...", flush=True)
    print("=" * 50, flush=True)

    # 加载 txt2img 管线（基座模型 + LoRA）
    load_pipeline()

    _start_time = time.time()

    available = get_available_controlnet_modes()
    print(f"\n  服务已就绪 → http://127.0.0.1:8000", flush=True)
    print(f"  API 文档 → http://127.0.0.1:8000/docs", flush=True)
    print(f"  ControlNet 可用: {available}", flush=True)
    print(f"  TripoSR 3D 生成: 首次使用自动下载（~1.68GB）", flush=True)
    print("=" * 50, flush=True)


# ===== txt2img 接口（向后兼容） =====

@app.post("/generate")
async def generate(req: GenerateRequest):
    """
    纯文本生成图片（原有接口，不受影响）。

    - **prompt**: 必须，英文描述。例如 "1girl, raiden shogun, purple eyes, masterpiece"
    - **steps**: 可选，默认 25
    - **guidance_scale**: 可选，默认 7.5
    - **seed**: 可选，固定相同结果
    """
    from model_loader import get_pipeline

    pipe = get_pipeline()
    if pipe is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成，请稍后重试")

    actual_seed = req.seed if req.seed is not None else int(time.time() * 1000) % (2**31)
    print(f"\n[generate] prompt: {req.prompt[:80]}...", flush=True)
    print(f"[generate] steps={req.steps}, cfg={req.guidance_scale}, seed={actual_seed}", flush=True)

    try:
        generator = torch.Generator(pipe.device).manual_seed(actual_seed)
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
        raise HTTPException(status_code=500, detail="GPU 显存不足。请降低 steps 参数后重试。")
    except Exception as e:
        print(f"[generate] 错误: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"图片生成失败: {str(e)}")

    # 存档 + 返回
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_prompt = req.prompt[:30].replace(" ", "_").replace(",", "").replace("/", "_")
    filename = f"{timestamp}_{safe_prompt}.png"
    image.save(OUTPUT_DIR / filename)
    print(f"[generate] 已保存: {OUTPUT_DIR / filename}", flush=True)

    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    return Response(
        content=img_bytes.getvalue(),
        media_type="image/png",
        headers={"X-Seed": str(actual_seed), "X-Filename": filename},
    )


# ===== ControlNet 可控生成接口 =====

@app.post("/generate-controlled")
async def generate_controlled(
    image: UploadFile = File(..., description="参考图（草图/线稿/轮廓），PNG 或 JPG"),
    prompt: str = Form(..., description="正向提示词（英文）"),
    control_mode: str = Form("canny", description="控制方式: canny 或 scribble"),
    steps: int = Form(25, ge=10, le=100),
    guidance_scale: float = Form(7.5, ge=1.0, le=20.0),
    control_strength: float = Form(0.8, ge=0.1, le=2.0, description="ControlNet 控制力度，越大越严格贴合参考图"),
    max_size: int = Form(DEFAULT_CONTROLNET_MAX_SIZE, ge=512, le=1024, description="输入图最大边长（SD 1.5 建议 768，过大显存爆炸）"),
    canny_low: int = Form(100, ge=0, le=255, description="Canny 低阈值"),
    canny_high: int = Form(200, ge=0, le=255, description="Canny 高阈值"),
    negative_prompt: str = Form(DEFAULT_NEGATIVE),
    seed: int | None = Form(None),
):
    """
    可控生成：上传一张参考图，AI 保持其结构骨架，按 prompt 描述填充内容。

    - **image**: 必须，参考图文件
    - **prompt**: 必须，英文描述
    - **control_mode**: canny（线稿精修）或 scribble（草图生成）
    - **control_strength**: 控制力度，0.1=松（更多创意），2.0=紧（严格贴合）
    - **max_size**: 输入图最大边长（默认 768）。SD 1.5 原生 512×512，
      过大会导致极慢甚至显存不足，服务会自动缩放
    """
    # ---- 1. 读取参考图 ----
    try:
        ref_bytes = await image.read()
        ref_image = Image.open(io.BytesIO(ref_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="无法读取参考图，请确认为有效 PNG/JPG 文件。")

    print(f"\n[generate-controlled] prompt: {prompt[:80]}...", flush=True)
    print(f"[generate-controlled] mode={control_mode}, ref_size={ref_image.size}", flush=True)

    # ---- 1.5 缩放输入图（SD 1.5 原生 512×512，过大图片极慢且可能 OOM）----
    ref_image = _resize_for_controlnet(ref_image, max_size)

    # ---- 2. 预处理（图片 → 结构骨架） ----
    if control_mode not in PREPROCESSORS:
        available = list(PREPROCESSORS.keys())
        raise HTTPException(status_code=400, detail=f"不支持 mode='{control_mode}'。可用: {available}")

    preprocessor = PREPROCESSORS[control_mode]
    kwargs = {}
    if control_mode == "canny":
        kwargs = {"low": canny_low, "high": canny_high}

    try:
        control_image = preprocessor(ref_image, **kwargs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图片预处理失败: {str(e)}")

    print(f"[generate-controlled] 预处理完成, control_image_size={control_image.size}", flush=True)

    # ---- 3. 获取 ControlNet 管线 ----
    try:
        pipe = get_controlnet_pipeline(control_mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[generate-controlled] ControlNet 加载失败: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"ControlNet 模型加载失败: {str(e)}")

    # ---- 4. 推理 ----
    actual_seed = seed if seed is not None else int(time.time() * 1000) % (2**31)
    print(f"[generate-controlled] steps={steps}, cfg={guidance_scale}, "
          f"control_strength={control_strength}, seed={actual_seed}", flush=True)

    try:
        generator = torch.Generator(pipe.device).manual_seed(actual_seed)
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=control_image,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=control_strength,
                generator=generator,
            )
        output_image: Image.Image = result.images[0]
    except torch.cuda.OutOfMemoryError:
        raise HTTPException(status_code=500, detail="GPU 显存不足。请降低 steps 参数后重试。")
    except Exception as e:
        print(f"[generate-controlled] 推理失败: {traceback.format_exc()}", flush=True)
        raise HTTPException(status_code=500, detail=f"图片生成失败: {str(e)}")

    # ---- 5. 存档 ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_prompt = prompt[:20].replace(" ", "_").replace(",", "").replace("/", "_")

    # 存原始参考图
    ref_filename = f"{timestamp}_{control_mode}_ref.png"
    ref_image.save(OUTPUT_DIR / ref_filename)

    # 存预处理图（方便调试）
    preproc_filename = f"{timestamp}_{control_mode}_preproc.png"
    control_image.save(OUTPUT_DIR / preproc_filename)

    # 存生成图
    out_filename = f"{timestamp}_{control_mode}_{safe_prompt}.png"
    output_image.save(OUTPUT_DIR / out_filename)

    print(f"[generate-controlled] 已保存: 参考图={ref_filename}, "
          f"预处理={preproc_filename}, 生成={out_filename}", flush=True)

    # ---- 6. 返回 ----
    img_bytes = io.BytesIO()
    output_image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    return Response(
        content=img_bytes.getvalue(),
        media_type="image/png",
        headers={
            "X-Seed": str(actual_seed),
            "X-Filename": out_filename,
            "X-Preprocess": preproc_filename,
        },
    )


# ===== 3D 模型生成接口 =====

@app.post("/generate-3d")
async def generate_3d(
    image: UploadFile = File(..., description="角色/物体参考图（PNG 或 JPG）"),
    prompt: str = Form("", description="预留参数，TripoSR 不使用 prompt"),
    output_format: str = Form("glb", description="输出格式: glb 或 obj"),
    resolution: int = Form(256, ge=128, le=512, description="Mesh 精度（128=快, 256=标准, 512=高精度）"),
    seed: int | None = Form(None, description="TripoSR 是确定性模型，seed 作用有限"),
):
    """
    图片转 3D 模型：上传角色设定图，AI 自动生成带贴图的 3D 模型。

    - **image**: 必须，输入的角色/物体图片
    - **output_format**: glb（推荐，贴图内嵌，Unity 原生支持）或 obj
    - **resolution**: 128=快速预览, 256=标准质量, 512=最高精度（更慢）
    """
    from tsr.utils import remove_background, resize_foreground

    # ---- 1. 读取参考图 ----
    try:
        ref_bytes = await image.read()
        ref_image = Image.open(io.BytesIO(ref_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="无法读取参考图，请确认为有效 PNG/JPG 文件。")

    original_size = ref_image.size
    print(f"\n[generate-3d] image_size={original_size}, format={output_format}, "
          f"resolution={resolution}", flush=True)

    # ---- 2. 预处理：去背景 + 调整 ----
    try:
        print("[generate-3d] 正在去除背景...", flush=True)
        ref_image = remove_background(ref_image)
        print(f"[generate-3d] 去背景完成, mode={ref_image.mode}, size={ref_image.size}", flush=True)

        # 保存去背景后的中间结果（方便排查是抠图问题还是模型问题）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_name = "model"
        rembg_path = OUTPUT_DIR / f"{timestamp}_3d_rembg.png"
        ref_image.save(rembg_path)
        print(f"[generate-3d] 已保存去背景结果: {rembg_path}", flush=True)

        ref_image = resize_foreground(ref_image, 0.85)  # 需要 RGBA 做裁剪
        ref_image = ref_image.convert("RGB")  # 裁剪完后转 RGB，TripoSR 需要
        print(f"[generate-3d] 预处理完成, size={ref_image.size}", flush=True)

        # 保存最终预处理图（喂给 TripoSR 的样子）
        preproc_path = OUTPUT_DIR / f"{timestamp}_3d_preproc.png"
        ref_image.save(preproc_path)
        print(f"[generate-3d] 已保存预处理结果: {preproc_path}", flush=True)
    except Exception as e:
        print(f"[generate-3d] 预处理失败: {traceback.format_exc()}", flush=True)
        raise HTTPException(status_code=500, detail=f"图片预处理失败: {str(e)}")

    # ---- 3. 获取 TripoSR 模型 ----
    try:
        model = get_triposr_model()
    except Exception as e:
        print(f"[generate-3d] TripoSR 加载失败: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"TripoSR 模型加载失败: {str(e)}")

    # ---- 4. 设置精度 + 分块大小（节省显存） ----
    model.set_marching_cubes_resolution(resolution)
    # 渲染器分块：每次只处理 4096 个点，避免一次性撑爆显存
    # 默认 chunk_size=0（不分块），256³ 需要 16M 点同时处理 → OOM
    model.renderer.set_chunk_size(4096)
    print(f"[generate-3d] 渲染器分块大小: 4096", flush=True)

    # ---- 5. 推理 ----
    device = str(model.device) if hasattr(model, 'device') else "cuda"
    actual_seed = seed if seed is not None else int(time.time() * 1000) % (2**31)
    print(f"[generate-3d] 正在生成 3D 模型（这可能需要 5-30 秒）...", flush=True)

    try:
        if seed is not None:
            torch.manual_seed(actual_seed)

        with torch.no_grad():
            scene_codes = model(ref_image, device=device)
    except torch.cuda.OutOfMemoryError:
        fallback_res = max(128, resolution // 2)
        print(f"[generate-3d] 显存不足，降级到 resolution={fallback_res} 重试...", flush=True)
        torch.cuda.empty_cache()
        model.set_marching_cubes_resolution(fallback_res)
        with torch.no_grad():
            scene_codes = model(ref_image, device=device)
        resolution = fallback_res
    except Exception as e:
        print(f"[generate-3d] 推理失败: {traceback.format_exc()}", flush=True)
        raise HTTPException(status_code=500, detail=f"3D 模型生成失败: {str(e)}")

    # ---- 6. 提取 mesh ----
    try:
        print("[generate-3d] 正在提取 3D mesh...", flush=True)
        mesh = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=resolution)[0]
        print(f"[generate-3d] mesh: {len(mesh.vertices)} 顶点, {len(mesh.faces)} 面", flush=True)
    except torch.cuda.OutOfMemoryError:
        fallback_res = max(128, resolution // 2)
        print(f"[generate-3d] 提取 mesh 时显存不足，降级到 resolution={fallback_res}...", flush=True)
        torch.cuda.empty_cache()
        model.set_marching_cubes_resolution(fallback_res)
        mesh = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=fallback_res)[0]
        print(f"[generate-3d] mesh: {len(mesh.vertices)} 顶点, {len(mesh.faces)} 面", flush=True)
    except Exception as e:
        print(f"[generate-3d] Mesh 提取失败: {traceback.format_exc()}", flush=True)
        raise HTTPException(status_code=500, detail=f"Mesh 提取失败: {str(e)}")

    # ---- 8. 导出 ----
    try:
        model_bytes = io.BytesIO()
        if output_format == "obj":
            mesh.export(model_bytes, file_type="obj")
            media_type = "model/obj"
            ext = "obj"
        else:
            mesh.export(model_bytes, file_type="glb")
            media_type = "model/gltf-binary"
            ext = "glb"
        model_bytes.seek(0)
        model_data = model_bytes.getvalue()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"3D 模型导出失败: {str(e)}")

    # ---- 8. 存档 ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = "model"  # TripoSR 不使用 prompt，用固定名
    filename = f"{timestamp}_3d_{safe_name}.{ext}"
    archive_path = OUTPUT_DIR / filename
    archive_path.write_bytes(model_data)
    print(f"[generate-3d] 已保存: {archive_path} "
          f"({len(model_data)/1024:.0f} KB, "
          f"{len(mesh.vertices)} verts, {len(mesh.faces)} faces)", flush=True)

    # ---- 9. 清理显存 ----
    del scene_codes
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- 10. 返回 ----
    return Response(
        content=model_data,
        media_type=media_type,
        headers={
            "X-Seed": str(actual_seed),
            "X-Filename": filename,
            "X-Format": output_format,
            "X-Vertices": str(len(mesh.vertices)),
            "X-Faces": str(len(mesh.faces)),
        },
    )


# ===== 健康检查 =====

@app.get("/health")
async def health():
    """
    健康检查：返回服务状态、模型信息、ControlNet 可用模式。
    """
    from model_loader import (
        get_pipeline, get_loaded_controlnet_modes, get_available_controlnet_modes,
    )

    pipe = get_pipeline()
    lora_loaded = Path(r"D:\aigc-project\lora_output\adapter_model.safetensors").exists()
    uptime = (time.time() - _start_time) if _start_time else 0

    # 检查 TripoSR 状态（不触发加载）
    from model_loader import _triposr_model
    triposr_loaded = _triposr_model is not None

    return HealthResponse(
        status="ready" if pipe is not None else "loading",
        model="Counterfeit-V2.5 + LoRA (原神风格) + ControlNet + TripoSR",
        device="cuda" if (pipe and str(pipe.device).startswith("cuda")) else "cpu",
        lora_loaded=lora_loaded,
        controlnet_modes=get_loaded_controlnet_modes() or get_available_controlnet_modes(),
        triposr_loaded=triposr_loaded,
        uptime_seconds=round(uptime, 1),
    )


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


# ===== 关闭事件 =====

@app.on_event("shutdown")
async def shutdown():
    """服务关闭时释放 GPU 显存。"""
    from model_loader import get_pipeline, _triposr_model

    pipe = get_pipeline()
    if pipe is not None:
        del pipe

    # 清理 TripoSR 模型
    if _triposr_model is not None:
        del _triposr_model
        import model_loader
        model_loader._triposr_model = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[shutdown] 服务已关闭，显存已释放。", flush=True)


# ===== 直接双击 main.py 或 python main.py 启动 =====
if __name__ == "__main__":
    import uvicorn

    print("=" * 50, flush=True)
    print("  AIGC 推理服务 (txt2img + ControlNet) — 启动中...", flush=True)
    print("=" * 50, flush=True)
    print("", flush=True)
    print("  也可以双击 start.bat 启动（更简单）", flush=True)
    print("", flush=True)

    uvicorn.run(app, host="127.0.0.1", port=8000)
