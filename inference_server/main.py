"""
FastAPI 推理服务：接收文字 prompt + 可选参考图（ControlNet），返回 AI 生成的图片/3D模型/PBR材质。

启动方式：
    cd d:\aigc-project\inference_server
    uvicorn main:app --host 127.0.0.1 --port 8000

接口说明：
    POST /generate             → txt2img（纯文本生图，向后兼容）
    POST /generate-controlled  → ControlNet 可控生成（图片 + prompt）
    POST /generate-3d          → TripoSR 图片转 3D 模型
    POST /generate-pbr         → StableMaterials PBR 材质生成
    GET  /health               → 检查服务是否就绪
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
import zipfile
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import Response, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from model_loader import (
    load_pipeline,
    get_pipeline,
    get_controlnet_pipeline,
    get_available_controlnet_modes,
    get_triposr_model,
    get_pbr_pipeline,
    _restore_sd_pipeline,
    _pbr_pipeline,
)

# 工作流模块
from workflows.character_concept import CharacterConceptWorkflow
from workflows.asset_generator import AssetGeneratorWorkflow
from workflows.model_3d import Model3DWorkflow
from workflows.pbr_material import PBRMaterialWorkflow
from prompts.engine import get_prompt_engine


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
    pbr_loaded: bool = False
    uptime_seconds: float


# ===== FastAPI 应用 =====
app = FastAPI(
    title="AIGC 推理服务",
    description="LoRA/ControlNet/TripoSR/StableMaterials — 游戏资产全自动生成",
    version="0.3.0",
)

_start_time: float | None = None

# ===== 工作流注册表 =====
_prompt_engine = get_prompt_engine()
_workflow_registry: dict[str, object] = {}  # workflow_id → workflow instance


def _init_workflows():
    """初始化所有工作流实例（懒初始化，首次访问时调用）。"""
    global _workflow_registry
    if _workflow_registry:
        return
    eng = _prompt_engine
    _workflow_registry = {
        "character_concept": CharacterConceptWorkflow(eng),
        "asset_generator": AssetGeneratorWorkflow(eng),
        "model_3d": Model3DWorkflow(eng),
        "pbr_material": PBRMaterialWorkflow(eng),
    }
    print(f"[workflows] 已注册 {len(_workflow_registry)} 个工作流", flush=True)


# ===== 挂载 Web 演示界面静态文件 =====
_WEB_UI_DIR = Path(__file__).parent / "web_ui"
if _WEB_UI_DIR.exists():
    app.mount("/workflow-ui", StaticFiles(directory=str(_WEB_UI_DIR), html=True), name="workflow_ui")
    print(f"[workflows] Web UI 已挂载: http://127.0.0.1:8000/workflow-ui/", flush=True)


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

# ===== 422 错误日志中间件（排查表单验证问题） =====
from fastapi.exceptions import RequestValidationError
from fastapi import Request

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """422 错误时打印详细日志，帮助定位是哪个字段验证失败。"""
    body = None
    try:
        body = await request.form()
        print(f"\n[422] 表单验证失败 — {request.method} {request.url.path}", flush=True)
        print(f"[422] 收到的字段: {list(body.keys())}", flush=True)
        for key, value in body.items():
            val_str = str(value)
            if len(val_str) > 200:
                val_str = val_str[:200] + f"... ({len(val_str)} chars)"
            print(f"  {key} = {val_str}", flush=True)
    except Exception:
        pass

    print(f"[422] 验证错误:", flush=True)
    for error in exc.errors():
        print(f"  - {'.'.join(str(loc) for loc in error['loc'])}: {error['msg']}", flush=True)

    # 返回标准 FastAPI 422 响应
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


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
    print(f"  工作流 Web UI → http://127.0.0.1:8000/workflow-ui/", flush=True)
    print(f"  ControlNet 可用: {available}", flush=True)
    print(f"  TripoSR 3D 生成: 首次使用自动下载（~1.68GB）", flush=True)
    print(f"  StableMaterials PBR 材质: 首次使用自动下载（~2-3GB）", flush=True)
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

    # ---- 2.5 确保预处理图尺寸与缩放后一致（HED/MiDaS 等可能改变尺寸）----
    if control_image.size != ref_image.size:
        print(f"[generate-controlled] 预处理图尺寸不一致，对齐到 {ref_image.size}", flush=True)
        control_image = control_image.resize(ref_image.size, Image.LANCZOS)

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


# ===== PBR 材质生成接口 =====

@app.post("/generate-pbr")
async def generate_pbr(
    prompt: str = Form(..., description="材质描述（英文），如 'rough stone wall'"),
    tileable: bool = Form(True, description="是否生成无缝平铺贴图"),
    steps: int = Form(25, ge=5, le=50),
    guidance_scale: float = Form(10.0, ge=1.0, le=20.0),
    seed: int | None = Form(None),
):
    """
    PBR 材质生成：输入文字描述，AI 自动生成完整的 PBR 纹理贴图集。

    返回 ZIP 压缩包，包含：
      - basecolor.png          — 基础颜色贴图（sRGB）
      - normal.png             — 法线方向贴图
      - metallic_smoothness.png — R=金属度, A=光滑度（已打包，Unity Standard Shader 直接可用）
      - height.png             — 高度/置换贴图
      - roughness_raw.png      — 原始粗糙度贴图（调试用）
      - preview.png            — 256px 预览缩略图

    纹理尺寸：512×512（SD 原生分辨率）
    推理时间：LCM 4 步约 5-10 秒（首次需下载模型 ~2-3GB）
    """
    from model_loader import get_pbr_pipeline, _restore_sd_pipeline

    print(f"\n[generate-pbr] prompt: {prompt[:80]}...", flush=True)
    print(f"[generate-pbr] tileable={tileable}, steps={steps}, "
          f"cfg={guidance_scale}", flush=True)

    # 获取 PBR 管线（首次自动下载权重，同时卸载 SD 管线）
    try:
        pipe = get_pbr_pipeline()
    except Exception as e:
        print(f"[generate-pbr] PBR 管线加载失败: {e}", flush=True)
        _restore_sd_pipeline()
        raise HTTPException(status_code=500, detail=f"PBR 模型加载失败: {str(e)}")

    # 推理
    actual_seed = seed if seed is not None else int(time.time() * 1000) % (2**31)

    try:
        generator_obj = torch.Generator(device=pipe.device).manual_seed(actual_seed)
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator_obj,
                tileable=tileable,
            )

        # 解包 5 张 PBR 贴图（result.images[0] 是 StableMaterialsMaterial 对象）
        material = result.images[0]
        basecolor: Image.Image = material.basecolor
        normal: Image.Image = material.normal
        height: Image.Image = material.height
        roughness: Image.Image = material.roughness
        metallic: Image.Image = material.metallic
    except AttributeError as e:
        _restore_sd_pipeline()
        print(f"[generate-pbr] 结果解包失败（模型返回格式不符预期）: {e}", flush=True)
        raise HTTPException(
            status_code=500,
            detail=f"PBR 贴图解包失败: {e}。请检查 StableMaterials 模型版本是否兼容。"
        )
    except torch.cuda.OutOfMemoryError:
        _restore_sd_pipeline()
        raise HTTPException(status_code=500, detail="GPU 显存不足，无法生成 PBR 材质。请重启服务后重试。")
    except Exception as e:
        _restore_sd_pipeline()
        print(f"[generate-pbr] 推理失败: {traceback.format_exc()}", flush=True)
        raise HTTPException(status_code=500, detail=f"PBR 材质生成失败: {str(e)}")

    # 存档 + 纹理打包
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_prompt = prompt[:20].replace(" ", "_").replace(",", "").replace("/", "_")

    def _save_map(img: Image.Image, suffix: str) -> Path:
        path = OUTPUT_DIR / f"{timestamp}_pbr_{safe_prompt}_{suffix}.png"
        img.save(path)
        return path

    _save_map(basecolor, "basecolor")
    _save_map(normal, "normal")
    _save_map(height, "height")
    _save_map(roughness, "roughness_raw")
    _save_map(metallic, "metallic_raw")

    # 打包 Metallic(R) + Smoothness(1-Roughness, A) → Unity _MetallicGlossMap
    metallic_arr = np.array(metallic.convert("L"))
    roughness_arr = np.array(roughness.convert("L"))
    smoothness_arr = (255 - roughness_arr).astype(np.uint8)

    h, w = metallic_arr.shape
    packed = np.zeros((h, w, 4), dtype=np.uint8)
    packed[:, :, 0] = metallic_arr    # R = Metallic
    packed[:, :, 3] = smoothness_arr  # A = Smoothness (1 - Roughness)

    packed_img = Image.fromarray(packed, mode="RGBA")
    _save_map(packed_img, "metallic_smoothness")

    # 生成预览缩略图（Basecolor 缩小到 256x256）
    preview = basecolor.copy()
    preview.thumbnail((256, 256), Image.LANCZOS)
    _save_map(preview, "preview")

    # 打包 ZIP
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for suffix in [
            "basecolor", "normal", "height",
            "roughness_raw", "metallic_raw",
            "metallic_smoothness", "preview",
        ]:
            file_path = OUTPUT_DIR / f"{timestamp}_pbr_{safe_prompt}_{suffix}.png"
            if file_path.exists():
                zf.write(file_path, f"{suffix}.png")

    zip_buffer.seek(0)
    zip_data = zip_buffer.getvalue()

    # 恢复 SD 管线到 GPU
    _restore_sd_pipeline()

    print(f"[generate-pbr] PBR 材质生成完成（seed={actual_seed}, "
          f"ZIP={len(zip_data)/1024:.0f} KB）。", flush=True)

    return Response(
        content=zip_data,
        media_type="application/zip",
        headers={
            "X-Seed": str(actual_seed),
            "X-Filename": f"{timestamp}_pbr_{safe_prompt}",
            "X-Tileable": str(tileable).lower(),
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

    # 检查 TripoSR / PBR 状态（不触发加载）
    from model_loader import _triposr_model
    triposr_loaded = _triposr_model is not None
    from model_loader import _pbr_pipeline
    pbr_loaded = _pbr_pipeline is not None

    return HealthResponse(
        status="ready" if pipe is not None else "loading",
        model="Counterfeit-V2.5 + LoRA (原神风格) + ControlNet + TripoSR + StableMaterials",
        device="cuda" if (pipe and str(pipe.device).startswith("cuda")) else "cpu",
        lora_loaded=lora_loaded,
        controlnet_modes=get_loaded_controlnet_modes() or get_available_controlnet_modes(),
        triposr_loaded=triposr_loaded,
        pbr_loaded=pbr_loaded,
        uptime_seconds=round(uptime, 1),
    )


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


# ===== 工作流 API =====

@app.get("/workflows")
async def list_workflows():
    """
    列出所有可用的游戏美术工作流及其参数说明。

    返回每个工作流的 ID、名称、描述、输入参数 schema。
    Web UI 和 Unity 插件用这个接口动态生成表单。
    """
    _init_workflows()
    workflows = _prompt_engine.list_workflows()
    return {
        "workflows": workflows,
        "total": len(workflows),
    }


@app.post("/workflows/run")
async def run_workflow(
    workflow: str = Form(..., description="工作流 ID: character_concept / prop_icon / scene_mood / ui_elements"),
    prompt: str = Form("", description="描述文字（英文，model_3d 可选）"),
    seed: int | None = Form(None),
    steps: int = Form(25, ge=10, le=100),
    guidance_scale: float = Form(7.5, ge=1.0, le=20.0),
    style_suffix: str = Form("", description="额外风格关键词"),
    reference_image: UploadFile | None = File(None, description="参考图（可选，道具图标/UI元素）"),
    mood: str = Form("", description="氛围选择（asset_generator 场景风格: magical/sunset/night/stormy/peaceful）"),
    mode: str = Form("", description="生成模式（角色概念图: turnaround / individual）"),
    style: str = Form("", description="素材风格（asset_generator: icon / scene / ui）"),
    control_mode: str = Form("", description="ControlNet 模式（asset_generator: canny / scribble / depth）"),
    control_strength: float = Form(0.85, ge=0.1, le=2.0, description="ControlNet 控制力度"),
    resolution: int = Form(256, ge=128, le=512, description="3D Mesh 精度（model_3d: 128/256/512）"),
    output_format: str = Form("glb", description="3D 输出格式（model_3d: glb/obj）"),
    tileable: bool = Form(True, description="无缝平铺（pbr_material）"),
    wide: bool = Form(True, description="宽幅构图（场景氛围图）"),
    stitch_grid: bool = Form(True, description="拼接画板（角色概念图）"),
    elements: str = Form("", description="UI 元素列表 JSON 数组（ui_elements）"),
    variants: int = Form(1, ge=1, le=4, description="生成变体数量（场景氛围图）"),
    remove_bg: bool = Form(True, description="去背景（道具图标）"),
):
    """
    执行一个游戏美术工作流。

    - **workflow**: 必选，工作流 ID
    - **prompt**: 必选，描述文字
    - **seed**: 可选，随机种子
    - **steps**: 步数，默认 25
    - **guidance_scale**: 引导强度，默认 7.5

    工作流特定参数通过 params 字段传递：
    - character_concept: stitch_grid
    - prop_icon: reference_image, remove_bg
    - scene_mood: mood, wide, variants
    - ui_elements: reference_image, elements
    """
    import json

    _init_workflows()

    if workflow not in _workflow_registry:
        available = list(_workflow_registry.keys())
        raise HTTPException(
            status_code=400,
            detail=f"未知工作流: '{workflow}'。可用: {available}"
        )

    # 读取可选的参考图
    ref_bytes = None
    if reference_image is not None:
        try:
            ref_bytes = await reference_image.read()
        except Exception:
            raise HTTPException(status_code=400, detail="无法读取参考图。")

    # 解析 elements JSON
    elements_list = None
    if elements:
        try:
            elements_list = json.loads(elements)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="elements 参数格式错误，应为 JSON 数组。")

    # 构建参数
    params = {
        "prompt": prompt.strip(),
        "seed": seed,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "style_suffix": style_suffix.strip() or "",
        "reference_image": ref_bytes,
        "mood": mood.strip() or "",
        "mode": mode.strip() or "",
        "style": style.strip() or "",
        "control_mode": control_mode.strip() or "",
        "control_strength": control_strength,
        "resolution": resolution,
        "output_format": output_format.strip() or "glb",
        "tileable": tileable,
        "wide": wide,
        "stitch_grid": stitch_grid,
        "remove_bg": remove_bg,
        "variants": variants,
    }
    if elements_list is not None:
        params["elements"] = elements_list

    # 执行工作流
    print(f"\n[workflows/run] workflow={workflow}, prompt={prompt[:80]}...", flush=True)

    try:
        wf = _workflow_registry[workflow]
        result = wf.generate(params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except torch.cuda.OutOfMemoryError:
        raise HTTPException(status_code=500, detail="GPU 显存不足，请降低 steps 后重试。")
    except Exception as e:
        import traceback
        print(f"[workflows/run] 错误: {traceback.format_exc()}", flush=True)
        raise HTTPException(status_code=500, detail=f"工作流执行失败: {str(e)}")

    metadata = result.get("metadata", {})
    output_format = result.get("format", "png")

    if output_format == "zip":
        # 返回 ZIP（PBR 材质 / UI 元素）
        zip_data = result.get("zip_data", b"")
        return Response(
            content=zip_data,
            media_type="application/zip",
            headers={
                "X-Seed": str(metadata.get("seed", "")),
                "X-Workflow": workflow,
                "X-Element-Count": str(metadata.get("element_count", 0)),
                "X-Tileable": str(metadata.get("tileable", "")),
            },
        )
    elif output_format in ("glb", "obj"):
        # 返回 3D 模型二进制（GLB 或 OBJ）
        model_data = result.get("model_data", b"")
        if not model_data:
            raise HTTPException(status_code=500, detail="3D 模型生成失败，未产生模型数据。")

        media_type = result.get("media_type", "model/gltf-binary")
        ext = output_format
        return Response(
            content=model_data,
            media_type=media_type,
            headers={
                "X-Seed": str(metadata.get("seed", "")),
                "X-Workflow": workflow,
                "X-Vertices": str(metadata.get("vertices", 0)),
                "X-Faces": str(metadata.get("faces", 0)),
                "X-Format": ext,
                "X-Filename": f"model_{metadata.get('seed', '')}.{ext}",
                "Content-Disposition": (
                    f"attachment; filename=model_{metadata.get('seed', '')}.{ext}"
                ),
            },
        )
    else:
        # 返回 PNG（优先返回 composite）
        output_img = result.get("composite") or (
            result["images"][0] if result.get("images") else None
        )
        if output_img is None:
            raise HTTPException(status_code=500, detail="工作流未产生任何输出图片。")

        img_bytes = io.BytesIO()
        output_img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return Response(
            content=img_bytes.getvalue(),
            media_type="image/png",
            headers={
                "X-Seed": str(metadata.get("seed", "")),
                "X-Workflow": workflow,
                "X-Mood": str(metadata.get("mood", "")),
            },
        )


# ===== 关闭事件 =====

@app.on_event("shutdown")
async def shutdown():
    """服务关闭时释放 GPU 显存。"""
    from model_loader import get_pipeline, _triposr_model, _pbr_pipeline

    pipe = get_pipeline()
    if pipe is not None:
        del pipe

    # 清理 TripoSR 模型
    if _triposr_model is not None:
        del _triposr_model
        import model_loader
        model_loader._triposr_model = None

    # 清理 PBR 管线
    if _pbr_pipeline is not None:
        del _pbr_pipeline
        import model_loader
        model_loader._pbr_pipeline = None

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
