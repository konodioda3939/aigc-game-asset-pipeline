"""
工作流基类 — 所有游戏美术工作流的抽象父类。

提供：
  - 统一的 txt2img / img2img / ControlNet 生成封装
  - 图像后处理工具（拼接画板、去背景、居中裁切）
  - 输出存档
"""

import io
import time
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from prompts.engine import PromptTemplateEngine, get_prompt_engine

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULT_NEGATIVE = (
    "lowres, bad anatomy, bad hands, text, error, extra digit, "
    "fewer digits, cropped, worst quality, low quality, normal quality, "
    "jpeg artifacts, signature, watermark, username, blurry"
)


class BaseWorkflow(ABC):
    """游戏美术工作流基类。"""

    def __init__(self, workflow_id: str, prompt_engine: PromptTemplateEngine | None = None):
        self.workflow_id = workflow_id
        self.prompt_engine = prompt_engine or get_prompt_engine()
        self.template = self.prompt_engine.load_template(workflow_id)
        self.fast_mode = False  # LCM 快速模式开关（项目 C 推理优化），由 API 透传设置

    # ===== 子类必须实现 =====

    @abstractmethod
    def generate(self, params: dict) -> dict:
        """
        执行工作流，返回结果字典。

        参数：
          params: 包含 prompt, seed, steps, guidance_scale, 以及工作流特定参数

        返回：
          {
            "images": [PIL.Image, ...],      # 生成的图片列表
            "composite": PIL.Image | None,    # 合成的预览图（如拼接画板）
            "format": "png" | "zip",          # 输出格式
            "metadata": dict,                 # 种子、耗时等元信息
          }
        """
        ...

    # ===== 通用生成封装 =====

    def _get_pipeline(self):
        """获取 SD 1.5 + LoRA 管线（复用 model_loader 全局单例）。"""
        from model_loader import get_pipeline
        pipe = get_pipeline()
        if pipe is None:
            raise RuntimeError("SD 管线尚未加载，请等待服务就绪。")
        return pipe

    def _txt2img(
        self,
        prompt: str,
        negative_prompt: str = DEFAULT_NEGATIVE,
        steps: int = 25,
        guidance_scale: float = 7.5,
        seed: int | None = None,
        width: int = 512,
        height: int = 512,
    ) -> Image.Image:
        """纯文本生图。"""
        pipe = self._get_pipeline()
        from model_loader import ensure_lcm_mode  # 方法内 import，避免循环依赖
        actual_seed = seed if seed is not None else int(time.time() * 1000) % (2**31)

        # ⚡ 快速模式（项目 C 推理优化）：切 LCM，并夹紧到 LCM 适用范围（≤8 步、低 CFG）
        # 子类（如 character_concept）可能把 steps 抬到 30、cfg 抬到 8.5，这里统一压回 LCM 区间。
        if self.fast_mode:
            ensure_lcm_mode(True)
            steps = min(steps, 8)
            guidance_scale = 1.5
            print(f"  [txt2img] ⚡ 快速模式（LCM）: steps={steps}, cfg={guidance_scale}", flush=True)
        else:
            ensure_lcm_mode(False)

        print(f"  [txt2img] prompt: {prompt[:80]}...", flush=True)
        print(f"  [txt2img] steps={steps}, cfg={guidance_scale}, "
              f"size={width}×{height}, seed={actual_seed}", flush=True)

        generator = torch.Generator(pipe.device).manual_seed(actual_seed)
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
            )
        return result.images[0]

    def _img2img(
        self,
        prompt: str,
        init_image: Image.Image,
        strength: float = 0.65,
        negative_prompt: str = DEFAULT_NEGATIVE,
        steps: int = 25,
        guidance_scale: float = 7.5,
        seed: int | None = None,
    ) -> Image.Image:
        """
        图生图（基于 SD 1.5 + LoRA 管线）。

        StableDiffusionPipeline 原生支持 img2img：
        传入 image 参数（PIL Image）和 strength 参数即可。
        strength=1.0 等于纯 txt2img，strength=0.0 等于完全保留原图。
        """
        pipe = self._get_pipeline()
        from model_loader import ensure_lcm_mode  # img2img 不支持快速模式，强制标准模式
        ensure_lcm_mode(False)
        actual_seed = seed if seed is not None else int(time.time() * 1000) % (2**31)

        # 确保 init_image 是 RGB 且尺寸合理
        if init_image.mode != "RGB":
            init_image = init_image.convert("RGB")
        # 先限制最大尺寸（防止大图导致 img2img 极慢）
        init_image = self._resize_for_controlnet(init_image, self._CONTROLNET_MAX_SIZE)
        # SD 1.5 要求尺寸为 8 的倍数
        w, h = init_image.size
        w = (w // 8) * 8
        h = (h // 8) * 8
        if (w, h) != init_image.size:
            init_image = init_image.resize((w, h), Image.LANCZOS)

        print(f"  [img2img] prompt: {prompt[:80]}...", flush=True)
        print(f"  [img2img] strength={strength}, steps={steps}, "
              f"seed={actual_seed}", flush=True)

        generator = torch.Generator(pipe.device).manual_seed(actual_seed)
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        return result.images[0]

    # SD 1.5 原生分辨率 512×512，超过 768 后显存和速度急剧恶化
    _CONTROLNET_MAX_SIZE = 768
    _LATENT_ALIGN = 8

    @staticmethod
    def _resize_for_controlnet(image: Image.Image, max_size: int = 768) -> Image.Image:
        """
        将输入图缩放到适合 SD 1.5 + ControlNet 处理的尺寸。

        保持宽高比，长边不超过 max_size，尺寸对齐到 8 的倍数。
        1024×1024 → 768×768（速度提升约 5 倍，显存安全）。
        """
        w, h = image.size
        longest = max(w, h)
        if longest <= max_size:
            return image

        scale = max_size / longest
        new_w = int(w * scale)
        new_h = int(h * scale)
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8
        print(f"  [resize] 自动缩放: {w}×{h} → {new_w}×{new_h} "
              f"（SD 1.5 原生 512×512，大图会极慢）", flush=True)
        return image.resize((new_w, new_h), Image.LANCZOS)

    def _controlnet_generate(
        self,
        prompt: str,
        control_image: Image.Image,
        control_mode: str = "canny",
        control_strength: float = 0.85,
        negative_prompt: str = DEFAULT_NEGATIVE,
        steps: int = 25,
        guidance_scale: float = 7.5,
        seed: int | None = None,
    ) -> Image.Image:
        """ControlNet 可控生成（复用现有管线）。"""
        from model_loader import get_controlnet_pipeline, ensure_lcm_mode

        # ControlNet 与 txt2img 共享 UNet：若上一次生图启用了 LCM 快速模式，UNet 上还挂着
        # LCM-LoRA，会让 ControlNet（用 DPM scheduler）崩图。强制切回标准模式。
        ensure_lcm_mode(False)
        pipe = get_controlnet_pipeline(control_mode)
        actual_seed = seed if seed is not None else int(time.time() * 1000) % (2**31)

        # ---- 关键：缩放输入图，否则大图会极慢甚至 OOM ----
        original_size = control_image.size
        control_image = self._resize_for_controlnet(control_image, self._CONTROLNET_MAX_SIZE)

        # 预处理：根据模式提取控制图
        from main import PREPROCESSORS

        if control_mode in PREPROCESSORS:
            preprocessor = PREPROCESSORS[control_mode]
            processed = preprocessor(control_image)
        else:
            processed = control_image

        # 确保预处理后尺寸一致
        if processed.size != control_image.size:
            processed = processed.resize(control_image.size, Image.LANCZOS)

        print(f"  [controlnet] mode={control_mode}, strength={control_strength}, "
              f"size={control_image.size[0]}×{control_image.size[1]} "
              f"(原始: {original_size[0]}×{original_size[1]}), "
              f"seed={actual_seed}", flush=True)

        generator = torch.Generator(pipe.device).manual_seed(actual_seed)
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=processed,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=control_strength,
                generator=generator,
            )
        return result.images[0]

    # ===== 图像后处理 =====

    @staticmethod
    def _stitch_grid(
        images: list[Image.Image],
        cols: int = 2,
        labels: list[str] | None = None,
    ) -> Image.Image:
        """
        将多张图拼接为网格画板。

        参数：
          images: 图片列表（假设全部同尺寸）
          cols: 每行列数
          labels: 每张图的标签（可选）
        """
        if not images:
            raise ValueError("images 列表为空")

        n = len(images)
        rows = (n + cols - 1) // cols

        img_w, img_h = images[0].size
        label_h = 30 if labels else 0

        canvas_w = cols * img_w
        canvas_h = rows * (img_h + label_h)
        canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))

        for i, img in enumerate(images):
            row = i // cols
            col = i % cols
            x = col * img_w
            y = row * (img_h + label_h)

            # 贴图
            if img.mode == "RGBA":
                canvas.paste(img, (x, y), img)
            else:
                canvas.paste(img, (x, y))

            # 标签
            if labels and i < len(labels):
                draw = ImageDraw.Draw(canvas)
                label_text = labels[i]
                # 用简单文字（无自定义字体时用默认）
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except OSError:
                    font = ImageFont.load_default()
                bbox = draw.textbbox((0, 0), label_text, font=font)
                tw = bbox[2] - bbox[0]
                draw.text(
                    (x + (img_w - tw) // 2, y + img_h + 5),
                    label_text,
                    fill=(200, 200, 200),
                    font=font,
                )

        return canvas

    @staticmethod
    def _remove_background(image: Image.Image) -> Image.Image:
        """
        用 rembg 去除背景，返回 RGBA 图片。

        复用 TripoSR 已有的 rembg 依赖。
        """
        from rembg import remove
        return remove(image)

    @staticmethod
    def _center_crop(
        image: Image.Image,
        target_size: int = 512,
        padding_ratio: float = 0.15,
    ) -> Image.Image:
        """
        自动裁剪到图像主体区域，居中后填充到目标尺寸。

        策略：
          1. 如果图片有透明通道，用 alpha 通道找到主体边界
          2. 否则用简单的中心裁切
          3. 在主体周围保留 padding_ratio 的留白
          4. 缩放到 target_size × target_size
        """
        if image.mode == "RGBA":
            # 用 alpha 通道找主体
            alpha = np.array(image.split()[-1])
            rows = np.any(alpha > 30, axis=1)
            cols = np.any(alpha > 30, axis=0)
            if rows.any() and cols.any():
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
            else:
                ymin, ymax = 0, image.height
                xmin, xmax = 0, image.width
        else:
            # RGB：中心区域裁切
            w, h = image.size
            margin_w = int(w * 0.1)
            margin_h = int(h * 0.1)
            xmin, ymin = margin_w, margin_h
            xmax, ymax = w - margin_w, h - margin_h

        # 添加 padding
        box_w = xmax - xmin
        box_h = ymax - ymin
        pad_w = int(box_w * padding_ratio)
        pad_h = int(box_h * padding_ratio)

        xmin = max(0, xmin - pad_w)
        ymin = max(0, ymin - pad_h)
        xmax = min(image.width, xmax + pad_w)
        ymax = min(image.height, ymax + pad_h)

        # 裁剪
        cropped = image.crop((xmin, ymin, xmax, ymax))

        # 缩放到目标尺寸（保持正方形，用最大边）
        crop_w, crop_h = cropped.size
        max_side = max(crop_w, crop_h)

        # 创建正方形画布
        if cropped.mode == "RGBA":
            square = Image.new("RGBA", (max_side, max_side), (0, 0, 0, 0))
        else:
            square = Image.new("RGB", (max_side, max_side), (255, 255, 255))

        offset_x = (max_side - crop_w) // 2
        offset_y = (max_side - crop_h) // 2
        square.paste(cropped, (offset_x, offset_y))

        # 缩放到目标尺寸
        square = square.resize((target_size, target_size), Image.LANCZOS)
        return square

    # ===== 存档 =====

    @staticmethod
    def _save_image(image: Image.Image, prefix: str, suffix: str = "") -> Path:
        """保存图片到 outputs/ 目录，返回路径。"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        name = f"{timestamp}_{prefix}"
        if suffix:
            name = f"{name}_{suffix}"
        path = OUTPUT_DIR / f"{name}.png"
        image.save(path)
        return path

    @staticmethod
    def _make_zip_response(images: dict[str, Image.Image], prefix: str) -> bytes:
        """将多张图打包为 ZIP 字节流。"""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, img in images.items():
                img_bytes = io.BytesIO()
                # RGBA 保留透明度
                fmt = "PNG"
                img.save(img_bytes, format=fmt)
                img_bytes.seek(0)
                zf.writestr(f"{name}.png", img_bytes.getvalue())
        zip_buffer.seek(0)
        return zip_buffer.getvalue()

    # ===== 对外接口 =====

    def get_meta(self) -> dict:
        """返回工作流元信息（供 API 和 UI）。"""
        return self.prompt_engine.get_workflow_meta(self.workflow_id) or {}

    def get_input_schema(self) -> dict:
        """返回工作流输入参数 schema。"""
        meta = self.get_meta()
        return meta.get("input_schema", {})
