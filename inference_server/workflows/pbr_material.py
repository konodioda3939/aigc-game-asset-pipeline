"""
PBR 材质生成工作流：文字描述 → 完整 PBR 贴图集。

流程（复用 Phase 6 StableMaterials）：
  文字描述 → SD 卸载到 CPU（腾显存）
           → StableMaterials 推理（25 步）
           → 生成 5 张 PBR 贴图（BaseColor/Normal/Height/Roughness/Metallic）
           → 打包 MetallicSmoothness（R=Metallic, A=Smoothness）
           → ZIP 返回
"""

import io
import time
import zipfile
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from PIL import Image

from .workflow_base import BaseWorkflow


class PBRMaterialWorkflow(BaseWorkflow):
    """PBR 材质生成 — 文字描述生成全套 PBR 纹理贴图。"""

    def __init__(self, prompt_engine=None):
        super().__init__("pbr_material", prompt_engine)

    def generate(self, params: dict) -> dict:
        prompt = params.get("prompt", "").strip()
        if not prompt:
            raise ValueError("材质描述不能为空。")

        steps = params.get("steps", 25)
        guidance_scale = params.get("guidance_scale", 10.0)
        tileable = params.get("tileable", True)
        seed = params.get("seed") or int(time.time() * 1000) % (2**31)

        print(f"\n{'='*50}", flush=True)
        print(f"  🧱 PBR 材质生成工作流 — 开始", flush=True)
        print(f"  prompt: {prompt[:80]}...", flush=True)
        print(f"  tileable={tileable}, steps={steps}, cfg={guidance_scale}", flush=True)
        print(f"{'='*50}", flush=True)

        # ==== 复用 StableMaterials 管线（Phase 6） ====
        from model_loader import get_pbr_pipeline, _restore_sd_pipeline

        pipe = get_pbr_pipeline()

        generator_obj = torch.Generator(device=pipe.device).manual_seed(seed)
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator_obj,
                tileable=tileable,
            )

        material = result.images[0]
        basecolor: Image.Image = material.basecolor
        normal: Image.Image = material.normal
        height: Image.Image = material.height
        roughness: Image.Image = material.roughness
        metallic: Image.Image = material.metallic

        # 打包 Metallic(R) + Smoothness(1-Roughness, A)
        metallic_arr = np.array(metallic.convert("L"))
        roughness_arr = np.array(roughness.convert("L"))
        smoothness_arr = (255 - roughness_arr).astype(np.uint8)
        h, w = metallic_arr.shape
        packed = np.zeros((h, w, 4), dtype=np.uint8)
        packed[:, :, 0] = metallic_arr
        packed[:, :, 3] = smoothness_arr
        packed_img = Image.fromarray(packed, mode="RGBA")

        # 预览缩略图
        preview = basecolor.copy()
        preview.thumbnail((256, 256), Image.LANCZOS)

        # ZIP 打包
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, img in [
                ("basecolor", basecolor),
                ("normal", normal),
                ("height", height),
                ("roughness", roughness),
                ("metallic", metallic),
                ("metallic_smoothness", packed_img),
                ("preview", preview),
            ]:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)
                zf.writestr(f"{name}.png", buf.getvalue())

        zip_buffer.seek(0)
        zip_data = zip_buffer.getvalue()

        # 存档
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_name = prompt[:20].replace(" ", "_").replace(",", "").replace("/", "_")
        archive_path = Path(__file__).parent.parent / "outputs" / f"{timestamp}_pbr_{safe_name}.zip"
        archive_path.write_bytes(zip_data)

        # 恢复 SD 管线
        _restore_sd_pipeline()

        print(f"  ✅ PBR 材质完成 — ZIP={len(zip_data)/1024:.0f} KB, "
              f"seed={seed}", flush=True)

        return {
            "images": [preview],
            "composite": preview,
            "format": "zip",
            "zip_data": zip_data,
            "metadata": {
                "seed": seed,
                "prompt": prompt,
                "tileable": tileable,
            },
        }
