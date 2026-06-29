"""
3D 模型生成工作流：图片 → 3D 模型（.glb）。

流程（复用 Phase 5 TripoSR）：
  上传参考图 → rembg 去背景 → resize_foreground 裁剪
             → TripoSR 推理 → Marching Cubes 提取 mesh
             → 导出 .glb（贴图内嵌）
"""

import io
import time
from pathlib import Path

from PIL import Image

from .workflow_base import BaseWorkflow

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"


class Model3DWorkflow(BaseWorkflow):
    """3D 模型生成 — 单张图片转带贴图的 3D mesh。"""

    def __init__(self, prompt_engine=None):
        super().__init__("model_3d", prompt_engine)

    def generate(self, params: dict) -> dict:
        prompt = params.get("prompt", "").strip()

        # 3D 模型生成：必须有参考图，prompt 可选
        reference_image = params.get("reference_image")
        if reference_image is None:
            raise ValueError("3D 模型生成需要上传参考图！请拖入一张角色或物体图片。")

        resolution = params.get("resolution", 256)
        output_format = params.get("output_format", "glb")
        seed = params.get("seed") or int(time.time() * 1000) % (2**31)

        # 读取参考图
        if isinstance(reference_image, bytes):
            ref_image = Image.open(io.BytesIO(reference_image)).convert("RGB")
        elif isinstance(reference_image, Image.Image):
            ref_image = reference_image.convert("RGB")
        else:
            raise ValueError("参考图格式不支持。")

        original_size = ref_image.size
        print(f"\n{'='*50}", flush=True)
        print(f"  📦 3D 模型生成工作流 — 开始", flush=True)
        print(f"  image: {original_size[0]}×{original_size[1]}", flush=True)
        print(f"  format={output_format}, resolution={resolution}", flush=True)
        print(f"{'='*50}", flush=True)

        # ---- 关键：大图先缩放，否则 rembg 和 TripoSR 都会极慢 ----
        longest = max(ref_image.size)
        if longest > 1024:
            scale = 1024 / longest
            new_w = int(ref_image.size[0] * scale)
            new_h = int(ref_image.size[1] * scale)
            print(f"[model_3d] 自动缩放: {ref_image.size[0]}×{ref_image.size[1]} "
                  f"→ {new_w}×{new_h} "
                  f"（大图会严重拖慢去背景和推理）", flush=True)
            ref_image = ref_image.resize((new_w, new_h), Image.LANCZOS)

        # ==== 复用 TripoSR 管线（Phase 5） ====
        from model_loader import get_triposr_model
        from tsr.utils import remove_background, resize_foreground

        # 预处理：去背景 + 裁剪
        print("[model_3d] 去背景...", flush=True)
        ref_image = remove_background(ref_image)
        print(f"[model_3d] 去背景完成, mode={ref_image.mode}, "
              f"size={ref_image.size}", flush=True)

        ref_image = resize_foreground(ref_image, 0.85)
        ref_image = ref_image.convert("RGB")

        # 获取模型
        model = get_triposr_model()
        model.set_marching_cubes_resolution(resolution)
        model.renderer.set_chunk_size(4096)

        device = str(model.device) if hasattr(model, 'device') else "cuda"
        print(f"[model_3d] 推理中...", flush=True)

        import torch
        with torch.no_grad():
            scene_codes = model(ref_image, device=device)

        # 提取 mesh
        print("[model_3d] 提取 3D mesh...", flush=True)
        mesh = model.extract_mesh(
            scene_codes, has_vertex_color=True, resolution=resolution
        )[0]
        print(f"[model_3d] mesh: {len(mesh.vertices)} 顶点, "
              f"{len(mesh.faces)} 面", flush=True)

        # 导出为字节
        model_bytes_io = io.BytesIO()
        if output_format == "obj":
            mesh.export(model_bytes_io, file_type="obj")
            media_type = "model/obj"
            ext = "obj"
        else:
            mesh.export(model_bytes_io, file_type="glb")
            media_type = "model/gltf-binary"
            ext = "glb"
        model_bytes_io.seek(0)
        model_data = model_bytes_io.getvalue()

        # 存档
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        archive_path = OUTPUT_DIR / f"{timestamp}_3d_model.{ext}"
        archive_path.write_bytes(model_data)
        print(f"[model_3d] 已保存: {archive_path} "
              f"({len(model_data)/1024:.0f} KB)", flush=True)

        # 清理显存
        del scene_codes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 生成预览缩略图（用原图作为预览）
        preview = ref_image.copy()
        preview.thumbnail((512, 512), Image.LANCZOS)

        print(f"  ✅ 3D 模型完成 — {len(mesh.vertices):,} 顶点, "
              f"格式={ext}", flush=True)

        return {
            "images": [preview],
            "composite": preview,
            "format": ext,  # "glb" 或 "obj"
            "media_type": media_type,
            "model_data": model_data,
            "metadata": {
                "seed": seed,
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "file_format": ext,
                "prompt": prompt,
            },
        }
