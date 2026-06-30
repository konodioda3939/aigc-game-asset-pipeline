"""
ComfyUI-StableMaterials: PBR Material Generation Node

Wraps gvecchio/StableMaterials (MatFuse-based PBR pipeline) as a ComfyUI node.
Input: text prompt describing a material
Output: 5 PBR texture maps (BaseColor, Normal, Height, Roughness, Metallic)
"""

import os
import sys
import traceback
import numpy as np
import torch
from PIL import Image

# 强制使用镜像（不用 setdefault，因为可能已被设为空值）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"

import folder_paths
import comfy.model_management as mm

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


class StableMaterialsNode:
    """Generate PBR material textures from a text prompt."""

    # Class-level cache for the pipeline
    _pipe = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "rough stone wall, weathered, natural"
                }),
                "steps": ("INT", {
                    "default": 25, "min": 5, "max": 50, "step": 1
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 10.0, "min": 1.0, "max": 20.0, "step": 0.5
                }),
                "tileable": ("BOOLEAN", {
                    "default": True, "label_on": "seamless", "label_off": "single"
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFF
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("basecolor", "normal", "height", "roughness", "metallic")
    FUNCTION = "generate"
    CATEGORY = "AIGC/PBR"

    def generate(self, prompt, steps, guidance_scale, tileable, seed):
        try:
            return self._do_generate(prompt, steps, guidance_scale, tileable, seed)
        except Exception as e:
            print(f"[StableMaterials] ERROR: {e}", flush=True)
            traceback.print_exc()
            # Return blank images on failure
            blank = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (blank, blank, blank, blank, blank)

    def _do_generate(self, prompt, steps, guidance_scale, tileable, seed):
        # ---- Load pipeline (once) ----
        if StableMaterialsNode._pipe is None:
            print("[StableMaterials] Loading pipeline from local cache...", flush=True)
            from diffusers import DiffusionPipeline

            # 优先从本地缓存加载（模型已完整下载，不走网络）
            try:
                pipe = DiffusionPipeline.from_pretrained(
                    "gvecchio/StableMaterials",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    cache_dir=r"d:\aigc-project\cache\hub",
                    local_files_only=True,
                )
                print("[StableMaterials] Loaded from local cache.", flush=True)
            except Exception as e:
                print(f"[StableMaterials] Local cache failed ({e}), trying mirror...", flush=True)
                pipe = DiffusionPipeline.from_pretrained(
                    "gvecchio/StableMaterials",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    cache_dir=r"d:\aigc-project\cache\hub",
                )
            device = mm.get_torch_device()
            pipe.to(device)
            StableMaterialsNode._pipe = pipe
            print("[StableMaterials] Pipeline loaded.", flush=True)

        pipe = StableMaterialsNode._pipe
        device = mm.get_torch_device()

        # Ensure pipeline is on correct device
        pipe_device = str(getattr(pipe, 'device', 'cpu'))
        if pipe_device != str(device):
            print(f"[StableMaterials] Moving pipeline: {pipe_device} -> {device}", flush=True)
            pipe.to(device)

        # ---- Run inference ----
        if seed == 0:
            seed = torch.randint(0, 2**31, (1,)).item()

        generator = torch.Generator(device=device).manual_seed(seed)

        print(f"[StableMaterials] Generating: '{prompt[:80]}' steps={steps} "
              f"cfg={guidance_scale} seed={seed}", flush=True)

        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                tileable=tileable,
            )

        material = result.images[0]

        # ---- Convert to ComfyUI tensor format ([B, H, W, C], float32, 0-1) ----
        textures = [
            material.basecolor,
            material.normal,
            material.height,
            material.roughness,
            material.metallic,
        ]
        names = ["basecolor", "normal", "height", "roughness", "metallic"]

        tensors = []
        for name, img in zip(names, textures):
            if img.mode != "RGB":
                img = img.convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr).unsqueeze(0)
            tensors.append(tensor)
            print(f"  [StableMaterials] {name}: {img.size[0]}x{img.size[1]}", flush=True)

        # Cleanup VRAM
        mm.soft_empty_cache()

        print(f"[StableMaterials] Done!", flush=True)
        return tuple(tensors)


# Register node
NODE_CLASS_MAPPINGS["StableMaterials"] = StableMaterialsNode
NODE_DISPLAY_NAME_MAPPINGS["StableMaterials"] = "StableMaterials (PBR)"
