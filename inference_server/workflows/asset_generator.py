"""
游戏素材生成工作流 — 统一入口。

三种风格 × 两种模式：
  风格: icon（图标）/ scene（场景）/ ui（UI元素）
  模式: text（纯文字生图）/ controlnet（参考图 + ControlNet 精修）

这替代了之前的 prop_icon、scene_mood、ui_elements 三个独立工作流。
"""

import io
import time

from PIL import Image

from .workflow_base import BaseWorkflow, DEFAULT_NEGATIVE


class AssetGeneratorWorkflow(BaseWorkflow):
    """游戏素材生成 — 选风格 + 选模式，统一出图。"""

    def __init__(self, prompt_engine=None):
        super().__init__("asset_generator", prompt_engine)

    def generate(self, params: dict) -> dict:
        prompt = params.get("prompt", "").strip()
        if not prompt:
            raise ValueError("描述不能为空。")

        steps = params.get("steps", 25)
        guidance_scale = params.get("guidance_scale", 7.5)
        seed = params.get("seed") or int(time.time() * 1000) % (2**31)

        # --- 风格 ---
        style = params.get("style", self.template.get("default_style", "icon"))
        styles_cfg = self.template.get("styles", {})
        style_cfg = styles_cfg.get(style, styles_cfg.get("icon", {}))
        style_label = style_cfg.get("label", style)
        style_suffix = style_cfg.get("suffix", "")

        # --- 模式 ---
        reference_image = params.get("reference_image")
        use_controlnet = reference_image is not None
        control_mode = params.get("control_mode", "canny")
        control_strength = params.get("control_strength", 0.85)

        # --- 构建 prompt ---
        rendered = f"{prompt}, {style_suffix}"
        # 追加全局风格后缀
        style_suffix_param = params.get("style_suffix", "")
        if style_suffix_param:
            rendered = f"{rendered}, {style_suffix_param}"
        else:
            rendered = f"{rendered}, masterpiece, best quality"

        # --- 负面 prompt ---
        base_negative = self.template.get("negative_prompt", DEFAULT_NEGATIVE)
        extra_negative = style_cfg.get("negative_extra", "")
        negative = f"{base_negative}, {extra_negative}" if extra_negative else base_negative

        # --- 分辨率 ---
        width = style_cfg.get("width", 512)
        height = style_cfg.get("height", 512)

        print(f"\n{'='*50}", flush=True)
        print(f"  🎯 游戏素材生成 — {style_label}", flush=True)
        print(f"  prompt: {prompt[:80]}...", flush=True)
        print(f"  mode: {'ControlNet/' + control_mode if use_controlnet else 'txt2img'}", flush=True)
        print(f"  size: {width}×{height}, seed={seed}", flush=True)
        print(f"{'='*50}", flush=True)

        # ---- 生成 ----
        if use_controlnet:
            if isinstance(reference_image, bytes):
                reference_image = Image.open(io.BytesIO(reference_image)).convert("RGB")

            result = self._controlnet_generate(
                prompt=rendered,
                control_image=reference_image,
                control_mode=control_mode,
                control_strength=control_strength,
                negative_prompt=negative,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )
        else:
            result = self._txt2img(
                prompt=rendered,
                negative_prompt=negative,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                width=width,
                height=height,
            )

        # ---- 后处理（仅图标风格：去背景 + 居中）----
        if style_cfg.get("post_remove_bg"):
            try:
                result = self._remove_background(result)
            except Exception as e:
                print(f"  [post] ⚠️ 去背景失败（跳过）: {e}", flush=True)

        if style_cfg.get("post_center_crop"):
            try:
                result = self._center_crop(result, target_size=512, padding_ratio=0.15)
            except Exception as e:
                print(f"  [post] ⚠️ 裁切失败（跳过）: {e}", flush=True)

        # ---- 存档 ----
        self._save_image(result, f"asset_{style}", f"seed{seed}")

        print(f"  ✅ 素材生成完成 — {style_label}", flush=True)

        return {
            "images": [result],
            "composite": result,
            "format": "png",
            "metadata": {
                "seed": seed,
                "style": style,
                "mode": "controlnet" if use_controlnet else "text",
                "prompt": prompt,
            },
        }
