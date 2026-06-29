"""
角色概念图工作流：文字 → 角色转身图。

流程：
  文字 → Prompt 模板（character turnaround sheet）
       → 单次 txt2img（宽幅 1024×576）
       → 一张图包含正面/侧面/背面/3/4 多角度
       → 模型天然保证角色一致性
"""

import time

from .workflow_base import BaseWorkflow, DEFAULT_NEGATIVE


class CharacterConceptWorkflow(BaseWorkflow):
    """角色概念图 — 角色转身图（单张多角度）。"""

    def __init__(self, prompt_engine=None):
        super().__init__("character_concept", prompt_engine)

    def generate(self, params: dict) -> dict:
        prompt = params.get("prompt", "").strip()
        if not prompt:
            raise ValueError("角色描述不能为空。")

        steps = params.get("steps", 25)
        guidance_scale = params.get("guidance_scale", 7.5)
        seed = params.get("seed") or int(time.time() * 1000) % (2**31)

        turnaround_cfg = self.template.get("turnaround", {})
        if not turnaround_cfg:
            raise ValueError("模板缺少 turnaround 配置。")

        # 渲染转身图 prompt
        tpl = turnaround_cfg.get("template", "")
        # 风格后缀固定使用模板默认值，不做用户可调（区分于 asset_generator）
        rendered = tpl.replace("{prompt}", prompt)
        rendered = rendered.replace(
            "{style_suffix}",
            self.template.get("style_suffix", ""),
        )

        width = turnaround_cfg.get("width", 1024)
        height = turnaround_cfg.get("height", 576)
        negative = turnaround_cfg.get(
            "negative",
            self.template.get("negative_prompt", DEFAULT_NEGATIVE),
        )
        if steps < 30:
            steps = turnaround_cfg.get("steps_hint", 30)
        if guidance_scale < 8.0:
            guidance_scale = turnaround_cfg.get("guidance_scale", 8.5)

        print(f"\n{'='*50}", flush=True)
        print(f"  🎭 角色概念图 — 转身图", flush=True)
        print(f"  prompt: {prompt[:80]}...", flush=True)
        print(f"  size: {width}×{height}, steps={steps}, "
              f"cfg={guidance_scale}, seed={seed}", flush=True)
        print(f"{'='*50}", flush=True)

        result = self._txt2img(
            prompt=rendered,
            negative_prompt=negative,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            width=width,
            height=height,
        )

        self._save_image(result, "character_turnaround", f"seed{seed}")

        print(f"  ✅ 角色转身图完成", flush=True)

        return {
            "images": [result],
            "composite": result,
            "format": "png",
            "metadata": {
                "seed": seed,
                "prompt": prompt,
                "size": f"{width}×{height}",
            },
        }
