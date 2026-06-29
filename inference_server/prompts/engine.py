"""
Prompt 模板引擎：加载、渲染、管理游戏美术工作流的 Prompt 模板。

设计理念：
  Prompt 模板从代码中分离出来，存入 JSON 文件。美术换风格/调关键词
  不需要改 Python 代码，只需要编辑 JSON（或者通过 Web UI 生成）。
"""

import json
from pathlib import Path
from typing import Any

_PROMPTS_DIR = Path(__file__).parent

# 工作流配置元数据（ID → 名称/描述/输入 schema）
_WORKFLOW_META = {
    "character_concept": {
        "id": "character_concept",
        "name": "角色概念图",
        "name_en": "Character Concept",
        "description": "输入角色描述，AI 一次生成角色转身参考图（正面/侧面/背面/3/4），角色天然一致。",
        "icon": "🎭",
        "input_schema": {
            "prompt": {
                "type": "text",
                "label": "角色描述",
                "placeholder": "例如: female knight, silver armor, blue cape, fantasy style",
                "required": True,
            },
        },
        "output": "1 张 1024×576 角色转身图（正面/侧面/背面/3/4）",
    },
    "asset_generator": {
        "id": "asset_generator",
        "name": "游戏素材生成",
        "name_en": "Game Asset Generator",
        "description": "选风格（图标/场景/UI）+ 选模式（文字直出 / ControlNet精修），统一出图。",
        "icon": "🎯",
        "input_schema": {
            "prompt": {
                "type": "text",
                "label": "描述文字",
                "placeholder": "例如: golden sword / ancient forest ruins / dark fantasy panel",
                "required": True,
            },
            "style": {
                "type": "select",
                "label": "素材风格",
                "options": [
                    {"value": "icon", "label": "⚔️ 图标"},
                    {"value": "scene", "label": "🏞️ 场景"},
                    {"value": "ui", "label": "🎨 UI 元素"},
                ],
                "default": "icon",
            },
            "reference_image": {
                "type": "image",
                "label": "参考图（可选）",
                "placeholder": "上传参考图 → ControlNet 精修；不上传 → 纯文字生成",
                "required": False,
            },
            "control_mode": {
                "type": "select",
                "label": "ControlNet 模式（有参考图时生效）",
                "options": [
                    {"value": "canny", "label": "📐 Canny — 线稿精修"},
                    {"value": "scribble", "label": "✏️ Scribble — 草图生成"},
                    {"value": "depth", "label": "📏 Depth — 深度保持"},
                ],
                "default": "canny",
            },
        },
        "output": "1 张 PNG（图标自动去背景居中，场景宽幅 768×512）",
    },
    "model_3d": {
        "id": "model_3d",
        "name": "3D 模型生成",
        "name_en": "3D Model",
        "description": "上传物体/道具图片（白色背景最佳），AI 自动生成带贴图的 3D 模型。默认 OBJ 格式，Unity 原生支持。",
        "icon": "📦",
        "input_schema": {
            "prompt": {
                "type": "text",
                "label": "模型描述（可选）",
                "placeholder": "仅用于存档命名，不影响 3D 生成",
                "required": False,
            },
            "reference_image": {
                "type": "image",
                "label": "参考图（必须）",
                "placeholder": "上传物体/角色正面照，白色背景效果最佳",
                "required": True,
            },
            "resolution": {
                "type": "select",
                "label": "Mesh 精度",
                "options": [
                    {"value": "128", "label": "⚡ 128 — 快速预览"},
                    {"value": "256", "label": "🎯 256 — 标准质量（推荐）"},
                    {"value": "512", "label": "💎 512 — 最高精度（更慢）"},
                ],
                "default": "256",
            },
            "output_format": {
                "type": "select",
                "label": "输出格式",
                "options": [
                    {"value": "obj", "label": "📦 OBJ — Unity 原生支持（推荐）"},
                    {"value": "glb", "label": "🎯 GLB — 需 glTFast 插件，贴图内嵌"},
                ],
                "default": "obj",
            },
        },
        "output": "1 个 3D 模型文件（OBJ 或 GLB），可直接导入 Unity",
    },
    "pbr_material": {
        "id": "pbr_material",
        "name": "PBR 材质",
        "name_en": "PBR Material",
        "description": "输入材质描述（如 'rough stone wall'），AI 自动生成完整 PBR 贴图集（颜色/法线/粗糙度/金属度）。",
        "icon": "🧱",
        "input_schema": {
            "prompt": {
                "type": "text",
                "label": "材质描述",
                "placeholder": "例如: rough stone wall, wooden floor planks, rusted metal panel",
                "required": True,
            },
            "tileable": {
                "type": "boolean",
                "label": "无缝平铺 (Tileable)",
                "default": True,
            },
            "guidance_scale": {
                "type": "text",
                "label": "引导强度",
                "placeholder": "默认 10.0（1-20）",
                "required": False,
            },
        },
        "output": "ZIP 含 7 张 PBR 贴图（basecolor/normal/roughness/metallic/height/metallic_smoothness/preview）",
    },
}


class PromptTemplateEngine:
    """Prompt 模板引擎 — 加载 JSON 模板，渲染为最终 prompt。"""

    def __init__(self, prompts_dir: Path | None = None):
        self._prompts_dir = prompts_dir or _PROMPTS_DIR
        self._cache: dict[str, dict] = {}

    def load_template(self, workflow_name: str) -> dict:
        """加载某个工作流的 Prompt 模板 JSON。"""
        if workflow_name in self._cache:
            return self._cache[workflow_name]

        template_path = self._prompts_dir / f"{workflow_name}_prompts.json"
        if not template_path.exists():
            raise FileNotFoundError(
                f"找不到工作流 '{workflow_name}' 的 Prompt 模板: {template_path}"
            )

        with open(template_path, "r", encoding="utf-8") as f:
            template = json.load(f)

        self._cache[workflow_name] = template
        return template

    def render(
        self,
        workflow_name: str,
        prompt: str,
        **kwargs,
    ) -> str:
        """
        渲染单个 prompt。

        参数：
          workflow_name: 工作流 ID
          prompt: 用户输入的原始描述
          kwargs: 模板变量（如 mood, view, style_suffix 等）

        返回：
          渲染后的完整 prompt 字符串
        """
        template = self.load_template(workflow_name)

        # 先构建替换变量字典
        vars_dict: dict[str, str] = {
            "prompt": prompt,
            "style_suffix": kwargs.get(
                "style_suffix", template.get("style_suffix", "masterpiece, best quality")
            ),
        }

        # 如果有 mood，注入 mood 对应的关键词
        mood = kwargs.get("mood")
        if mood and "moods" in template:
            mood_config = template["moods"].get(mood, {})
            if isinstance(mood_config, dict):
                vars_dict["mood_keywords"] = mood_config.get("keywords", "")
            else:
                vars_dict["mood_keywords"] = str(mood_config)

        # 如果有 view，取对应视角的模板
        view = kwargs.get("view")
        if view and "views" in template:
            view_template = template["views"].get(view, "")
            if isinstance(view_template, dict):
                view_template = view_template.get("template", "")
            # 先把 view_template 里的 {prompt} 等替换掉
            for k, v in vars_dict.items():
                view_template = view_template.replace(f"{{{k}}}", v)
            return view_template

        # 用基础模板或默认 prompt 格式
        base_template = template.get("base_template", "{prompt}, {style_suffix}")
        result = base_template
        for k, v in vars_dict.items():
            result = result.replace(f"{{{k}}}", v)

        # 注入 mood_keywords 如果有
        if "mood_keywords" in vars_dict and vars_dict["mood_keywords"]:
            result = f"{result}, {vars_dict['mood_keywords']}"

        return result

    def render_all_views(
        self,
        workflow_name: str,
        prompt: str,
        **kwargs,
    ) -> dict[str, str]:
        """
        渲染某个工作流的所有视图/变体。

        返回：
          {view_name: rendered_prompt} 字典
        """
        template = self.load_template(workflow_name)
        views = template.get("views", {})

        if not views:
            # 没有多视图的工作流 → 单一渲染
            return {"default": self.render(workflow_name, prompt, **kwargs)}

        result = {}
        for view_name, view_config in views.items():
            view_kwargs = dict(kwargs)
            view_kwargs["view"] = view_name
            result[view_name] = self.render(workflow_name, prompt, **view_kwargs)

        return result

    def get_style_suffix(self, workflow_name: str) -> str:
        """获取某工作流的默认风格后缀。"""
        template = self.load_template(workflow_name)
        return template.get("style_suffix", "masterpiece, best quality")

    def get_negative_prompt(self, workflow_name: str) -> str:
        """获取某工作流的默认负面 prompt。"""
        template = self.load_template(workflow_name)
        return template.get("negative_prompt", "")

    def list_workflows(self) -> list[dict[str, Any]]:
        """列出所有可用工作流的元信息（供 API 和 Web UI 使用）。"""
        result = []
        for wf_id, meta in _WORKFLOW_META.items():
            # 检查对应的模板文件是否存在
            template_path = self._prompts_dir / f"{wf_id}_prompts.json"
            available = template_path.exists()
            result.append({**meta, "available": available})
        return result

    def get_workflow_meta(self, workflow_name: str) -> dict | None:
        """获取单个工作流的元信息。"""
        return _WORKFLOW_META.get(workflow_name)

    def get_resolution_hint(self, workflow_name: str) -> tuple[int, int]:
        """获取某工作流推荐的输出分辨率。"""
        template = self.load_template(workflow_name)
        size = template.get("resolution", {})
        w = size.get("width", 512)
        h = size.get("height", 512)
        return (w, h)


# 全局单例
_engine: PromptTemplateEngine | None = None


def get_prompt_engine() -> PromptTemplateEngine:
    """获取 PromptTemplateEngine 全局单例。"""
    global _engine
    if _engine is None:
        _engine = PromptTemplateEngine()
    return _engine
