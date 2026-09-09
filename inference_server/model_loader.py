"""模型加载器：加载Stable Diffusion + LoRA权重 + ControlNet，全局只加载一次，复用管线。"""
import os
import sys

# ==== 打印立即刷新，不做缓冲（用户能看到每一步进度，不会以为卡了）====
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ==== 必须在 import diffusers 之前设置，否则连不上 HuggingFace ====
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 大文件下载超时设为 10 分钟（默认太短，1.4GB 容易超时断连）
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'

import torch
import threading
from pathlib import Path
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    LCMScheduler,
)
from peft import PeftModel

# ==== TripoSR 路径设置（需要 torchmcubes 兼容模块 + TripoSR 源码）====
_TRIPOSR_DIR = Path(r"D:\aigc-project\TripoSR")
_COMPAT_DIR = Path(r"D:\aigc-project\inference_server")  # torchmcubes compat

if str(_COMPAT_DIR) not in sys.path:
    sys.path.insert(0, str(_COMPAT_DIR))  # torchmcubes compat 优先
if str(_TRIPOSR_DIR) not in sys.path:
    sys.path.insert(0, str(_TRIPOSR_DIR))  # TripoSR 源码

# ===== 配置 =====
MODEL_NAME = "gsdf/Counterfeit-V2.5"
LORA_DIR = Path(r"D:\aigc-project\lora_output")
CACHE_DIR = Path(r"D:\aigc-project\cache\hub")

# ===== 模型注册表（多模型切换，项目 ⑤ 扩展）=====
# 每条：base_id 是 HF 仓库 id；lora_dir 非空 → 构建时 PeftModel merge_and_unload 融合进 UNet（如原神 LoRA）；
# needs_download 仅作标记（首用时 from_pretrained 会自动下载到 CACHE_DIR，走 hf-mirror）；
# prompt_suffix 非空 → /generate 自动拼到 prompt 末尾（texture 模式加纹路引导词，压住人脸）；
# lcm_compatible → 能否叠 SD1.5 LCM-LoRA：UNet cross_attention_dim 必须是 768（SD1.5）。
#   SD2.x 模型是 1024，与本项目的 lcm-lora-sdv1-5 不兼容，fast_mode 会自动降级走标准 DPM（见 main.py /generate）。
MODEL_REGISTRY = {
    "anime": {
        "label": "二次元（原神风）",
        "base_id": MODEL_NAME,
        "lora_dir": str(LORA_DIR),
        "needs_download": False,
        "lcm_compatible": True,   # SD1.5（Counterfeit）
    },
    "realistic": {
        "label": "写实风",
        "base_id": "SG161222/Realistic_Vision_V5.1_noVAE",  # 修正：原 _no_inpaint 仓库名不存在（HF 401）
        "lora_dir": None,
        "needs_download": True,
        "lcm_compatible": True,   # SD1.5（Realistic Vision）
    },
    "texture": {
        "label": "纹理/图案",
        "base_id": "dream-textures/texture-diffusion",
        "lora_dir": None,
        "needs_download": True,
        "lcm_compatible": False,  # ⚠️ SD2.x（cross_attention_dim=1024），不能叠 SD1.5 LCM-LoRA
        "prompt_suffix": "seamless tileable texture, flat, game asset, no human, no face",
    },
}
DEFAULT_MODEL = "anime"

# ControlNet 模型 HuggingFace ID 映射
# ControlNet 模型 ID
# modelscope: 国内镜像，优先使用（大文件下载更稳）
# hf: HuggingFace 原版，作为备选
CONTROLNET_MODEL_SOURCES = {
    "canny": {
        # ModelScope 直接使用 HuggingFace 同款 ID（不用 AI-ModelScope/ 前缀）
        "modelscope": "lllyasviel/control_v11p_sd15_canny",
        "hf": "lllyasviel/sd-controlnet-canny",
    },
    "scribble": {
        "modelscope": "lllyasviel/control_v11p_sd15_scribble",
        "hf": "lllyasviel/sd-controlnet-scribble",
    },
    "depth": {
        "modelscope": "lllyasviel/control_v11f1p_sd15_depth",
        "hf": "lllyasviel/sd-controlnet-depth",
    },
}


# ===== 全局单例 =====
_device = None       # "cuda" 或 "cpu"
_dtype = None        # torch.float16 或 torch.float32

_pipeline = None     # txt2img 管线（向后兼容）
_shared_components = None  # dict(vae, text_encoder, tokenizer, unet, scheduler)，ControlNet 共享

_controlnet_models = {}     # mode → ControlNetModel
_controlnet_pipelines = {}  # mode → StableDiffusionControlNetPipeline

_triposr_model = None       # TripoSR TSR 模型实例（全局单例）

_pbr_pipeline = None       # StableMaterials PBR 管线（全局单例）
_pbr_lock = threading.Lock()  # PBR 管线互斥锁（防止并发卸载冲突）

_lcm_active = False           # LCM 快速模式是否启用（项目 C 推理优化）

# ===== 多模型切换状态（项目 ⑤ 扩展）=====
_active_model_key = None          # 当前已加载的模型 key（None=尚未加载）
_swap_lock = threading.Lock()     # 换装互斥锁（_pipeline 原本无锁，防并发 /generate 竞态换装）


# ===== 设备检测 =====

def _detect_device():
    """检测 GPU/CPU，返回 (device_str, torch_dtype)。"""
    global _device, _dtype
    if _device is not None:
        return _device, _dtype

    if torch.cuda.is_available():
        _device = "cuda"
        _dtype = torch.float16
        print(f"[model_loader] 使用 GPU: {torch.cuda.get_device_name(0)}", flush=True)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[model_loader] 显存: {vram:.1f} GB", flush=True)
    else:
        _device = "cpu"
        _dtype = torch.float32
        print("[model_loader] ⚠️ 未检测到 GPU，使用 CPU（生成会较慢）", flush=True)

    return _device, _dtype


# ===== txt2img 管线（向后兼容） =====

def _build_pipeline(spec: dict) -> StableDiffusionPipeline:
    """
    根据注册表 spec 构建一条 txt2img 管线：
      基座 from_pretrained → DPMScheduler → 可选 LoRA merge_and_unload → to(device) → 填 _shared_components。
    不触碰全局 _pipeline（由调用方赋值），便于多模型换装复用同一构建逻辑。
    """
    global _shared_components

    device, dtype = _detect_device()
    base_id = spec["base_id"]
    label = spec.get("label", base_id)

    # --- 加载基座模型（hf-mirror 失败自动回退直连 huggingface.co）---
    print(f"[model_loader] 正在加载基座模型: {base_id}（{label}）...", flush=True)
    saved_endpoint = os.environ.get("HF_ENDPOINT")
    pipe = None
    for endpoint_name in ["hf-mirror", "直连"]:
        try:
            if endpoint_name == "直连" and saved_endpoint:
                os.environ.pop("HF_ENDPOINT", None)
            print(f"[model_loader] 尝试 {endpoint_name}...", flush=True)
            pipe = StableDiffusionPipeline.from_pretrained(
                base_id,
                torch_dtype=dtype,
                safety_checker=None,
                cache_dir=str(CACHE_DIR),
            )
            break
        except Exception as e:
            print(f"[model_loader] ⚠️ {endpoint_name} 加载失败: {type(e).__name__}: {e}", flush=True)
            if endpoint_name == "直连":
                raise
    if saved_endpoint:
        os.environ["HF_ENDPOINT"] = saved_endpoint
    # 兼容修正：部分仓库（如 Realistic_Vision V5.1）scheduler 配置为 algorithm_type=deis，
    # 新版 diffusers 默认 final_sigmas_type=zero 与 deis 不兼容会抛 ValueError → 强制 sigma_min。
    sched_cfg = dict(pipe.scheduler.config)
    if sched_cfg.get("algorithm_type") == "deis":
        sched_cfg["final_sigmas_type"] = "sigma_min"
        print("[model_loader] 已修正 scheduler 兼容性（deis → final_sigmas_type=sigma_min）", flush=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(sched_cfg)
    print("[model_loader] 基座模型加载完成。", flush=True)

    # --- 可选 LoRA 权重融合 ---
    lora_dir = spec.get("lora_dir")
    if lora_dir:
        lora_path = Path(lora_dir)
        lora_weights = lora_path / "adapter_model.safetensors"
        if lora_weights.exists():
            print(f"[model_loader] 正在加载 LoRA 权重: {lora_weights}", flush=True)
            unet = PeftModel.from_pretrained(pipe.unet, str(lora_path))
            pipe.unet = unet.merge_and_unload()
            pipe.unet = pipe.unet.to(device, dtype=dtype)
            print("[model_loader] LoRA 权重已融合进 UNet。", flush=True)
        else:
            print(f"[model_loader] ⚠️ 未找到 LoRA 权重文件: {lora_weights}", flush=True)

    # --- 移到设备 + 优化 ---
    pipe = pipe.to(device)
    # 注意力计算：用 PyTorch 2.x 原生 SDPA（diffusers 在 torch>=2.0 自动启用 AttnProcessor2_0），
    # 不再用 enable_attention_slicing()。
    # 原因（项目 C 推理优化，基线实测 2026-06-30）：512×512/25步 峰值显存仅 2.63GB（8GB 的 33%），
    # 显存非常充裕；attention_slicing 是"以速度换显存"的妥协，在显存不缺时纯属白拖慢。
    # SDPA 同时更快、显存也够用。如未来遇到显存吃紧（大图/多模型共存），再取消下行注释切回。
    # pipe.enable_attention_slicing()

    # --- 保存共享组件（供 ControlNet 管线复用 UNet/VAE/TextEncoder） ---
    _shared_components = {
        "vae": pipe.vae,
        "text_encoder": pipe.text_encoder,
        "tokenizer": pipe.tokenizer,
        "unet": pipe.unet,          # ← 已融合 LoRA 的 UNet（无 LoRA 则是原版 UNet）
        "scheduler": pipe.scheduler,
    }

    print(f"[model_loader] 管线构建完成（{label}）。", flush=True)
    return pipe


def load_pipeline() -> StableDiffusionPipeline:
    """
    加载【默认】txt2img 管线（DEFAULT_MODEL=anime）。
    向后兼容：服务 startup 与旧调用方仍用本函数；多模型按需切换请用 get_pipeline_for_model()。
    """
    global _pipeline, _active_model_key

    if _pipeline is not None:
        print("[model_loader] txt2img 管线已加载，复用现有实例。", flush=True)
        return _pipeline

    _pipeline = _build_pipeline(MODEL_REGISTRY[DEFAULT_MODEL])
    _active_model_key = DEFAULT_MODEL
    print(f"[model_loader] txt2img 管线就绪（默认 {DEFAULT_MODEL}）。", flush=True)
    return _pipeline


def get_pipeline() -> StableDiffusionPipeline | None:
    """获取 txt2img 管线实例。"""
    if _pipeline is None:
        return load_pipeline()
    return _pipeline


# ===== 多模型换装（项目 ⑤ 扩展）=====

def _drop_current_pipeline():
    """
    丢弃当前 txt2img 管线 + 失效共享组件/ControlNet 缓存，释放显存。
    ControlNet 管线与本管线共享同一 UNet（_shared_components），换基座后必须清掉，
    否则 ControlNet 会带着旧 UNet 出图；清掉后按需懒重建（见 get_controlnet_pipeline）。
    """
    global _pipeline, _shared_components, _lcm_active
    _pipeline = None
    _shared_components = None
    _controlnet_pipelines.clear()
    _controlnet_models.clear()
    _lcm_active = False  # 新管线没有 LCM-LoRA；fast_mode 会通过 ensure_lcm_mode 重新挂

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[model_loader] 已卸载旧管线并释放显存。", flush=True)


def get_pipeline_for_model(model_key: str) -> StableDiffusionPipeline:
    """
    按 key 取 txt2img 管线；若与当前活动模型不同则【换装】（卸旧的、构建新的）。
    线程安全（_swap_lock）：并发 /generate 不会同时换装导致串模型。
    命中当前模型时直接返回，零开销。
    """
    global _pipeline, _active_model_key

    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"未知模型 key: '{model_key}'。可用: {list(MODEL_REGISTRY.keys())}"
        )

    with _swap_lock:
        # 命中：当前已是该模型，直接复用
        if _pipeline is not None and _active_model_key == model_key:
            return _pipeline

        spec = MODEL_REGISTRY[model_key]
        print(f"[model_loader] 切换模型 → {model_key}（{spec.get('label', '')}）...", flush=True)

        # 换装前先卸载旧管线（首次加载时 _pipeline 为 None，跳过）
        if _pipeline is not None:
            _drop_current_pipeline()

        # 构建新管线（本地缓存命中 ~8s；首次用该模型会先从 hf-mirror 下载到 CACHE_DIR）
        _pipeline = _build_pipeline(spec)
        _active_model_key = model_key
        print(f"[model_loader] 模型就绪: {model_key}", flush=True)
        return _pipeline


def get_active_model_key() -> str | None:
    """当前已加载的模型 key（None=尚未加载）。"""
    return _active_model_key


def get_model_info() -> list:
    """返回注册表里所有模型的信息（含当前是否活动），供 /models、/health 用。"""
    return [
        {
            "key": key,
            "label": spec.get("label", key),
            "needs_download": spec.get("needs_download", False),
            "active": (_active_model_key == key),
        }
        for key, spec in MODEL_REGISTRY.items()
    ]


# ===== LCM 快速模式（项目 C 推理优化）=====
# LCM-LoRA 叠加在「已融合角色 LoRA 的 UNet」之上，配合 LCMScheduler 实现 4-8 步出图。
# 角色 LoRA 已 merge_and_unload 进 UNet 基础权重，LCM-LoRA 只是叠加其上的 peft adapter，
# unload 只移除 adapter、不影响角色 LoRA —— 因此标准/LCM 模式可安全互切。
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"  # 注意：sdv1-5 带横杠，不是 sdv15


def _download_lcm_lora() -> str:
    """下载 LCM-LoRA 单文件到本地（绕过 hf-mirror 对该仓库 /api/ 目录查询的 401）。"""
    from huggingface_hub import hf_hub_download

    local_dir = CACHE_DIR / "lcm-lora"
    local_dir.mkdir(parents=True, exist_ok=True)
    weight_name = "pytorch_lora_weights.safetensors"
    local_file = local_dir / weight_name
    if local_file.exists():
        return str(local_file)

    print(f"[model_loader] 下载 LCM-LoRA（首次约 135MB）...", flush=True)
    saved_endpoint = os.environ.get("HF_ENDPOINT")
    last_err = None
    try:
        for use_mirror in [True, False]:
            try:
                if use_mirror:
                    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
                else:
                    os.environ.pop("HF_ENDPOINT", None)
                print(f"[model_loader] LCM-LoRA 尝试 {'hf-mirror' if use_mirror else '直连'}...", flush=True)
                hf_hub_download(
                    repo_id=LCM_LORA_ID,
                    filename=weight_name,
                    local_dir=str(local_dir),
                )
                print(f"[model_loader] LCM-LoRA 下载成功。", flush=True)
                return str(local_file)
            except Exception as e:
                last_err = e
                print(f"[model_loader] LCM 下载失败: {str(e)[:120]}", flush=True)
        raise RuntimeError(f"LCM-LoRA 下载失败（镜像和直连均失败）: {last_err}")
    finally:
        if saved_endpoint:
            os.environ["HF_ENDPOINT"] = saved_endpoint
        else:
            os.environ.pop("HF_ENDPOINT", None)


def ensure_lcm_mode(active: bool) -> bool:
    """
    幂等地切换 txt2img 管线到 / 离开 LCM 快速模式。返回最终是否处于 LCM 模式。

    active=True:  加载 LCM-LoRA + 切 LCMScheduler（4-8 步出图，快约 5 倍）
    active=False: 卸载 LCM-LoRA + 恢复 DPMSolverMultistepScheduler（标准 25 步）

    已融合的角色 LoRA 不受影响（已 merge_and_unload 进 UNet 基础权重）。
    ControlNet 管线共享同一个 UNet，因此切回标准模式时务必卸载 LCM-LoRA，
    否则 ControlNet 会带着 LCM-LoRA 却用 DPM scheduler，出图会崩。

    ⚠️ 永不抛异常：若 load 失败（如 SD2.x 模型叠 SD1.5 LCM-LoRA 架构不兼容、或权重损坏），
    会清理残留 adapter 并回退标准模式、返回 False，由调用方决定是否降级（见 main.py /generate）。
    残留空 adapter 是真实陷阱：load 中途抛异常时 peft 已建好 adapter 结构但 _lcm_active 未置 True，
    下次再 load 会与残留 adapter 形状冲突 → 持续 500。故加载前先防御性 unload。
    """
    global _lcm_active
    pipe = get_pipeline()
    if pipe is None:
        return False

    if active and not _lcm_active:
        try:
            pipe.unload_lora_weights()  # 防御性：清掉上次失败 load 残留的空 adapter
            lcm_file = _download_lcm_lora()
            print(f"[model_loader] → 启用 LCM 快速模式...", flush=True)
            pipe.load_lora_weights(lcm_file)
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            _lcm_active = True
            print(f"[model_loader] LCM 已启用（建议 steps=4-8, cfg=1.0-2.0）。", flush=True)
        except Exception as e:
            print(f"[model_loader] ⚠️ LCM 加载失败，回退标准模式：{e}", flush=True)
            try:
                pipe.unload_lora_weights()
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            except Exception:
                pass
            _lcm_active = False
            return False
    elif not active and _lcm_active:
        print(f"[model_loader] → 恢复标准模式...", flush=True)
        pipe.unload_lora_weights()
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        _lcm_active = False
        print(f"[model_loader] 标准模式已恢复。", flush=True)

    return _lcm_active


def is_lcm_active() -> bool:
    """返回当前是否处于 LCM 快速模式。"""
    return _lcm_active


# ===== ControlNet 管线 =====

def _download_via_modelscope(model_id: str, cache_dir: Path) -> str | None:
    """
    通过 ModelScope（国内 AI 模型库）下载模型，返回本地路径。
    失败返回 None。
    """
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        local_path = snapshot_download(model_id, cache_dir=str(cache_dir))
        return local_path
    except Exception as e:
        print(f"[model_loader] ModelScope 下载失败: {e}", flush=True)
        return None


def _download_and_load_controlnet(
    hf_id: str,
    dtype: torch.dtype,
    use_mirror: bool = True,
    max_retries: int = 10,
) -> ControlNetModel | None:
    """
    健壮的 ControlNet 下载 + 加载。

    只下载必需的两个文件（config.json + 一个权重文件），而非整个仓库。
    优先下载 fp16 版本（~725MB），比完整仓库（~2.9GB）省 75% 流量和时间。

    参数：
      use_mirror: True=走镜像, False=直连 huggingface.co
    返回：
      ControlNetModel 实例，或 None（所有尝试均失败）
    """
    from huggingface_hub.utils import EntryNotFoundError

    # --- 如果直连，临时覆盖 HF_ENDPOINT ---
    saved_endpoint = os.environ.get("HF_ENDPOINT")
    if not use_mirror and saved_endpoint:
        os.environ.pop("HF_ENDPOINT", None)
        print("[model_loader] 已切换为直连 huggingface.co（不走镜像）", flush=True)

    try:
        local_dir = CACHE_DIR / "controlnet" / hf_id.replace("/", "--")
        local_dir = Path(str(local_dir))
        local_dir.mkdir(parents=True, exist_ok=True)

        # 检测是否已下载完成
        already_downloaded = (
            (local_dir / "config.json").exists()
            and any(local_dir.glob("*.safetensors"))
        )

        if not already_downloaded:
            # 只下载必需文件（config.json + 一个权重文件）
            # 优先 fp16（~725MB），不存在则回退 fp32（~1.45GB）
            # 比 snapshot_download 全仓库下载（~2.9GB）省一半以上
            weight_file = "diffusion_pytorch_model.fp16.safetensors"
            print(f"[model_loader] 正在下载模型（优先 fp16 ~725MB）...", flush=True)

            # 下载 config.json（几 KB，秒下）
            _download_single_file(hf_id, "config.json", local_dir, max_retries)
            print(f"[model_loader] config.json 下载完成。", flush=True)

            # 下载权重文件（~725MB fp16 或 ~1.45GB fp32）
            try:
                _download_single_file(hf_id, weight_file, local_dir, max_retries)
            except EntryNotFoundError:
                # fp16 不存在，回退到 fp32
                weight_file = "diffusion_pytorch_model.safetensors"
                print(f"[model_loader] fp16 版本不存在，改下 fp32（~1.45GB）...", flush=True)
                _download_single_file(hf_id, weight_file, local_dir, max_retries)
            except Exception:
                # 其他错误也尝试 fp32 回退
                weight_file = "diffusion_pytorch_model.safetensors"
                print(f"[model_loader] fp16 下载失败，尝试 fp32 版本...", flush=True)
                _download_single_file(hf_id, weight_file, local_dir, max_retries)

            print(f"[model_loader] 权重文件下载完成。", flush=True)

        # --- 从本地加载模型 ---
        print(f"[model_loader] 正在加载 ControlNet 模型权重...", flush=True)
        controlnet = ControlNetModel.from_pretrained(
            str(local_dir),
            torch_dtype=dtype,
            local_files_only=True,
        )
        print(f"[model_loader] ControlNet 模型加载成功。", flush=True)
        return controlnet

    except Exception as e:
        print(f"[model_loader] 加载失败: {e}", flush=True)
        return None

    finally:
        # --- 恢复环境变量 ---
        if not use_mirror:
            if saved_endpoint:
                os.environ["HF_ENDPOINT"] = saved_endpoint
            else:
                os.environ.pop("HF_ENDPOINT", None)
            os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'


def _download_single_file(
    repo_id: str,
    filename: str,
    local_dir: Path,
    max_retries: int = 10,
):
    """
    下载单个 HF 文件，带重试 + 断点续传。

    hf_hub_download 默认启用 resume_download，
    下载中断后重试会自动从断点继续。

    注意：404（文件不存在）不重试，直接抛出让上层回退到其他格式。
    """
    import time
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(local_dir),
                resume_download=True,
            )
            return  # 成功
        except EntryNotFoundError:
            # 404 意味着服务端根本没有这个文件，重试毫无意义
            # 直接抛出，让上层 _download_and_load_controlnet 回退到 fp32
            raise
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = min(2 ** attempt, 120)
                print(f"[model_loader] {filename} 下载失败 (尝试 {attempt}/{max_retries}): "
                      f"{str(e)[:150]}", flush=True)
                print(f"[model_loader] {wait} 秒后重试...", flush=True)
                time.sleep(wait)

    raise last_error


def load_controlnet_model(control_mode: str) -> ControlNetModel:
    """
    加载 ControlNet 模型。

    下载优先级（从省流量和缓存复用角度排序）：
      1. HuggingFace 镜像（hf-mirror.com）→ 只下载 2 个文件（~725MB fp16 或 ~1.45GB fp32）
      2. HuggingFace 直连 → 同上，作为镜像不可用时的备选
      3. ModelScope snapshot → 最后手段（会下载整个仓库 ~4GB，但国内网络更稳）
    """
    if control_mode not in CONTROLNET_MODEL_SOURCES:
        raise ValueError(
            f"不支持的 ControlNet 模式: '{control_mode}'。"
            f"可用: {list(CONTROLNET_MODEL_SOURCES.keys())}"
        )

    if control_mode in _controlnet_models:
        return _controlnet_models[control_mode]

    device, dtype = _detect_device()
    sources = CONTROLNET_MODEL_SOURCES[control_mode]
    hf_id = sources["hf"]

    # --- 策略 1：HuggingFace 镜像（只下载 2 个文件，最省流量）---
    print(f"[model_loader] 通过 HuggingFace 镜像下载: {hf_id} ...", flush=True)
    controlnet = _download_and_load_controlnet(hf_id, dtype, use_mirror=True)

    # --- 策略 2：HuggingFace 直连（不走镜像）---
    if controlnet is None:
        print("[model_loader] HuggingFace 镜像失败，尝试直连 huggingface.co ...", flush=True)
        controlnet = _download_and_load_controlnet(hf_id, dtype, use_mirror=False)

    # --- 策略 3：ModelScope snapshot（最后手段，会下载整个仓库 ~4GB）---
    if controlnet is None:
        modelscope_id = sources.get("modelscope")
        if modelscope_id:
            print(f"[model_loader] HuggingFace 均失败，尝试 ModelScope: {modelscope_id} ...", flush=True)
            local_path = _download_via_modelscope(modelscope_id, CACHE_DIR / "modelscope")
            if local_path:
                print(f"[model_loader] ModelScope 下载完成: {local_path}", flush=True)
                controlnet = ControlNetModel.from_pretrained(
                    local_path, torch_dtype=dtype,
                )

    if controlnet is None:
        raise RuntimeError(
            f"\n{'='*60}\n"
            f"  ControlNet '{control_mode}' 下载失败（所有策略均失败）\n"
            f"{'='*60}\n"
            f"\n"
            f"  已尝试：\n"
            f"    1. HuggingFace 镜像 (hf-mirror.com，10 次重试)\n"
            f"    2. HuggingFace 直连 (huggingface.co，10 次重试)\n"
            f"    3. ModelScope 国内镜像 (modelscope.cn)\n"
            f"\n"
            f"  手动解决方案：\n"
            f"    方式 A（推荐）：换个网络好的时段，重启服务自动重试\n"
            f"    方式 B：手动下载后放入指定目录\n"
            f"      → 浏览器打开 https://huggingface.co/{hf_id}\n"
            f"      → 下载 diffusion_pytorch_model.safetensors 和 config.json\n"
            f"      → 放入 {CACHE_DIR / 'controlnet' / hf_id.replace('/', '--')}\n"
            f"      → 重启服务即可\n"
            f"{'='*60}\n"
        )

    _controlnet_models[control_mode] = controlnet
    print(f"[model_loader] ControlNet [{control_mode}] 加载完成。", flush=True)
    return controlnet


def get_controlnet_pipeline(control_mode: str = "canny") -> StableDiffusionControlNetPipeline:
    """
    获取 ControlNet 管线。

    与 txt2img 管线共享 UNet/VAE/TextEncoder/Tokenizer/Scheduler，
    不会重复占用显存。ControlNet 模型本身约 1.4GB（fp16）。

    用法：
        pipe = get_controlnet_pipeline("canny")
        result = pipe(prompt=..., image=canny_preprocessed_image, ...)
    """
    # 确保基础组件已加载
    if _shared_components is None:
        load_pipeline()

    # 已有同模式管线，直接返回
    if control_mode in _controlnet_pipelines:
        return _controlnet_pipelines[control_mode]

    # 加载 ControlNet 模型
    controlnet = load_controlnet_model(control_mode)
    device, _ = _detect_device()

    # 用共享组件 + ControlNet 创建新管线
    comps = _shared_components
    pipe = StableDiffusionControlNetPipeline(
        vae=comps["vae"],
        text_encoder=comps["text_encoder"],
        tokenizer=comps["tokenizer"],
        unet=comps["unet"],            # ← 复用已融合 LoRA 的 UNet
        controlnet=controlnet,
        scheduler=comps["scheduler"],
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    # 显式调用 .to() 确保管线内部设备标记正确（组件已在 GPU 上，不会重复转移）
    pipe = pipe.to(device)

    # ControlNet 管线不需要 attention_slicing（UNet 已配置过）

    _controlnet_pipelines[control_mode] = pipe
    print(f"[model_loader] ControlNet 管线 [{control_mode}] 就绪。", flush=True)
    return pipe


def get_loaded_controlnet_modes() -> list:
    """返回已加载的 ControlNet 模式列表。"""
    return list(_controlnet_pipelines.keys())


def get_available_controlnet_modes() -> list:
    """返回所有支持的 ControlNet 模式列表。"""
    return list(CONTROLNET_MODEL_SOURCES.keys())


# ===== TripoSR 3D 生成管线 =====

# TripoSR 模型 HuggingFace ID
TRIPOSR_MODEL_ID = "stabilityai/TripoSR"
TRIPOSR_CONFIG = "config.yaml"
TRIPOSR_WEIGHTS = "model.ckpt"

# ===== StableMaterials PBR 材质生成 =====
PBR_MODEL_ID = "gvecchio/StableMaterials"
PBR_CACHE_DIR = CACHE_DIR / "pbr" / "StableMaterials"


def load_triposr_model():
    """
    加载 TripoSR 模型（懒加载，首次调用时下载 ~1.68GB 权重）。

    HF 上的权重文件使用了新版 ViT 键名（encoder.layer.X.attention.attention.query），
    而我们克隆的 TripoSR 代码期望旧版键名（layers.X.attention.q_proj）。
    此函数会在加载时自动做键名转换。

    返回：
        TSR 模型实例（已在 GPU 上，eval 模式）
    """
    global _triposr_model

    if _triposr_model is not None:
        return _triposr_model

    device, dtype = _detect_device()

    # 延迟导入 TripoSR（确保 sys.path 已配置）
    from tsr.system import TSR
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf
    import re

    # --- 下载模型权重 ---
    print(f"[model_loader] 正在加载 TripoSR 模型: {TRIPOSR_MODEL_ID} ...", flush=True)
    print(f"[model_loader] 模型权重 ~1.68GB，首次下载请耐心等待...", flush=True)

    saved_endpoint = os.environ.get("HF_ENDPOINT")

    for strategy_name, use_mirror in [("HuggingFace 镜像", True), ("HuggingFace 直连", False)]:
        try:
            if not use_mirror and saved_endpoint:
                os.environ.pop("HF_ENDPOINT", None)
                print("[model_loader] 切换为直连 huggingface.co ...", flush=True)
            elif use_mirror and saved_endpoint:
                os.environ["HF_ENDPOINT"] = saved_endpoint

            # 手动下载配置文件
            config_path = hf_hub_download(
                repo_id=TRIPOSR_MODEL_ID,
                filename=TRIPOSR_CONFIG,
            )
            # 手动下载权重文件
            weight_path = hf_hub_download(
                repo_id=TRIPOSR_MODEL_ID,
                filename=TRIPOSR_WEIGHTS,
            )
            print(f"[model_loader] TripoSR 文件下载完成（{strategy_name}）。", flush=True)
            break
        except Exception as e:
            print(f"[model_loader] {strategy_name} 下载失败: {e}", flush=True)
            if strategy_name == "HuggingFace 直连":
                if saved_endpoint:
                    os.environ["HF_ENDPOINT"] = saved_endpoint
                raise RuntimeError(
                    f"TripoSR 模型下载失败（所有策略均失败）。\n"
                    f"模型地址: https://huggingface.co/{TRIPOSR_MODEL_ID}\n"
                    f"请检查网络后重启服务。"
                )

    # 恢复环境变量
    if saved_endpoint:
        os.environ["HF_ENDPOINT"] = saved_endpoint

    # --- 加载 checkpoint 并做键名转换 ---
    print("[model_loader] 正在转换模型权重键名...", flush=True)
    ckpt = torch.load(weight_path, map_location="cpu")
    ckpt = _remap_triposr_keys(ckpt)
    print("[model_loader] 键名转换完成。", flush=True)

    # --- 用配置创建模型，然后加载转换后的权重 ---
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    model = TSR(cfg)
    model.load_state_dict(ckpt)

    # --- 移到 GPU ---
    model.to(device)
    model.eval()

    # --- 设置 marching cubes 分辨率 ---
    model.set_marching_cubes_resolution(256)

    _triposr_model = model
    print(f"[model_loader] TripoSR 模型就绪（设备: {device}）。", flush=True)
    return model


def _remap_triposr_keys(state_dict: dict) -> dict:
    """
    将 HF 新版 ViT 键名转换为 TripoSR 代码期望的旧版键名。

    新版 (HF checkpoint):  image_tokenizer.model.encoder.layer.X.attention.attention.query.weight
    旧版 (TripoSR 代码):    image_tokenizer.model.layers.X.attention.q_proj.weight

    映射规则：
      encoder.layer.{N}  →  layers.{N}
      attention.attention.query  →  attention.q_proj
      attention.attention.key    →  attention.k_proj
      attention.attention.value  →  attention.v_proj
      attention.output.dense     →  attention.o_proj
      intermediate.dense         →  mlp.fc1
      output.dense               →  mlp.fc2  (仅 MLP 路径)
    """
    import re

    new_dict = {}
    renamed_count = 0

    for key, value in state_dict.items():
        new_key = key

        # 只处理 image_tokenizer 相关的键
        if key.startswith("image_tokenizer.model."):
            # encoder.layer.X → layers.X
            new_key = re.sub(r'encoder\.layer\.(\d+)', r'layers.\1', new_key)

            # attention.attention.query/key/value → attention.q_proj/k_proj/v_proj
            new_key = new_key.replace('attention.attention.query', 'attention.q_proj')
            new_key = new_key.replace('attention.attention.key', 'attention.k_proj')
            new_key = new_key.replace('attention.attention.value', 'attention.v_proj')

            # attention.output.dense → attention.o_proj
            new_key = new_key.replace('attention.output.dense', 'attention.o_proj')

            # intermediate.dense → mlp.fc1
            new_key = new_key.replace('intermediate.dense', 'mlp.fc1')

            # output.dense → mlp.fc2 (注意：attention.output.dense 已经在上一步被替换)
            new_key = new_key.replace('output.dense', 'mlp.fc2')

            if new_key != key:
                renamed_count += 1

        new_dict[new_key] = value

    print(f"[model_loader] 重映射了 {renamed_count} 个权重键。", flush=True)
    return new_dict


def get_triposr_model():
    """获取 TripoSR 模型实例（首次调用自动下载+加载）。"""
    return load_triposr_model()


# ====== VRAM 管理（SD 管线卸载/恢复，为 PBR 腾出显存） =====

def _offload_sd_pipeline():
    """
    将 SD 1.5 + ControlNet 管线卸载到 CPU，释放显存给 StableMaterials。

    StableMaterials 是独立架构，不与 SD 1.5 共享组件。
    8GB 显存无法同时容纳两个管线。
    """
    global _pipeline, _controlnet_models, _controlnet_pipelines

    offloaded_anything = False

    if _pipeline is not None and str(_pipeline.device) != "cpu":
        print("[model_loader] 将 SD 管线转移到 CPU（为 StableMaterials 腾出显存）...", flush=True)
        _pipeline.to("cpu")
        offloaded_anything = True

    # ControlNet 管线也卸载
    for pipe in _controlnet_pipelines.values():
        if pipe is not None and str(pipe.device) != "cpu":
            pipe.to("cpu")
            offloaded_anything = True

    for model in _controlnet_models.values():
        if model is not None and str(model.device) != "cpu":
            model.to("cpu")
            offloaded_anything = True

    if offloaded_anything:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print("[model_loader] SD 管线已卸载到 CPU，显存已释放。", flush=True)


def _restore_sd_pipeline():
    """
    将 SD 1.5 + ControlNet 管线恢复到 GPU。

    在 PBR 生成完成后调用，恢复常规文生图/ControlNet 功能。
    """
    global _pipeline, _controlnet_models, _controlnet_pipelines

    device, _ = _detect_device()
    if device != "cuda":
        return  # 没有 GPU，无需恢复

    restored_anything = False

    if _pipeline is not None and str(_pipeline.device) != "cuda":
        print("[model_loader] 恢复 SD 管线到 GPU...", flush=True)
        _pipeline.to("cuda")
        restored_anything = True

    for mode in list(_controlnet_models.keys()):
        model = _controlnet_models.get(mode)
        if model is not None and str(model.device) != "cuda":
            model.to("cuda")
            restored_anything = True

    for pipe in _controlnet_pipelines.values():
        if pipe is not None and str(pipe.device) != "cuda":
            pipe.to("cuda")
            restored_anything = True

    if restored_anything:
        print("[model_loader] SD 管线已恢复到 GPU。", flush=True)


# ====== StableMaterials PBR 管线 ======

def load_pbr_pipeline():
    """
    加载 StableMaterials PBR 管线（懒加载，首次调用时下载 ~2-3GB 权重）。

    StableMaterials 是专用的 PBR 材质生成管线，使用 MatFuse 架构
    （改编自 LDM），不依赖 SD 1.5。支持 LCM 4 步快速推理。

    返回：
        StableMaterialsPipeline 实例（已在 GPU 上）
    """
    global _pbr_pipeline

    if _pbr_pipeline is not None:
        print("[model_loader] PBR 管线已加载，复用现有实例。", flush=True)
        return _pbr_pipeline

    device, dtype = _detect_device()

    # --- 卸载 SD 管线到 CPU，为 StableMaterials 腾出显存 ---
    _offload_sd_pipeline()

    # --- 多策略下载，带重试和断点续传（同 TripoSR 模式）---
    print(f"[model_loader] 正在加载 StableMaterials PBR 管线: {PBR_MODEL_ID} ...", flush=True)
    print(f"[model_loader] 模型权重 ~2-3GB，首次下载请耐心等待...", flush=True)

    saved_endpoint = os.environ.get("HF_ENDPOINT")
    pipe = None

    for strategy_name, use_mirror in [("HuggingFace 镜像", True), ("HuggingFace 直连", False)]:
        try:
            if not use_mirror and saved_endpoint:
                os.environ.pop("HF_ENDPOINT", None)
                print("[model_loader] 切换为直连 huggingface.co ...", flush=True)
            elif use_mirror and saved_endpoint:
                os.environ["HF_ENDPOINT"] = saved_endpoint

            from diffusers import DiffusionPipeline

            pipe = DiffusionPipeline.from_pretrained(
                PBR_MODEL_ID,
                torch_dtype=dtype,
                trust_remote_code=True,
                cache_dir=str(CACHE_DIR),
            )
            # 使用模型默认 scheduler（标准 25-30 步推理，质量最佳）
            # 注意：LCM scheduler 需要配合 unet_lcm 权重使用，当前用标准 UNet
            print(f"[model_loader] 使用模型默认 scheduler（标准推理模式）。", flush=True)

            print(f"[model_loader] StableMaterials 加载成功（{strategy_name}）。", flush=True)
            break
        except Exception as e:
            print(f"[model_loader] {strategy_name} 下载失败: {e}", flush=True)
            pipe = None
            if strategy_name == "HuggingFace 直连":
                if saved_endpoint:
                    os.environ["HF_ENDPOINT"] = saved_endpoint
                raise RuntimeError(
                    f"\n{'='*60}\n"
                    f"  StableMaterials 模型下载失败（所有策略均失败）\n"
                    f"{'='*60}\n"
                    f"\n"
                    f"  已尝试：\n"
                    f"    1. HuggingFace 镜像 (hf-mirror.com)\n"
                    f"    2. HuggingFace 直连 (huggingface.co)\n"
                    f"\n"
                    f"  手动解决方案：\n"
                    f"    方式 A（推荐）：换个网络好的时段，重启服务自动重试\n"
                    f"    方式 B：手动下载\n"
                    f"      → 浏览器打开 https://huggingface.co/{PBR_MODEL_ID}\n"
                    f"      → 下载所有文件\n"
                    f"      → 放入 {PBR_CACHE_DIR}\n"
                    f"      → 重启服务即可\n"
                    f"{'='*60}\n"
                )

    # 恢复环境变量
    if saved_endpoint:
        os.environ["HF_ENDPOINT"] = saved_endpoint

    # --- 移到 GPU + 优化 ---
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    _pbr_pipeline = pipe
    print(f"[model_loader] StableMaterials PBR 管线就绪（设备: {device}）。", flush=True)
    return pipe


def get_pbr_pipeline():
    """获取 StableMaterials PBR 管线实例（首次调用自动下载+加载，线程安全）。"""
    with _pbr_lock:
        return load_pbr_pipeline()
