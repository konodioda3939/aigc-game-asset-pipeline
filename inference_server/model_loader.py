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
from pathlib import Path
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
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

def load_pipeline() -> StableDiffusionPipeline:
    """
    加载 txt2img 管线 + LoRA 权重。
    服务启动时调用一次，之后全局复用。
    """
    global _pipeline, _shared_components

    if _pipeline is not None:
        print("[model_loader] txt2img 管线已加载，复用现有实例。", flush=True)
        return _pipeline

    device, dtype = _detect_device()

    # --- 加载基座模型 ---
    print(f"[model_loader] 正在加载基座模型: {MODEL_NAME} ...", flush=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        safety_checker=None,
        cache_dir=str(CACHE_DIR),
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    print("[model_loader] 基座模型加载完成。", flush=True)

    # --- 加载 LoRA 权重 ---
    lora_weights = LORA_DIR / "adapter_model.safetensors"
    if lora_weights.exists():
        print(f"[model_loader] 正在加载 LoRA 权重: {lora_weights}", flush=True)
        unet = PeftModel.from_pretrained(pipe.unet, str(LORA_DIR))
        pipe.unet = unet.merge_and_unload()
        pipe.unet = pipe.unet.to(device, dtype=dtype)
        print("[model_loader] LoRA 权重已融合进 UNet。", flush=True)
    else:
        print("[model_loader] ⚠️ 未找到 LoRA 权重文件", flush=True)

    # --- 移到设备 + 优化 ---
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    # --- 保存共享组件（供 ControlNet 管线复用 UNet/VAE/TextEncoder） ---
    _shared_components = {
        "vae": pipe.vae,
        "text_encoder": pipe.text_encoder,
        "tokenizer": pipe.tokenizer,
        "unet": pipe.unet,          # ← 已融合 LoRA 的 UNet
        "scheduler": pipe.scheduler,
    }

    _pipeline = pipe
    print(f"[model_loader] txt2img 管线就绪。", flush=True)
    return _pipeline


def get_pipeline() -> StableDiffusionPipeline | None:
    """获取 txt2img 管线实例。"""
    if _pipeline is None:
        return load_pipeline()
    return _pipeline


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
