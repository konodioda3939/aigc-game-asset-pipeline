"""
推理性能基准测试脚本（项目 C：推理优化）

用途
----
测量「SD 1.5 + LoRA」文生图的【单图耗时】和【峰值显存】，
作为「优化前 vs 优化后」的对比基准。

它直接复用 inference_server.model_loader 的全局管线，
所以测出来的数字 = 真实推理服务的性能，对比才有意义。

用法
----
# 1) 现状基线（已开 attention_slicing 时的旧版，历史对照）
D:/anaconda3/envs/GPUpytorch-env/python.exe benchmarks/bench_inference.py --label baseline

# 2) SDPA 优化后（默认，model_loader 已切 SDPA）
D:/anaconda3/envs/GPUpytorch-env/python.exe benchmarks/bench_inference.py --label sdpa

# 3) LCM 4 步快速模式（叠加 LCM-LoRA）
D:/anaconda3/envs/GPUpytorch-env/python.exe benchmarks/bench_inference.py --label lcm --lcm

输出
----
1. 控制台打印结果表
2. benchmarks/results/<label>.json  —— 结构化结果，供后续脚本读取对比
3. benchmarks/results/<label>_seed42.png —— 第一次生成的图，留作画质对比基准
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path

# ==== 打印立即刷新 + UTF-8（Windows 控制台中文不乱码）====
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ==== HF 镜像必须在 import diffusers/torch 加载模型之前设置（国内网络）====
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

# ==== 让脚本能 import inference_server.model_loader ====
_ROOT = Path(__file__).resolve().parent.parent
_INF_DIR = _ROOT / "inference_server"
if str(_INF_DIR) not in sys.path:
    sys.path.insert(0, str(_INF_DIR))

import torch
from model_loader import get_pipeline, CACHE_DIR  # noqa: E402  (import 必须在 sys.path 设置之后)

# ==== 常量 ====
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# LCM-LoRA 权重（SD 1.5 专用，约 135MB，走 hf-mirror）
# 注意：仓库名是 lcm-lora-sdv1-5（v1-5 中间有横杠），不是 lcm-lora-sdv15
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"

# 固定测试用 prompt（原神风格，匹配项目训练的 LoRA，保证每次考卷一样）
TEST_PROMPT = (
    "1girl, raiden shogun, purple eyes, long hair, "
    "masterpiece, best quality, highly detailed"
)
TEST_NEGATIVE = (
    "lowres, bad anatomy, bad hands, text, error, extra digit, "
    "fewer digits, cropped, worst quality, low quality, blurry"
)
FIXED_SEED = 42  # 固定种子 → 每次生成的图几乎一样，对比才公平


def _bytes_to_gb(num_bytes: float) -> float:
    return num_bytes / (1024 ** 3)


def apply_lcm(pipe):
    """
    叠加 LCM-LoRA + 切换 LCMScheduler，把管线变成「4 步快速出图」模式。

    LCM-LoRA 是独立小 LoRA（~64MB），叠加在已融合角色 LoRA 的 UNet 之上，不冲突。
    首次会联网下载（走 hf-mirror，失败回退直连 huggingface.co）。

    注意：直接用 pipe.load_lora_weights(repo_id) 会先调 model_info 查仓库目录，
    hf-mirror 对该仓库的 /api/ 路径返回 401；故改为手动 hf_hub_download 单文件
    （走 /resolve/ 路径，公开文件不触发认证）后，再 load_lora_weights 本地文件。
    """
    from diffusers import LCMScheduler
    from huggingface_hub import hf_hub_download

    weight_name = "pytorch_lora_weights.safetensors"
    local_dir = CACHE_DIR / "lcm-lora"
    local_file = local_dir / weight_name
    local_dir.mkdir(parents=True, exist_ok=True)

    if not local_file.exists():
        print(f"[lcm] 下载 {LCM_LORA_ID}/{weight_name}（约 64MB）...", flush=True)
        saved_endpoint = os.environ.get("HF_ENDPOINT")
        last_err = None
        downloaded = False
        for strategy, use_mirror in [("hf-mirror", True), ("直连 huggingface.co", False)]:
            try:
                if use_mirror:
                    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
                else:
                    os.environ.pop("HF_ENDPOINT", None)
                print(f"[lcm] 尝试 {strategy} ...", flush=True)
                hf_hub_download(
                    repo_id=LCM_LORA_ID,
                    filename=weight_name,
                    local_dir=str(local_dir),
                )
                downloaded = True
                print(f"[lcm] {strategy} 下载成功。", flush=True)
                break
            except Exception as e:
                last_err = e
                print(f"[lcm] {strategy} 失败: {str(e)[:150]}", flush=True)
        # 恢复镜像设置
        if saved_endpoint:
            os.environ["HF_ENDPOINT"] = saved_endpoint
        if not downloaded:
            raise RuntimeError(f"LCM-LoRA 下载失败（镜像和直连均失败）: {last_err}")

    print(f"[lcm] 加载本地 LCM-LoRA: {local_file}", flush=True)
    pipe.load_lora_weights(str(local_file))
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    print(f"[lcm] LCM-LoRA 已加载，scheduler 切换为 LCMScheduler。", flush=True)


def _measure_one(pipe, prompt, negative, steps, cfg, width, height, seed):
    """
    跑一次完整文生图，返回 (耗时秒, 峰值显存GB, PIL.Image)。

    峰值显存用 torch.cuda.max_memory_allocated() —— 它统计「这次推理」新占用的
    显存峰值（在调用前 reset 过），最能反映生图本身的显存开销。
    """
    torch.cuda.reset_peak_memory_stats()      # 清零，只统计这一次
    torch.cuda.synchronize()                  # 等 GPU 空闲，计时才准

    generator = torch.Generator(pipe.device).manual_seed(seed)
    t0 = time.perf_counter()

    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=cfg,
            width=width,
            height=height,
            generator=generator,
        )

    torch.cuda.synchronize()                  # 等最后一步算完
    elapsed = time.perf_counter() - t0
    peak_bytes = torch.cuda.max_memory_allocated()
    peak_gb = _bytes_to_gb(peak_bytes)

    return elapsed, peak_gb, result.images[0]


def run_scenario(
    label: str,
    steps: int = 25,
    cfg: float = 7.5,
    size: int = 512,
    seed: int = FIXED_SEED,
    repeat: int = 3,
    warmup: int = 1,
    lcm: bool = False,
):
    """
    跑一个测试场景：先 warmup（预热，不计时），再正式测 repeat 次。

    warmup 很关键：第一次生图会触发惰性初始化（CUDA kernel 编译、显存池分配、
    LCM-LoRA 首次加载等），比后续慢很多，不能计入统计。
    """
    mode = "LCM 快速" if lcm else "标准"
    print(f"\n{'=' * 60}")
    print(f"  📊 测试场景: {label}（{mode}模式）")
    print(f"  尺寸 {size}×{size} · 步数 {steps} · CFG {cfg} · 种子 {seed}")
    print(f"  预热 {warmup} 次 + 正式测 {repeat} 次")
    print(f"{'=' * 60}")

    pipe = get_pipeline()  # 复用全局单例（首次会加载模型，约 30-60s）

    if lcm:
        apply_lcm(pipe)

    # ---- 预热 ----
    for i in range(warmup):
        print(f"  [warmup {i + 1}/{warmup}] 预热中（不计时）...", flush=True)
        _measure_one(pipe, TEST_PROMPT, TEST_NEGATIVE, steps, cfg, size, size, seed)

    # ---- 正式测量 ----
    times, peaks = [], []
    first_image = None
    for i in range(repeat):
        elapsed, peak_gb, img = _measure_one(
            pipe, TEST_PROMPT, TEST_NEGATIVE, steps, cfg, size, size, seed
        )
        times.append(elapsed)
        peaks.append(peak_gb)
        if first_image is None:
            first_image = img  # 保存第一张做画质对比基准
        per_step = elapsed / steps if steps else 0
        print(
            f"  [run {i + 1}/{repeat}] 耗时 {elapsed:6.2f}s "
            f"（每步 {per_step:5.2f}s）· 峰值显存 {peak_gb:5.2f} GB",
            flush=True,
        )

    # ---- 统计 ----
    avg_time = sum(times) / len(times)
    stats = {
        "label": label,
        "mode": "lcm" if lcm else "standard",
        "size": f"{size}x{size}",
        "steps": steps,
        "guidance_scale": cfg,
        "seed": seed,
        "repeat": repeat,
        "time_seconds": {
            "mean": round(avg_time, 3),
            "min": round(min(times), 3),
            "max": round(max(times), 3),
        },
        "per_step_seconds": {
            "mean": round(avg_time / steps, 3) if steps else 0,
        },
        "peak_vram_gb": {
            "mean": round(sum(peaks) / len(peaks), 3),
            "max": round(max(peaks), 3),
        },
        "all_times": [round(t, 3) for t in times],
        "all_peaks": [round(p, 3) for p in peaks],
    }

    # ---- 存第一张图（画质对比基准）----
    if first_image is not None:
        img_path = RESULTS_DIR / f"{label}_seed{seed}.png"
        first_image.save(img_path)
        print(f"  🖼️  基准图已存: {img_path}")

    return stats


def _print_summary(all_stats: list):
    """打印一张总览表。"""
    print(f"\n\n{'=' * 70}")
    print("  📋 基准测试结果总览")
    print(f"{'=' * 70}")
    header = f"{'场景':<14} {'模式':<8} {'尺寸':<10} {'步数':<6} {'平均耗时':<10} {'峰值显存':<10}"
    print(header)
    print("-" * 70)
    for s in all_stats:
        print(
            f"{s['label']:<14} {s['mode']:<8} {s['size']:<10} {s['steps']:<6} "
            f"{s['time_seconds']['mean']:<10.2f} {s['peak_vram_gb']['mean']:<10.2f}"
        )
    print(f"{'=' * 70}")
    print("  （平均耗时单位：秒；峰值显存单位：GB）\n")


def main():
    parser = argparse.ArgumentParser(description="SD 1.5 + LoRA 推理性能基准测试")
    parser.add_argument("--label", default="run",
                        help="结果标签，如 baseline / sdpa / lcm（决定输出文件名）")
    parser.add_argument("--lcm", action="store_true",
                        help="启用 LCM 快速模式（叠加 LCM-LoRA，4 步出图）")
    parser.add_argument("--size", type=int, default=512,
                        help="生图尺寸（正方形边长，默认 512）")
    parser.add_argument("--steps", type=int, default=25, help="推理步数（默认 25）")
    parser.add_argument("--cfg", type=float, default=7.5, help="引导强度（默认 7.5）")
    parser.add_argument("--repeat", type=int, default=3, help="正式测量次数（默认 3）")
    parser.add_argument("--warmup", type=int, default=1, help="预热次数（默认 1）")
    args = parser.parse_args()

    # LCM 模式默认参数：用户未显式指定时，用 LCM 推荐值（4 步、低 CFG）
    if args.lcm:
        if args.steps == 25:
            args.steps = 4
        if args.cfg == 7.5:
            args.cfg = 1.5
        print(f"⚡ LCM 快速模式：steps={args.steps}, cfg={args.cfg}")

    # 打印环境信息（便于复现）
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"显存总量: {vram_total:.1f} GB")
    else:
        print("⚠️ 未检测到 GPU！")

    # 跑一个场景
    stats = run_scenario(
        label=args.label,
        steps=args.steps,
        cfg=args.cfg,
        size=args.size,
        repeat=args.repeat,
        warmup=args.warmup,
        lcm=args.lcm,
    )

    _print_summary([stats])

    # 存 JSON
    out_json = RESULTS_DIR / f"{args.label}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"💾 结果已存: {out_json}")
    print(f"   后续优化后用 --label <新标签> 再跑一次，即可对比。")


if __name__ == "__main__":
    main()
