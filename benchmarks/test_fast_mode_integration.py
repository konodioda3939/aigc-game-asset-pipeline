"""
集成测试：验证推理服务「快速模式」开关端到端可用（项目 C 推理优化）。

前提：服务已在 http://127.0.0.1:8000 运行（python main.py 或 start.bat）。
脚本会自己轮询 /health 等服务就绪，然后连续打 3 次请求：
  1. 标准模式（fast_mode=False）—— 应 ~2.7s
  2. 快速模式（fast_mode=True） —— 应 ~0.8s（LCM 6 步）
  3. 切回标准（fast_mode=False） —— 应再次 ~2.7s（证明 LCM 能正确卸载、标准模式恢复）

用法（服务已启动后）：
  D:/anaconda3/envs/GPUpytorch-env/python.exe benchmarks/test_fast_mode_integration.py
"""
import sys
import time
import json
import urllib.request

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = "http://127.0.0.1:8000"
PROMPT = "1girl, raiden shogun, purple eyes, long hair, masterpiece, best quality"


def wait_ready(timeout=300):
    """轮询 /health 直到服务就绪（模型加载完）。"""
    print("等待服务就绪（首次加载模型约 30-60s）...", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
                data = json.loads(r.read())
                if data.get("status") == "ready":
                    print(f"[ready] 服务就绪，耗时 {time.time() - t0:.0f}s\n", flush=True)
                    return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError("等待服务就绪超时（>300s），请确认服务已启动。")


def call_generate(fast_mode, seed=42):
    """调 /generate，返回 (耗时秒, 图片字节数)。"""
    body = json.dumps({
        "prompt": PROMPT,
        "fast_mode": fast_mode,
        "seed": seed,
        "steps": 25,          # 传默认值；fast_mode=True 时服务端会自动覆盖为 LCM 参数
        "guidance_scale": 7.5,
    }).encode()
    req = urllib.request.Request(
        BASE + "/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as r:
        img = r.read()
    return time.perf_counter() - t0, len(img)


def main():
    wait_ready()

    print("=" * 56)
    print("  集成测试：快速模式开关（标准 → LCM → 切回标准）")
    print("=" * 56)

    scenarios = [
        ("① 标准模式 fast_mode=False", False),
        ("② ⚡快速模式 fast_mode=True", True),
        ("③ 切回标准 fast_mode=False", False),
    ]
    results = []
    for name, fm in scenarios:
        dt, size = call_generate(fm)
        print(f"  {name}: {dt:5.2f}s  ({size / 1024:.0f} KB)", flush=True)
        results.append(dt)

    std1, lcm, std2 = results
    print("\n" + "-" * 56)
    print(f"  标准模式两次耗时: {std1:.2f}s / {std2:.2f}s")
    print(f"  LCM 快速模式耗时: {lcm:.2f}s")
    print(f"  加速比: {std1 / lcm:.1f}×")

    ok_speed = lcm < std1 * 0.5
    ok_recover = abs(std1 - std2) / std1 < 0.35
    print("-" * 56)
    print(f"  {'✅' if ok_speed else '⚠️'} 快速模式加速: {'正常' if ok_speed else '异常'}")
    print(f"  {'✅' if ok_recover else '⚠️'} 切回标准恢复: {'正常' if ok_recover else '异常（LCM 可能未正确卸载）'}")
    print("=" * 56)


if __name__ == "__main__":
    main()
