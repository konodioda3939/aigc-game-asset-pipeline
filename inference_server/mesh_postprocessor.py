"""
3D 模型后处理编排模块（纯 CPU，不 import torch / diffusers）。

通过 subprocess 调用 Blender headless 跑 blender_scripts/post_process_mesh.py，
收集输出（各级 LOD + 预览图 + manifest）并打包成 ZIP。

被 main.py 的 /post-process-mesh 接口调用。设计要点：
  - 纯 CPU 子进程，不碰 GPU/显存，不影响 SD/TripoSR 全局单例
  - Blender.exe 路径三级回退（env > 默认 > 候选扫描）
  - subprocess 超时兜底，编码用 utf-8/replace 防 Windows GBK 崩
  - 失败时抛清晰异常，由接口层转 HTTP 500，不让服务崩
"""
import os
import json
import subprocess
from pathlib import Path

# bpy 脚本路径（与本模块同目录的 blender_scripts/ 下）
SCRIPT_PATH = Path(__file__).parent / "blender_scripts" / "post_process_mesh.py"

# Blender 可执行文件路径（当前机器已验证 + 换机器时的候选）
DEFAULT_BLENDER_PATH = r"D:\Steam\steamapps\common\Blender\blender.exe"
_CANDIDATE_BLENDER_PATHS = [
    r"D:\Steam\steamapps\common\Blender\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender\blender.exe",
]


def _resolve_blender_path() -> str:
    """三级回退找 blender.exe：环境变量 BLENDER_EXE > 默认 > 候选扫描。找不到抛 FileNotFoundError。"""
    env_path = os.environ.get("BLENDER_EXE")
    if env_path and Path(env_path).exists():
        return env_path
    if Path(DEFAULT_BLENDER_PATH).exists():
        return DEFAULT_BLENDER_PATH
    for p in _CANDIDATE_BLENDER_PATHS:
        if Path(p).exists():
            return p
    raise FileNotFoundError(
        "未找到 Blender 可执行文件。请安装 Blender，或设置环境变量 BLENDER_EXE 指向 blender.exe。"
        f"（已尝试：BLENDER_EXE={env_path or '未设置'}；默认路径 {DEFAULT_BLENDER_PATH} 不存在）"
    )


def run_post_process(
    input_glb_path,
    work_dir,
    target_faces: int = 8000,
    lod_faces=None,
    uv_unwrap: bool = True,
    render_preview: bool = True,
    timeout_sec: int = 300,
) -> bytes:
    """
    调 Blender headless 跑后处理，返回 ZIP 字节流。

    参数:
        input_glb_path: 输入 glb 路径
        work_dir: 临时工作目录，bpy 脚本输出到此（每请求一个，避免并发污染）
        target_faces: LOD0 目标面数
        lod_faces: 额外 LOD 面数列表，如 [2500, 800]
        uv_unwrap: 是否展 UV
        render_preview: 是否渲染预览图
        timeout_sec: 子进程超时秒数（默认 5 分钟）

    返回: ZIP bytes（LOD*.glb + preview*.png + manifest.json）

    异常:
        FileNotFoundError — Blender 未安装/找不到
        subprocess.TimeoutExpired — 超时
        RuntimeError — bpy 脚本执行失败（含脚本 traceback）
    """
    input_glb_path = str(input_glb_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    blender_exe = _resolve_blender_path()
    lod_faces = lod_faces or []
    lod_str = ",".join(str(x) for x in lod_faces)

    cmd = [
        blender_exe,
        "--background",
        "--addons", "io_scene_gltf2",   # 显式确保 glb 导入导出插件可用
        "--python", str(SCRIPT_PATH),
        "--",
        input_glb_path,
        str(work_dir),
        str(target_faces),
        lod_str,
        "true" if uv_unwrap else "false",
        "true" if render_preview else "false",
    ]

    print(f"[post-process] 启动 Blender headless: target={target_faces} "
          f"lod={lod_faces} uv={uv_unwrap} preview={render_preview}", flush=True)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            cwd=str(work_dir),
        )
    except subprocess.TimeoutExpired:
        raise subprocess.TimeoutExpired(
            cmd, timeout_sec,
            f"Blender 后处理超时（>{timeout_sec}秒），可能模型过大或 Blender 卡死。"
        )

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    # 把 bpy 脚本的 [BPY] 日志转发到服务日志，方便排查
    for line in stdout.splitlines():
        if line.startswith("[BPY]"):
            print(line, flush=True)

    manifest_path = work_dir / "manifest.json"
    if not manifest_path.exists():
        tail = "\n".join((stdout + "\n" + stderr).splitlines()[-25:])
        raise RuntimeError(
            f"Blender 后处理未产出 manifest.json（脚本可能未正常启动，如 glTF 插件未加载）。\n"
            f"returncode={proc.returncode}\n--- 输出尾部 ---\n{tail}"
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("status") != "ok":
        raise RuntimeError(
            f"Blender 后处理失败: {manifest.get('error')}\n"
            f"traceback:\n{manifest.get('traceback', '')}"
        )

    zip_data = _pack_zip(work_dir, manifest)
    print(f"[post-process] 完成: {len(manifest.get('lod_levels', []))} 级 LOD, "
          f"ZIP={len(zip_data) / 1024:.0f} KB", flush=True)
    return zip_data


def _pack_zip(work_dir: Path, manifest: dict) -> bytes:
    """把 work_dir 下的各级 LOD glb + 预览图 + manifest 打包成 ZIP（复用 PBR 的打包模式）。"""
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for lod in manifest.get("lod_levels", []):
            fp = work_dir / lod["file"]
            if fp.exists():
                zf.write(fp, lod["file"])
        preview = manifest.get("preview")
        if preview:
            fp = work_dir / preview
            if fp.exists():
                zf.write(fp, preview)
        zf.write(work_dir / "manifest.json", "manifest.json")
    buf.seek(0)
    return buf.getvalue()
