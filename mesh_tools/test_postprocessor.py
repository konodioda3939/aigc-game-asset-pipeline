"""临时单元测试：验证 mesh_postprocessor 模块（subprocess 调 Blender + ZIP 打包）。D-5b。"""
import sys
sys.path.insert(0, "d:/aigc-project/inference_server")
from mesh_postprocessor import run_post_process

z = run_post_process(
    input_glb_path="d:/aigc-project/mesh_tools/orange_raw.glb",
    work_dir="d:/aigc-project/mesh_tools/d5_test2",
    target_faces=8000,
    lod_faces=[2500, 800],
    uv_unwrap=True,
    render_preview=True,
)
out = "d:/aigc-project/mesh_tools/d5_test2.zip"
with open(out, "wb") as f:
    f.write(z)
print(f"OK zip={len(z)} bytes ({len(z)/1024:.0f} KB) -> {out}")
