"""
D-1 现状评估：批量体检 TripoSR 输出的毛坯模型。

读，不写。统计每个模型的面数/顶点/UV/材质/封闭性/尺寸，
输出到控制台 + mesh_tools/d1_assessment.json。

用法：
    D:/anaconda3/envs/GPUpytorch-env/python.exe mesh_tools/inspect_mesh.py
"""
import os
import sys
import json
import glob

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np          # noqa: E402
import trimesh              # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 支持命令行传目录（如 verify LOD 输出）；无参数时默认搜主管线 + ComfyUI 输出
if len(sys.argv) > 1:
    SEARCH_DIRS = sys.argv[1:]
else:
    SEARCH_DIRS = [
        os.path.join(PROJECT_ROOT, "inference_server", "outputs"),
        os.path.join(PROJECT_ROOT, "ComfyUI", "output"),
    ]
EXTS = ("*.glb", "*.obj", "*.gltf")


def human_count(n):
    if n >= 10000:
        return f"{n/10000:.2f}万"
    return str(n)


def inspect_one(path):
    rec = {"path": os.path.relpath(path, PROJECT_ROOT), "format": os.path.splitext(path)[1].lower()[1:]}
    rec["size_mb"] = round(os.path.getsize(path) / 1024 / 1024, 3)

    try:
        scene = trimesh.load(path, force="scene", process=False)
    except Exception as e:
        rec["error"] = f"load failed: {e}"
        return rec

    geoms = list(scene.geometry.values()) if hasattr(scene, "geometry") else [scene]
    geoms = [g for g in geoms if hasattr(g, "vertices")]

    if not geoms:
        rec["error"] = "no mesh geometry"
        return rec

    rec["geometry_count"] = len(geoms)
    rec["vertices"] = sum(len(g.vertices) for g in geoms)
    rec["faces"] = sum(len(g.faces) for g in geoms)

    # 视觉信息：区分真纹理 UV / 顶点色 / 无
    visual_types, has_uv, uv_unique, has_vertex_colors, has_texture_material = [], False, 0, False, False
    for g in geoms:
        vis = getattr(g, "visual", None)
        if vis is None:
            continue
        vtype = type(vis).__name__
        visual_types.append(vtype)
        if "Color" in vtype and getattr(vis, "vertex_colors", None) is not None:
            has_vertex_colors = True
        if "Texture" in vtype:
            has_texture_material = True
            uv = getattr(vis, "uv", None)
            if uv is not None and len(uv) > 0:
                has_uv = True
                try:
                    uv_unique = max(uv_unique, len(set(map(tuple, np.asarray(uv).reshape(-1, 2).tolist()))))
                except Exception:
                    uv_unique = max(uv_unique, len(uv))
    rec["visual_types"] = sorted(set(visual_types))
    rec["has_uv"] = has_uv
    rec["uv_unique_points"] = uv_unique
    rec["has_vertex_colors"] = has_vertex_colors
    rec["has_texture_material"] = has_texture_material

    # 封闭性
    rec["watertight"] = bool(all(getattr(g, "is_watertight", False) for g in geoms))

    # 包围盒
    try:
        bounds = scene.bounds
        extent = bounds[1] - bounds[0]
        rec["extent_xyz"] = [round(float(x), 4) for x in extent]
        rec["max_extent"] = round(float(extent.max()), 4)
    except Exception:
        rec["extent_xyz"] = None

    rec["game_ready_guess"] = rec["faces"] < 10000
    return rec


def main():
    paths = []
    for d in SEARCH_DIRS:
        if not os.path.isdir(d):
            continue
        for ext in EXTS:
            paths.extend(glob.glob(os.path.join(d, ext)))
    paths = sorted(paths)

    print(f"[D-1] found {len(paths)} mesh files, inspecting...\n", flush=True)
    results = []
    for p in paths:
        print(f"  - {os.path.basename(p)}", flush=True)
        results.append(inspect_one(p))

    # 表格（纯 ASCII，避开 Windows GBK 编码坑）
    bar = "=" * 120
    print("\n" + bar, flush=True)
    hdr = f"{'file':<44} {'faces':>9} {'verts':>9} {'UV':>6} {'VColor':>6} {'TexMat':>6} {'tight':>5} {'maxExt':>7} {'MB':>6}"
    print(hdr, flush=True)
    print("-" * 120, flush=True)
    for r in results:
        if "error" in r:
            print(f"{os.path.basename(r['path']):<44}  ERR {r['error']}", flush=True)
            continue
        uv_tag = f"{r['uv_unique_points']}" if r["has_uv"] else "no"
        print(
            f"{os.path.basename(r['path']):<44} {human_count(r['faces']):>9} "
            f"{human_count(r['vertices']):>9} {uv_tag:>6} "
            f"{'Y' if r['has_vertex_colors'] else 'N':>6} {'Y' if r['has_texture_material'] else 'N':>6} "
            f"{'Y' if r['watertight'] else 'N':>5} {r.get('max_extent', 0):>7.2f} {r['size_mb']:>6.2f}",
            flush=True,
        )
    print(bar, flush=True)

    ok = [r for r in results if "error" not in r]
    if ok:
        faces = [r["faces"] for r in ok]
        print(f"\n[SUMMARY] valid models: {len(ok)}", flush=True)
        print(f"  faces: min {human_count(min(faces))} | max {human_count(max(faces))} | avg {human_count(sum(faces)//len(faces))}", flush=True)
        print(f"  with UV coords        : {sum(1 for r in ok if r['has_uv'])}/{len(ok)}", flush=True)
        print(f"  with vertex colors    : {sum(1 for r in ok if r['has_vertex_colors'])}/{len(ok)}", flush=True)
        print(f"  with texture material : {sum(1 for r in ok if r['has_texture_material'])}/{len(ok)}", flush=True)
        print(f"  watertight            : {sum(1 for r in ok if r['watertight'])}/{len(ok)}", flush=True)
        print(f"  already <10k faces    : {sum(1 for r in ok if r['game_ready_guess'])}/{len(ok)}", flush=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "d1_assessment.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVED] {out}", flush=True)


if __name__ == "__main__":
    main()
