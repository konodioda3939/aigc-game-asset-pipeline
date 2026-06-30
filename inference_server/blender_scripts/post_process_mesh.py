"""
bpy 脚本：3D 模型后处理（减面 + 展UV + LOD + 预览渲染）。

被 Blender headless 调用：
    blender --background --factory-startup --python post_process_mesh.py -- <args>

固化 D-1/D-2/D-3 已验证的流程（TripoSR 毛坯 → 游戏级 Mesh）。
所有关键 bpy 调用都已在前序步骤验证过，勿随意改动：
  - Decimate: ratio=目标/原始, COLLAPSE 模式
  - smart_project: 必须 EDIT 模式 + 全选面后调
  - export_colors=True 保留顶点色 COLOR_0
  - 节点匹配用 type=='BSDF_PRINCIPLED'（Blender 5.1 改了节点名）

参数（sys.argv 里 -- 之后，按位置）:
    [0] INPUT_GLB       输入 glb 绝对路径
    [1] OUT_DIR         输出目录绝对路径
    [2] TARGET_FACES    LOD0 目标面数
    [3] LOD_FACES       额外 LOD 面数，逗号分隔（如 "2500,800"），可空
    [4] UV_UNWRAP       "true"/"false" 是否展 UV
    [5] RENDER_PREVIEW  "true"/"false" 是否渲染预览图

输出（写到 OUT_DIR）:
    LOD0.glb, LOD1.glb, ...    各级模型
    preview_LOD0.png           预览图（可选）
    manifest.json              每级面数/顶点/UV/顶点色状态 + 错误信息
"""
import bpy
import sys
import json
import math
import traceback
from pathlib import Path

import mathutils

# ===== 参数解析 =====
argv = sys.argv
user_args = argv[argv.index("--") + 1:] if "--" in argv else []
if len(user_args) < 6:
    print("[BPY][ERROR] 参数不足，需要 6 个：INPUT_GLB OUT_DIR TARGET_FACES LOD_FACES UV_UNWRAP RENDER_PREVIEW", flush=True)
    sys.exit(2)

INPUT_GLB = user_args[0]
OUT_DIR = Path(user_args[1])
TARGET_FACES = int(user_args[2])
LOD_FACES = [int(x) for x in user_args[3].split(",") if x.strip()]
UV_UNWRAP = user_args[4].lower() == "true"
RENDER_PREVIEW = user_args[5].lower() == "true"

OUT_DIR.mkdir(parents=True, exist_ok=True)
manifest = {
    "input": INPUT_GLB,
    "blender_version": bpy.app.version_string,
    "target_faces": TARGET_FACES,
    "lod_faces": LOD_FACES,
    "uv_unwrap": UV_UNWRAP,
    "render_preview": RENDER_PREVIEW,
    "lod_levels": [],
    "status": None,
    "error": None,
}


def log(msg):
    print(f"[BPY] {msg}", flush=True)


def clear_scene():
    """清空默认场景（Cube/Camera/Light）+ 孤儿数据。"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for col in (bpy.data.meshes, bpy.data.materials, bpy.data.images,
                bpy.data.curves, bpy.data.cameras, bpy.data.lights):
        for block in list(col):
            col.remove(block)


def import_glb(path):
    """导入 glb，返回第一个 MESH 对象。"""
    bpy.ops.import_scene.gltf(filepath=path)
    meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    if not meshes:
        raise RuntimeError("glb 里没有 MESH 对象")
    return meshes[0]


def get_bsdf(nt):
    """按类型找 Principled BSDF（不按名字，Blender 5.1 改名了）。"""
    for n in nt.nodes:
        if n.type == 'BSDF_PRINCIPLED':
            return n
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    out = next((x for x in nt.nodes if x.type == 'OUTPUT_MATERIAL'), None)
    if out is None:
        out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(bsdf.outputs[0], out.inputs[0])
    return bsdf


def make_vertex_color_material(obj):
    """顶点色 -> Base Color 材质（用于预览渲染）。"""
    if len(obj.data.color_attributes) == 0:
        return None
    ca_name = obj.data.color_attributes[0].name
    mat = bpy.data.materials.new(f"VC_{obj.name}")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = get_bsdf(nt)
    try:
        vc = nt.nodes.new("ShaderNodeVertexColor")
        vc.layer_name = ca_name
    except Exception:
        vc = nt.nodes.new("ShaderNodeColorAttribute")
        vc.layer_name = ca_name
    nt.links.new(vc.outputs["Color"], bsdf.inputs["Base Color"])
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return mat


def decimate_to(src_obj, name, target_faces, original_faces):
    """从原始 mesh 复制一份，Decimate COLLAPSE 砍到 target_faces（D-2 验证）。"""
    obj = src_obj.copy()
    obj.data = src_obj.data.copy()
    obj.name = name
    bpy.context.collection.objects.link(obj)
    ratio = max(0.001, min(1.0, target_faces / original_faces))
    m = obj.modifiers.new("Decimate", type='DECIMATE')
    m.decimate_type = 'COLLAPSE'
    m.ratio = ratio
    m.use_collapse_triangulate = True
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier="Decimate")
    return obj


def smart_uv_unwrap(obj):
    """Smart UV Project 展 UV（D-3 验证：必须 EDIT 模式 + 全选面）。"""
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=1.15192, island_margin=0.02,
                             correct_aspect=True, scale_to_bounds=False)
    bpy.ops.object.mode_set(mode='OBJECT')


def export_glb(obj, out_path):
    """导出 glb（D-2/D-3 验证的参数，export_colors 保留顶点色）。"""
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    try:
        bpy.ops.export_scene.gltf(
            filepath=str(out_path), use_selection=True, export_format='GLB',
            export_colors=True, export_yup=True, export_apply=True)
    except TypeError:
        bpy.ops.export_scene.gltf(
            filepath=str(out_path), use_selection=True, export_format='GLB',
            export_yup=True, export_apply=True)


def setup_camera_and_light():
    bpy.ops.object.light_add(type='SUN', location=(3, -3, 4))
    bpy.context.object.data.energy = 4.0
    bpy.context.object.rotation_euler = (math.radians(50), math.radians(20), math.radians(30))
    bpy.ops.object.camera_add(location=(1.9, -1.9, 1.3))
    cam = bpy.context.object
    d = mathutils.Vector((0, 0, 0)) - cam.location
    cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    bpy.context.scene.camera = cam


def render_preview(obj, out_png):
    for o in bpy.data.objects:
        if o.type == 'MESH':
            o.hide_render = (o.name != obj.name)
    scene = bpy.context.scene
    try:
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
    except TypeError:
        scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.filepath = str(out_png)
    bpy.ops.render.render(write_still=True)
    for o in bpy.data.objects:
        if o.type == 'MESH':
            o.hide_render = False


# ===== 主流程 =====
try:
    clear_scene()
    src = import_glb(INPUT_GLB)
    src.name = "source"
    original_faces = len(src.data.polygons)
    manifest["original_faces"] = original_faces
    manifest["original_verts"] = len(src.data.vertices)
    manifest["has_vertex_color"] = len(src.data.color_attributes) > 0
    log(f"original: verts={len(src.data.vertices)} faces={original_faces} "
        f"color_attrs={len(src.data.color_attributes)}")

    # LOD 级：LOD0 = target_faces，其余按 LOD_FACES
    all_lods = [("LOD0", TARGET_FACES)] + [(f"LOD{i+1}", f) for i, f in enumerate(LOD_FACES)]

    for name, tgt in all_lods:
        obj = decimate_to(src, name, tgt, original_faces)
        has_uv = False
        if UV_UNWRAP:
            smart_uv_unwrap(obj)
            has_uv = len(obj.data.uv_layers) > 0
        out_glb = OUT_DIR / f"{name}.glb"
        export_glb(obj, out_glb)
        rec = {
            "name": name,
            "target_faces": tgt,
            "actual_faces": len(obj.data.polygons),
            "actual_verts": len(obj.data.vertices),
            "has_uv": has_uv,
            "has_vertex_color": len(obj.data.color_attributes) > 0,
            "file": out_glb.name,
            "size_kb": round(out_glb.stat().st_size / 1024, 1) if out_glb.exists() else 0,
        }
        manifest["lod_levels"].append(rec)
        log(f"{name}: target={tgt} actual={rec['actual_faces']} uv={has_uv} "
            f"vc={rec['has_vertex_color']} ({rec['size_kb']} KB)")

    # 预览：用 LOD0 渲染顶点色
    if RENDER_PREVIEW and "LOD0" in bpy.data.objects:
        lod0 = bpy.data.objects["LOD0"]
        make_vertex_color_material(lod0)
        setup_camera_and_light()
        lod0.location = (0, 0, 0)
        preview_path = OUT_DIR / "preview_LOD0.png"
        render_preview(lod0, preview_path)
        manifest["preview"] = preview_path.name
        log(f"preview rendered: {preview_path.name}")

    manifest["status"] = "ok"
except Exception as e:
    manifest["status"] = "error"
    manifest["error"] = f"{type(e).__name__}: {e}"
    manifest["traceback"] = traceback.format_exc()
    log(f"ERROR: {type(e).__name__}: {e}")
    log(traceback.format_exc())
finally:
    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log(f"manifest written: {OUT_DIR / 'manifest.json'}")
