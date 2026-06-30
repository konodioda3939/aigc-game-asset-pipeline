"""解析 glb 的 JSON 头，看每个 primitive 的属性（COLOR_0 顶点色 / TEXCOORD_0 UV）和材质。

用法：
    python check_glb_attrs.py <file1.glb> [file2.glb ...]
纯只读，验证导出文件的真实内容。
"""
import struct
import json
import sys
import os


def read_glb_json(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"glTF", f"not a glb: {magic}"
        f.read(4)  # version
        f.read(4)  # total length
        chunk_len = struct.unpack("<I", f.read(4))[0]
        f.read(4)  # chunk type (JSON)
        return json.loads(f.read(chunk_len).decode("utf-8"))


def main():
    paths = sys.argv[1:]
    if not paths:
        print("usage: check_glb_attrs.py <glb files>", flush=True)
        return
    for path in paths:
        d = read_glb_json(path)
        print(f"== {os.path.basename(path)}  ({os.path.getsize(path)/1024:.0f} KB)", flush=True)
        for i, mesh in enumerate(d.get("meshes", [])):
            for j, prim in enumerate(mesh["primitives"]):
                attrs = prim.get("attributes", {})
                keys = sorted(attrs.keys())
                print(f"   mesh[{i}].prim[{j}]: attrs={keys}", flush=True)
                print(f"      -> COLOR_0(vertex color)={'YES' if 'COLOR_0' in attrs else 'NO'}  "
                      f"TEXCOORD_0(UV)={'YES' if 'TEXCOORD_0' in attrs else 'NO'}  "
                      f"material={prim.get('material')}", flush=True)
        for i, mat in enumerate(d.get("materials", [])):
            has_tex = False
            pbr = mat.get("pbrMetallicRoughness", {})
            bct = pbr.get("baseColorTexture")
            has_tex = bct is not None
            print(f"   material[{i}] '{mat.get('name')}': baseColorTexture={has_tex} "
                  f"baseColorFactor={pbr.get('baseColorFactor')}", flush=True)
        print("", flush=True)


if __name__ == "__main__":
    main()
