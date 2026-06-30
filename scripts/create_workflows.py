"""
生成 4 个 ComfyUI 游戏美术工作流 JSON 文件。

每个工作流都设计为可通过 ComfyUI 的 Web 界面加载，
也可通过 /prompt API 提交。

工作流:
1. character_concept  — 角色概念图 (txt2img, 1024x576, turnaround sheet)
2. asset_icon_text     — 游戏素材-图标-纯文字 (txt2img + rembg)
3. model_3d           — 3D 模型 (image → TripoSR → .obj)
4. pbr_material       — PBR 材质 (prompt → 5 texture maps)
"""

import json
import random
import os

OUTPUT_DIR = r"d:\aigc-project\ComfyUI\workflows"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_seed():
    return random.randint(1, 2**31 - 1)


def save_workflow(name, workflow):
    path = os.path.join(OUTPUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(workflow, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {path}")


# ============================================================
# 工作流 1: 角色概念图 (Character Turnaround Sheet)
# ============================================================
def create_character_concept():
    """
    节点图:
      DiffusersLoader(Counterfeit-V2.5) → model, clip, vae
      CLIPTextEncode(正面) → positive conditioning
      CLIPTextEncode(负面) → negative conditioning
      EmptyLatentImage(1024x576) → latent
      KSampler(30步, cfg=8.5, euler/normal)
      VAEDecode → SaveImage
    """
    seed = make_seed()

    return {
        "1": {
            "class_type": "DiffusersLoader",
            "inputs": {
                "model_path": "Counterfeit-V2.5"
            },
            "_meta": {"title": "Load Counterfeit-V2.5"}
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": (
                    "masterpiece, best quality, character turnaround sheet, "
                    "1girl, genshin impact style, raiden shogun, "
                    "multiple views of the same character, "
                    "front view, side view, back view, three quarter view, "
                    "full body, standing, white background, "
                    "character design sheet, reference sheet, "
                    "same character in all views, consistent design, "
                    "detailed character reference, game character concept art"
                ),
                "clip": ["1", 1]
            },
            "_meta": {"title": "Positive Prompt"}
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": (
                    "different characters, inconsistent design, merged views, "
                    "overlapping views, cropped views, missing views, "
                    "out of frame, close-up, portrait, face only, "
                    "upper body only, messy layout, text labels, watermark, "
                    "lowres, bad anatomy, bad hands, worst quality, "
                    "blurry, ugly, deformed, distorted body, bad proportions, "
                    "extra limbs, fused limbs, too many fingers, disfigured, "
                    "mutated, doll, plastic, 3D render, photo, realistic"
                ),
                "clip": ["1", 1]
            },
            "_meta": {"title": "Negative Prompt"}
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": 1024,
                "height": 576,
                "batch_size": 1
            },
            "_meta": {"title": "1024x576 Latent"}
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "seed": seed,
                "steps": 30,
                "cfg": 8.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0
            },
            "_meta": {"title": "KSampler"}
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["5", 0],
                "vae": ["1", 2]
            },
            "_meta": {"title": "VAE Decode"}
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "CharacterConcept",
                "images": ["6", 0]
            },
            "_meta": {"title": "Save Image"}
        }
    }


# ============================================================
# 工作流 2: 游戏素材-图标-纯文字 (txt2img + rembg)
# ============================================================
def create_asset_icon_text():
    seed = make_seed()

    return {
        "1": {
            "class_type": "DiffusersLoader",
            "inputs": {"model_path": "Counterfeit-V2.5"},
            "_meta": {"title": "Load Counterfeit-V2.5"}
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": (
                    "{prompt}, game icon, centered composition, clean design, "
                    "isolated, high contrast, polished, vibrant colors, "
                    "simple shapes, game asset, masterpiece, best quality"
                ),
                "clip": ["1", 1]
            },
            "_meta": {"title": "Positive (edit prompt here)"}
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": (
                    "realistic, photo, 3D render, complex background, "
                    "cluttered, lowres, worst quality, blurry, ugly, "
                    "jpeg artifacts, signature, watermark"
                ),
                "clip": ["1", 1]
            },
            "_meta": {"title": "Negative Prompt"}
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 512, "height": 512, "batch_size": 1},
            "_meta": {"title": "512x512 Latent"}
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "seed": seed,
                "steps": 25,
                "cfg": 7.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0
            },
            "_meta": {"title": "KSampler"}
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["5", 0],
                "vae": ["1", 2]
            },
            "_meta": {"title": "VAE Decode"}
        },
        "7": {
            "class_type": "RemoveBackground",
            "inputs": {"images": ["6", 0]},
            "_meta": {"title": "Remove Background (rembg)"}
        },
        "8": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "AssetIcon",
                "images": ["7", 0]
            },
            "_meta": {"title": "Save Image"}
        }
    }


# ============================================================
# 工作流 3: 3D 模型生成 (TripoSR)
# ============================================================
def create_model_3d():
    return {
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": "example.png"},
            "_meta": {"title": "Load Reference Image"}
        },
        "2": {
            "class_type": "TripoSRModelLoader",
            "inputs": {
                "model": "TripoSR_model.ckpt",
                "chunk_size": 8192
            },
            "_meta": {"title": "Load TripoSR Model"}
        },
        "3": {
            "class_type": "TripoSRSampler",
            "inputs": {
                "model": ["2", 0],
                "reference_image": ["1", 0],
                "geometry_resolution": 256,
                "threshold": 25.0
            },
            "_meta": {"title": "TripoSR Sampler"}
        },
        "4": {
            "class_type": "TripoSRViewer",
            "inputs": {
                "mesh": ["3", 0]
            },
            "_meta": {"title": "Save 3D Mesh (.obj)"}
        }
    }


# ============================================================
# 工作流 4: PBR 材质生成 (StableMaterials)
# ============================================================
def create_pbr_material():
    seed = make_seed()

    return {
        "1": {
            "class_type": "StableMaterials",
            "inputs": {
                "prompt": "rough stone wall, weathered, natural",
                "steps": 25,
                "guidance_scale": 10.0,
                "tileable": True,
                "seed": seed
            },
            "_meta": {"title": "StableMaterials PBR Generator"}
        },
        "2": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "PBR_BaseColor",
                "images": ["1", 0]
            },
            "_meta": {"title": "Save BaseColor"}
        },
        "3": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "PBR_Normal",
                "images": ["1", 1]
            },
            "_meta": {"title": "Save Normal"}
        },
        "4": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "PBR_Height",
                "images": ["1", 2]
            },
            "_meta": {"title": "Save Height"}
        },
        "5": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "PBR_Roughness",
                "images": ["1", 3]
            },
            "_meta": {"title": "Save Roughness"}
        },
        "6": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "PBR_Metallic",
                "images": ["1", 4]
            },
            "_meta": {"title": "Save Metallic"}
        }
    }


if __name__ == "__main__":
    print("Creating ComfyUI workflow JSONs...")
    print()

    save_workflow("character_concept.json", create_character_concept())
    save_workflow("asset_icon_text.json", create_asset_icon_text())
    save_workflow("model_3d.json", create_model_3d())
    save_workflow("pbr_material.json", create_pbr_material())

    print()
    print("Done! 4 workflow JSONs created in:", OUTPUT_DIR)
    print()
    print("Usage:")
    print("  1. Open ComfyUI: http://127.0.0.1:8188")
    print("  2. Drag a workflow .json file into the browser window")
    print("  3. Edit prompts/parameters as needed")
    print("  4. Click 'Queue Prompt' to generate")
