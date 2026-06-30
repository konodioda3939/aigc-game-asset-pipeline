"""
LoRA 格式转换: PEFT format -> ComfyUI format

PEFT 键名格式:
  base_model.model.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.lora_A.weight
  base_model.model.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.lora_B.weight

ComfyUI 键名格式:
  lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight
  lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k.lora_up.weight

映射规则:
  - PEFT lora_A -> ComfyUI lora_down
  - PEFT lora_B -> ComfyUI lora_up
  - base_model.model. 前缀去掉
  - 路径分隔符 . 替换为 _
  - 最后补 lora_down.weight / lora_up.weight
"""

import sys
import os
from safetensors.torch import load_file, save_file

# ===== 配置 =====
PEFT_PATH = r"d:\aigc-project\lora_output\adapter_model.safetensors"
OUTPUT_DIR = r"d:\aigc-project\ComfyUI\models\loras"
OUTPUT_NAME = "CounterfeitGenshin-LoRA.safetensors"


def convert_lora_peft_to_comfyui(input_path: str, output_path: str):
    """
    将 PEFT 格式的 LoRA 权重转换为 ComfyUI 兼容格式。

    PEFT 键名结构:
      base_model.model.<unet_path>.<module>.lora_<A/B>.weight

    ComfyUI 键名结构:
      lora_unet_<unet_path_with_underscores>_<module>.<lora_down/lora_up>.weight
    """
    print(f"Loading PEFT LoRA from: {input_path}")
    state_dict = load_file(input_path)
    print(f"Total keys: {len(state_dict)}")

    new_dict = {}
    converted = 0
    skipped = 0

    for key, tensor in state_dict.items():
        # 只处理 UNet 的 LoRA 权重
        if "base_model.model." not in key:
            print(f"  SKIP (no prefix match): {key}")
            skipped += 1
            continue

        # 去掉 base_model.model. 前缀
        trimmed = key.replace("base_model.model.", "", 1)

        # 判断 lora_A 还是 lora_B
        if ".lora_A." in trimmed:
            trimmed = trimmed.replace(".lora_A.weight", "")
            weight_type = "lora_down"
        elif ".lora_B." in trimmed:
            trimmed = trimmed.replace(".lora_B.weight", "")
            weight_type = "lora_up"
        else:
            print(f"  SKIP (no lora_A/B): {key}")
            skipped += 1
            continue

        # 将 . 替换为 _   (unet 路径中的 . 分隔符)
        # 但保留最后的模块名（如 to_k, to_v 等）
        # 例如: down_blocks.0.attentions.0...attn1.to_k
        #   -> down_blocks_0_attentions_0...attn1_to_k
        trimmed_underscored = trimmed.replace(".", "_")

        # 构建 ComfyUI 格式键名
        new_key = f"lora_unet_{trimmed_underscored}.{weight_type}.weight"

        # 验证形状
        new_dict[new_key] = tensor
        converted += 1

    print(f"\nConverted: {converted} keys")
    print(f"Skipped: {skipped} keys")

    if converted == 0:
        print("\nERROR: No keys were converted! Check the input file format.")
        print("First 5 keys in input:")
        for k in list(state_dict.keys())[:5]:
            print(f"  {k}")
        sys.exit(1)

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_file(new_dict, output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nSaved to: {output_path}")
    print(f"File size: {file_size_mb:.1f} MB")

    # 打印几个示例键名供验证
    print("\nSample converted keys:")
    for k in list(new_dict.keys())[:5]:
        print(f"  {k}: {list(new_dict[k].shape)}")

    return output_path


if __name__ == "__main__":
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    convert_lora_peft_to_comfyui(PEFT_PATH, output_path)
    print("\nDone! LoRA ready for ComfyUI.")
    print(f"Place in: {output_path}")
    print("Use 'LoraLoader' node in ComfyUI with strength 0.7-1.0")
