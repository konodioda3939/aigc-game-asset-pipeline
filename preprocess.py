"""数据预处理：将所有图片中心裁切为512x512，适配SD 1.5训练"""
import os, sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
from PIL import Image

SRC_DIR = Path(r"D:\aigc-project\data\style_images")
DST_DIR = Path(r"D:\aigc-project\data\processed")
DST_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = 512

valid_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
images = [f for f in SRC_DIR.iterdir() if f.suffix.lower() in valid_exts]

print(f"Processing {len(images)} images...")
print(f"Target: {TARGET_SIZE}x{TARGET_SIZE} center crop\n")

success = 0
for img_path in sorted(images):
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            w, h = img.size

            # 中心裁切为正方形
            if w > h:
                left = (w - h) // 2
                img = img.crop((left, 0, left + h, h))
            else:
                top = (h - w) // 2
                img = img.crop((0, top, w, top + w))

            # Resize到512x512
            img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

            # 保存
            out_name = img_path.stem + ".png"
            out_path = DST_DIR / out_name
            img.save(out_path, "PNG")
            success += 1
            print(f"  [{success:2d}] {img_path.name[:50]:<50} -> {out_name}")

    except Exception as e:
        print(f"  [FAIL] {img_path.name}: {e}")

print(f"\nDone! {success} images saved to {DST_DIR}")
print(f"Next step: run captioning script")
