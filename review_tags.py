"""标注审查工具：打印所有图片的tag，便于快速浏览修正"""
import os, sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path

IMAGE_DIR = Path(r"D:\aigc-project\data\processed")
txts = sorted(IMAGE_DIR.glob("*.txt"))

for i, txt_path in enumerate(txts):
    content = txt_path.read_text(encoding='utf-8').strip()
    tags = [t.strip() for t in content.split(",") if t.strip()]
    print(f"[{i+1:2d}] {txt_path.stem[:55]}")
    print(f"     ({len(tags)} tags) {', '.join(tags)}")
    print()
