import os, sys
from pathlib import Path
from PIL import Image

# 强制UTF-8输出,避免conda GBK编码报错
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

image_dir = Path(r"D:\aigc-project\data\style_images")
valid_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

if not image_dir.exists():
    print(f"[ERROR] Directory not found: {image_dir}")
    print(f"Please create it and put images inside.")
    exit(1)

images = [f for f in image_dir.iterdir() if f.suffix.lower() in valid_extensions]
other = [f for f in image_dir.iterdir() if f.suffix.lower() not in valid_extensions and f.is_file()]

print(f"Total image files: {len(images)}")
if other:
    print(f"Non-image files: {[f.name for f in other]}")

if len(images) == 0:
    print("[ERROR] No images found!")
    exit(1)

min_res = 512
issues = []
ok_res = 0
ok_count = 0

print(f"\n{'Filename':<50} {'Resolution':<12} {'Size':>10}  Status")
print("-" * 90)

for img_path in sorted(images):
    try:
        with Image.open(img_path) as img:
            w, h = img.size
            file_size_kb = os.path.getsize(img_path) / 1024
            status = "OK"

            if w < min_res or h < min_res:
                status = f"LOW-RES ({w}x{h})"
                issues.append((img_path.name, f"{w}x{h}", "Resolution too low"))
            else:
                ok_res += 1

            if file_size_kb < 10:
                status = f"SMALL ({file_size_kb:.0f}KB)"
                issues.append((img_path.name, f"{file_size_kb:.0f}KB", "File too small"))

            print(f"  {img_path.name:<48} {w}x{h:<10} {file_size_kb:>8.1f} KB  {status}")
            ok_count += 1
    except Exception as e:
        print(f"  {img_path.name:<48} {'ERROR':<12} {'---':>10}  {str(e)[:40]}")
        issues.append((img_path.name, str(e)[:40], "Cannot open"))

print(f"\n{'='*60}")
print(f"Summary: {ok_count} readable, {ok_res} pass resolution (>= {min_res}x{min_res})")
if issues:
    print(f"\n[WARNING] {len(issues)} issue(s):")
    for name, detail, reason in issues:
        print(f"  - {name}: {reason} ({detail})")
else:
    print("All images passed check!")

# Format breakdown
formats = {}
for img_path in images:
    fmt = img_path.suffix.lower()
    formats[fmt] = formats.get(fmt, 0) + 1
print(f"\nFormats: {formats}")
