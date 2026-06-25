"""批量修正标注：全局删除/替换无用标签"""
import os, sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path

IMAGE_DIR = Path(r"D:\aigc-project\data\processed")

# ===== 配置：在此编辑修正规则 =====
REMOVE_TAGS = {
    # 全局删除这些标签（Danbooru元标签，对训练无用）
    "sensitive",      # 内容分级标签
    "general",        # 兜底标签，无实际意义
    "questionable",   # 内容分级标签
    "explicit",        # 内容分级标签
}

# 如果你想全局加某个标签（谨慎使用）
ADD_TAGS = set()  # 例如: {"anime_coloring", "genshin_impact"}
# =================================

txts = sorted(IMAGE_DIR.glob("*.txt"))
changed = 0

for txt_path in txts:
    content = txt_path.read_text(encoding='utf-8').strip()
    tags = [t.strip() for t in content.split(",") if t.strip()]
    original = set(tags)

    # 删除
    tags = [t for t in tags if t not in REMOVE_TAGS]

    # 添加
    for add_tag in ADD_TAGS:
        if add_tag not in tags:
            tags.append(add_tag)

    new_set = set(tags)
    if original != new_set:
        txt_path.write_text(", ".join(tags), encoding='utf-8')
        # 显示变更
        removed = original - new_set
        added = new_set - original
        if removed:
            print(f"  [{txt_path.stem[:45]}] -{', '.join(sorted(removed))}")
        if added:
            print(f"  [{txt_path.stem[:45]}] +{', '.join(sorted(added))}")
        changed += 1

print(f"\nDone! Changed {changed}/{len(txts)} files.")
if changed == 0:
    print("No changes made. Edit REMOVE_TAGS/ADD_TAGS in this script.")
