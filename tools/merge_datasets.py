"""
merge_datasets.py — Merge dataset_minimap (4 classes) + dataset_champion (1 class)
into dataset_combined (5 classes).

Class mapping:
  dataset_minimap:  ally(0)->0, enemy(1)->1, teleport(2)->2, recall(3)->3  (unchanged)
  dataset_champion: champion_icon(0) -> 4

Usage:
  python tools/merge_datasets.py
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent

SRC_MINIMAP   = ROOT / "dataset_minimap"
SRC_CHAMPION  = ROOT / "dataset_champion"
DST           = ROOT / "dataset_combined"

YAML = """\
path: {path}
train: images/train
val:   images/val

nc: 5
names:
  - ally
  - enemy
  - teleport
  - recall
  - champion_icon
"""


def remap_label(src_txt: Path, dst_txt: Path, class_offset: int) -> None:
    lines = src_txt.read_text().splitlines()
    out = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        parts[0] = str(int(parts[0]) + class_offset)
        out.append(" ".join(parts))
    dst_txt.write_text("\n".join(out) + "\n" if out else "")


def merge(split: str) -> None:
    dst_img = DST / "images" / split
    dst_lbl = DST / "labels" / split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    # --- minimap (no remap needed) ---
    src_img_dir = SRC_MINIMAP / "images" / split
    src_lbl_dir = SRC_MINIMAP / "labels" / split
    for img in src_img_dir.glob("*"):
        shutil.copy2(img, dst_img / f"mm_{img.name}")
        lbl = src_lbl_dir / (img.stem + ".txt")
        if lbl.exists():
            shutil.copy2(lbl, dst_lbl / f"mm_{img.stem}.txt")

    # --- champion (remap class 0 -> 4) ---
    src_img_dir = SRC_CHAMPION / "images" / split
    src_lbl_dir = SRC_CHAMPION / "labels" / split
    for img in src_img_dir.glob("*"):
        shutil.copy2(img, dst_img / f"ch_{img.name}")
        lbl = src_lbl_dir / (img.stem + ".txt")
        if lbl.exists():
            remap_label(lbl, dst_lbl / f"ch_{img.stem}.txt", class_offset=4)

    imgs = list(dst_img.glob("*"))
    lbls = list(dst_lbl.glob("*"))
    print(f"  {split}: {len(imgs)} images, {len(lbls)} labels")


if __name__ == "__main__":
    print("Building dataset_combined ...")
    for split in ("train", "val"):
        merge(split)

    yaml_path = DST / "dataset.yaml"
    yaml_path.write_text(YAML.format(path=str(DST).replace("\\", "/")))
    print(f"Wrote {yaml_path}")
    print("Done.")
