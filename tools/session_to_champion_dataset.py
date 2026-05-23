"""
Copy session minimap captures into dataset_champion with a train/val split.

Champion training (yolo_champion.py) is single-class: ally (0) and enemy (1)
boxes from session labels become class 0; teleport/recall/map lines are dropped.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SESSION_IMG = ROOT / "session" / "images"
SESSION_LBL = ROOT / "session" / "labels"
OUT = ROOT / "dataset_champion"
# Minimap label ids from tools/annotation_tool: ally=0, enemy=1
CHAMP_ICON_CLASSES = {0, 1}


def _convert_label(text: str) -> str:
    lines_out: list[str] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        if cls not in CHAMP_ICON_CLASSES:
            continue
        lines_out.append(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
    return "\n".join(lines_out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val", type=int, default=12, help="validation image count")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for split")
    p.add_argument(
        "--images-dir",
        type=Path,
        default=SESSION_IMG,
        help="source images folder (default: session/images)",
    )
    p.add_argument(
        "--labels-dir",
        type=Path,
        default=SESSION_LBL,
        help="source labels folder (default: session/labels)",
    )
    args = p.parse_args()

    img_dir = args.images_dir.resolve()
    lbl_dir = args.labels_dir.resolve()
    imgs = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    if not imgs:
        raise SystemExit(f"No images under {img_dir}")

    n = len(imgs)
    if args.val >= n:
        raise SystemExit(f"--val ({args.val}) must be < image count ({n})")

    rng = random.Random(args.seed)
    stems = list(imgs)
    rng.shuffle(stems)
    val_set = {s.stem for s in stems[: args.val]}
    train_set = {s.stem for s in stems[args.val :]}

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        d = OUT / sub
        d.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            if f.is_file():
                f.unlink()

    n_train = n_val = 0
    for src in sorted(imgs, key=lambda x: x.name):
        stem = src.stem
        is_val = stem in val_set
        split = "val" if is_val else "train"
        shutil.copy2(src, OUT / "images" / split / src.name)
        lbl_src = lbl_dir / f"{stem}.txt"
        text = lbl_src.read_text(encoding="utf-8", errors="replace") if lbl_src.exists() else ""
        converted = _convert_label(text)
        (OUT / "labels" / split / f"{stem}.txt").write_text(
            converted + ("\n" if converted else ""), encoding="utf-8"
        )
        if is_val:
            n_val += 1
        else:
            n_train += 1

    print(f"Images: {n}  ->  train={n_train}  val={n_val}  (seed={args.seed})")
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
