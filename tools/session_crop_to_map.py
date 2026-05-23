"""
Crop session minimap screenshots to the YOLO "map" ROI (class 4) and rewrite labels.

Everything outside the union of all `map` boxes is removed. Other classes are clipped
to the crop and re-normalized. Map lines are omitted (the image is the map region).

Defaults: session/images + session/labels -> session_cropped/images + labels.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image

MAP_CLASS = 4  # ally=0, enemy=1, teleport=2, recall=3, map=4


def _yolo_to_px(
    cx: float, cy: float, bw: float, bh: float, w: int, h: int
) -> tuple[float, float, float, float]:
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    return x1, y1, x2, y2


def _px_to_yolo(
    x1: float, y1: float, x2: float, y2: float, cw: int, ch: int
) -> tuple[float, float, float, float]:
    if cw <= 0 or ch <= 0:
        raise ValueError("crop dimensions must be positive")
    cx = ((x1 + x2) / 2) / cw
    cy = ((y1 + y2) / 2) / ch
    bw = abs(x2 - x1) / cw
    bh = abs(y2 - y1) / ch
    return cx, cy, bw, bh


def _union_map_roi(lines: list[str], w: int, h: int) -> tuple[int, int, int, int] | None:
    """Return integer PIL crop box (l, t, r, b) as union of all map boxes, or None."""
    xs1 = ys1 = math.inf
    xs2 = ys2 = -math.inf
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        if cls != MAP_CLASS:
            continue
        cx, cy, bw, bh = map(float, parts[1:])
        x1, y1, x2, y2 = _yolo_to_px(cx, cy, bw, bh, w, h)
        xs1 = min(xs1, x1)
        ys1 = min(ys1, y1)
        xs2 = max(xs2, x2)
        ys2 = max(ys2, y2)
    if xs1 == math.inf:
        return None
    l = max(0, int(math.floor(xs1)))
    t = max(0, int(math.floor(ys1)))
    r = min(w, int(math.ceil(xs2)))
    b = min(h, int(math.ceil(y2)))
    if r <= l or b <= t:
        return None
    return l, t, r, b


def _clip_box(
    x1: float, y1: float, x2: float, y2: float, l: int, t: int, r: int, b: int
) -> tuple[float, float, float, float] | None:
    ix1 = max(x1, float(l))
    iy1 = max(y1, float(t))
    ix2 = min(x2, float(r))
    iy2 = min(y2, float(b))
    if ix2 <= ix1 + 1e-6 or iy2 <= iy1 + 1e-6:
        return None
    return ix1, iy1, ix2, iy2


def _transform_labels(
    lines: list[str],
    iw: int,
    ih: int,
    crop: tuple[int, int, int, int],
    min_rel_side: float,
) -> list[str]:
    l, t, r, b = crop
    cw, ch = r - l, b - t
    out: list[str] = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        if cls == MAP_CLASS:
            continue
        cx, cy, bw, bh = map(float, parts[1:])
        x1, y1, x2, y2 = _yolo_to_px(cx, cy, bw, bh, iw, ih)
        clipped = _clip_box(x1, y1, x2, y2, l, t, r, b)
        if clipped is None:
            continue
        cx1, cy1, cx2, cy2 = clipped
        ncx, ncy, nbw, nbh = _px_to_yolo(cx1, cy1, cx2, cy2, cw, ch)
        if nbw < min_rel_side or nbh < min_rel_side:
            continue
        ncx = max(0.0, min(1.0, ncx))
        ncy = max(0.0, min(1.0, ncy))
        nbw = max(min_rel_side, min(1.0, nbw))
        nbh = max(min_rel_side, min(1.0, nbh))
        out.append(f"{cls} {ncx:.6f} {ncy:.6f} {nbw:.6f} {nbh:.6f}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    p.add_argument(
        "--images",
        type=Path,
        default=root / "session" / "images",
        help="input images directory",
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=root / "session" / "labels",
        help="input labels directory",
    )
    p.add_argument(
        "--out-images",
        type=Path,
        default=root / "session_cropped" / "images",
        help="output images directory",
    )
    p.add_argument(
        "--out-labels",
        type=Path,
        default=root / "session_cropped" / "labels",
        help="output labels directory",
    )
    p.add_argument(
        "--min-rel-side",
        type=float,
        default=1e-4,
        help="drop clipped boxes smaller than this (fraction of crop width/height)",
    )
    args = p.parse_args()

    args.out_images.mkdir(parents=True, exist_ok=True)
    args.out_labels.mkdir(parents=True, exist_ok=True)

    imgs = sorted(args.images.glob("*.png")) + sorted(args.images.glob("*.jpg"))
    n_ok = n_skip = 0
    skip_reasons: list[str] = []

    for img_path in imgs:
        stem = img_path.stem
        lbl_path = args.labels / f"{stem}.txt"
        text = (
            lbl_path.read_text(encoding="utf-8", errors="replace")
            if lbl_path.exists()
            else ""
        )
        lines = [ln for ln in text.splitlines() if ln.strip()]

        with Image.open(img_path) as im:
            im = im.convert("RGB")
            iw, ih = im.size
            crop = _union_map_roi(lines, iw, ih)
            if crop is None:
                n_skip += 1
                skip_reasons.append(f"{stem}: no map (class {MAP_CLASS}) box")
                continue
            l, t, r, b = crop
            cropped = im.crop((l, t, r, b))

        out_lines = _transform_labels(lines, iw, ih, crop, args.min_rel_side)
        out_img = args.out_images / img_path.name
        cropped.save(out_img, optimize=True)
        (args.out_labels / f"{stem}.txt").write_text(
            "\n".join(out_lines) + ("\n" if out_lines else ""),
            encoding="utf-8",
        )
        n_ok += 1

    print(f"Cropped: {n_ok}  skipped (no map label): {n_skip}")
    if skip_reasons and n_skip <= 20:
        for s in skip_reasons:
            print(f"  {s}")
    elif skip_reasons:
        for s in skip_reasons[:10]:
            print(f"  {s}")
        print(f"  ... and {n_skip - 10} more")

    print(f"Images -> {args.out_images}")
    print(f"Labels -> {args.out_labels}")


if __name__ == "__main__":
    main()
