"""
Synthesize teleport training images by:
  1. Cropping the teleport beacon from high-confidence frames (the real red/pink swirl).
  2. Randomly rescaling the crop (simulate early-phase / far-away sizes).
  3. Partially occluding it (cover 0-40% of one edge with the background).
  4. Pasting onto clean background frames from the recording pool.

This creates diverse training examples that teach the model to detect
teleport beams at varying sizes and partial visibility.
"""
import torch, cv2, numpy as np, random, shutil
from pathlib import Path

_orig = torch.load
def _p(*a, **k):
    k["weights_only"] = False
    return _orig(*a, **k)
torch.load = _p

random.seed(7)
np.random.seed(7)

_CP3    = Path(__file__).resolve().parent.parent
_REPO   = _CP3.parent   # repo root — needed for recordings/ which is not in cp3

IMG_DIR = _CP3 / "session_teleport/images"
LBL_DIR = _CP3 / "session_teleport/labels"
BG_DIR  = _REPO / "recordings/20260705_004828"   # background frames (not in cp3)
OUT_IMG = IMG_DIR
OUT_LBL = LBL_DIR

NAMES  = ["ally", "enemy", "teleport", "recall", "champion_icon"]
TP_IDX = NAMES.index("teleport")

# ── source frames (high-confidence teleport beams) ───────────────────────────
SOURCE_STEMS = [
    "20260626_034124__frame_000053",
    "20260626_180924__frame_000032",
    "20260626_180924__frame_000022",
    "20260626_034124__frame_000063",
    "20260626_034124__frame_000046",
    "20260626_034124__frame_000025",
    "20260626_000819__frame_001204",
    "20260626_180924__frame_000027",
    "20260626_034124__frame_000008",
    "20260626_034124__frame_000065",
    "20260626_034124__frame_000073",
    "20260626_180924__frame_000033",
    "20260626_013713__frame_000022",
    "20260626_180924__frame_000013",
    "frame_000483",
    "20260626_180924__frame_000042",
    "20260626_034124__frame_000021",
    "20260626_034124__frame_000064",
    "20260626_034124__frame_000099",
    "20260626_020910__frame_000025",
]

SYNTH_PER_SOURCE = 4   # synthetic images per regular source crop
SYNTH_PER_HARD   = 8   # synthetic images per hard source crop (overlapping / faint cases)
SCALE_RANGE      = (0.4, 1.1)  # rescale the crop to this fraction of original size
OCCLUDE_FRAC     = 0.4         # max fraction of crop edge to black out

# Hard frames: teleport beacon overlapping with / buried under champion icons
HARD_STEMS = [
    "frame_000225",
    "frame_000334",
    "frame_000924",
    "20260626_180924__frame_000015",
    # Round 2: still-missing after first pass
    "frame_000276",
    "frame_000778",
    "frame_000779",
]


def load_tp_crop(img_path: Path, lbl_path: Path):
    """Return list of (crop_bgr, norm_box) for each teleport label in the image."""
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    h, w = img.shape[:2]
    results = []
    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if not parts or int(parts[0]) != TP_IDX:
            continue
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = max(0, int((cx - bw / 2) * w))
        y1 = max(0, int((cy - bh / 2) * h))
        x2 = min(w,  int((cx + bw / 2) * w))
        y2 = min(h,  int((cy + bh / 2) * h))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img[y1:y2, x1:x2].copy()
        # Store the original box size relative to the full image
        results.append((crop, (cx, cy, bw, bh), (h, w)))
    return results


def partial_occlude(crop: np.ndarray, frac: float) -> np.ndarray:
    """Black out a random edge strip of the crop (at most `frac` of its width/height)."""
    out = crop.copy()
    ch, cw = out.shape[:2]
    side = random.choice(["left", "right", "top", "bottom"])
    amount = int(random.uniform(0.05, frac) * (cw if side in ("left", "right") else ch))
    if side == "left":
        out[:, :amount] = 0
    elif side == "right":
        out[:, max(0, cw - amount):] = 0
    elif side == "top":
        out[:amount, :] = 0
    else:
        out[max(0, ch - amount):, :] = 0
    return out


def paste_crop(bg: np.ndarray, crop: np.ndarray, cx_norm: float, cy_norm: float):
    """
    Paste `crop` onto `bg` centred at (cx_norm, cy_norm) in normalised coords.
    Returns updated bg and the actual normalised bounding box (cx, cy, bw, bh).
    """
    H, W = bg.shape[:2]
    ch, cw = crop.shape[:2]
    cx_px = int(cx_norm * W)
    cy_px = int(cy_norm * H)
    x1 = cx_px - cw // 2
    y1 = cy_px - ch // 2
    x2 = x1 + cw
    y2 = y1 + ch

    # Clamp to image
    sx1, sy1 = max(0, -x1), max(0, -y1)
    dx1, dy1 = max(0, x1), max(0, y1)
    dx2, dy2 = min(W, x2), min(H, y2)
    cx_slice = crop[sy1:sy1 + (dy2 - dy1), sx1:sx1 + (dx2 - dx1)]
    if cx_slice.size == 0:
        return bg, None

    bg[dy1:dy2, dx1:dx2] = cx_slice

    # Actual box in normalised coords
    act_cx = (dx1 + dx2) / 2 / W
    act_cy = (dy1 + dy2) / 2 / H
    act_bw = (dx2 - dx1) / W
    act_bh = (dy2 - dy1) / H
    return bg, (act_cx, act_cy, act_bw, act_bh)


# ── load background frames ──────────────────────────────────────────────────
tp_stems_set = {s for s in SOURCE_STEMS}
# Also exclude frames that already have teleport/recall labels in session_teleport
existing_tp_stems = set()
for lp in LBL_DIR.glob("*.txt"):
    if any(int(l.split()[0]) in (TP_IDX, NAMES.index("recall"))
           for l in lp.read_text().splitlines() if l.strip()):
        existing_tp_stems.add(lp.stem)

def _is_map_frame(path: Path) -> bool:
    """Reject frames that are mostly dark (UI screenshots, not minimap)."""
    img = cv2.imread(str(path))
    if img is None:
        return False
    # Minimap frames have significant colour variation and visible greens/browns.
    # A plain UI/desktop screenshot tends to be very dark or low-saturation.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat_mean = hsv[:, :, 1].mean()
    val_mean = hsv[:, :, 2].mean()
    return sat_mean > 30 and val_mean > 40   # discard dark / desaturated non-map frames

bg_paths = [p for p in sorted(BG_DIR.glob("*.png"))
            if p.stem not in tp_stems_set
            and p.stem not in existing_tp_stems
            and _is_map_frame(p)]
if not bg_paths:
    bg_paths = sorted(BG_DIR.glob("*.png"))
print(f"Background pool: {len(bg_paths)} map frames")

# ── remove old v2 synths ────────────────────────────────────────────────────
removed = 0
for p in list(OUT_IMG.glob("synthv2_*.png")) + list(OUT_LBL.glob("synthv2_*.txt")):
    p.unlink()
    removed += 1
if removed:
    print(f"Removed {removed} old synthv2 files")

# ── generate ────────────────────────────────────────────────────────────────
generated = 0
for stem in SOURCE_STEMS + HARD_STEMS:
    img_path = IMG_DIR / (stem + ".png")
    lbl_path = LBL_DIR / (stem + ".txt")
    if not img_path.exists() or not lbl_path.exists():
        print(f"  skip (missing): {stem}")
        continue

    crops = load_tp_crop(img_path, lbl_path)
    if not crops:
        continue

    n_synth = SYNTH_PER_HARD if stem in HARD_STEMS else SYNTH_PER_SOURCE
    for ci, (crop, (orig_cx, orig_cy, orig_bw, orig_bh), (src_h, src_w)) in enumerate(crops):
        for si in range(n_synth):
            # 1. Scale the crop
            scale = random.uniform(*SCALE_RANGE)
            new_h = max(8, int(crop.shape[0] * scale))
            new_w = max(8, int(crop.shape[1] * scale))
            scaled = cv2.resize(crop, (new_w, new_h),
                                interpolation=cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA)

            # 2. Optionally occlude
            if random.random() < 0.6:
                scaled = partial_occlude(scaled, OCCLUDE_FRAC)

            # 3. Pick a random background and paste
            bg_path = random.choice(bg_paths)
            bg = cv2.imread(str(bg_path))
            if bg is None:
                continue
            # Paste at a random map position (avoid edges)
            paste_cx = random.uniform(0.1, 0.9)
            paste_cy = random.uniform(0.1, 0.9)
            bg, box = paste_crop(bg, scaled, paste_cx, paste_cy)
            if box is None:
                continue
            act_cx, act_cy, act_bw, act_bh = box

            # 4. Slight HSV jitter on the whole image
            hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV).astype(np.int16)
            hsv[:, :, 0] = np.clip(hsv[:, :, 0] + random.randint(-8, 8), 0, 179)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] + random.randint(-20, 20), 0, 255)
            bg = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            # 5. Save
            out_stem = f"synthv2_{stem}_c{ci}_s{si}"
            cv2.imwrite(str(OUT_IMG / f"{out_stem}.png"), bg)
            (OUT_LBL / f"{out_stem}.txt").write_text(
                f"{TP_IDX} {act_cx:.6f} {act_cy:.6f} {act_bw:.6f} {act_bh:.6f}\n"
            )
            generated += 1

print(f"\nGenerated {generated} synthetic teleport images → {OUT_IMG}")

# Invalidate labels cache
cache = LBL_DIR.parent / "labels.cache"
if cache.exists():
    cache.unlink()
    print("Cleared labels.cache")
