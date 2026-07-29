"""
Extract labeled bounding-box crops from session_teleport for binary CNN training.

Positives  = boxes labeled "teleport"
Negatives  = boxes labeled ally / enemy / recall / champion_icon

No random patches — the CNN only ever sees crops from YOLO detections, so
negatives must also be real YOLO-box crops, not random map patches.

Output (YOLO-cls format):
  dataset_tp_cls/
    train/  teleport/  not_teleport/
    val/    teleport/  not_teleport/

Run from repo root:
  python tools/extract_tp_crops.py
"""
import cv2, random, shutil, numpy as np
from pathlib import Path

random.seed(42)
np.random.seed(42)

IMG_DIR  = Path("session_teleport/images")
LBL_DIR  = Path("session_teleport/labels")
OUT_ROOT = Path("dataset_tp_cls")
CROP_SZ  = 64
PAD_FRAC = 0.3
VAL_FRAC = 0.20   # fraction of each class held out for val (stratified)

NAMES   = ["ally", "enemy", "teleport", "recall", "champion_icon"]
TP_IDX  = NAMES.index("teleport")


def pad_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * PAD_FRAC), int(bh * PAD_FRAC)
    crop = img[max(0,y1-py):min(h,y2+py), max(0,x1-px):min(w,x2+px)]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (CROP_SZ, CROP_SZ), interpolation=cv2.INTER_AREA)


# ── collect all crops ─────────────────────────────────────────────────────────
pos_crops = []   # list of np arrays
neg_crops = []

for lbl_path in sorted(LBL_DIR.glob("*.txt")):
    img_path = IMG_DIR / (lbl_path.stem + ".png")
    if not img_path.exists():
        continue
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    h, w = img.shape[:2]

    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        cls = int(parts[0])
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = int((cx - bw/2) * w); y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w); y2 = int((cy + bh/2) * h)
        crop = pad_crop(img, x1, y1, x2, y2)
        if crop is None:
            continue
        if cls == TP_IDX:
            pos_crops.append(crop)
        else:
            neg_crops.append(crop)

print(f"Raw crops — teleport: {len(pos_crops)}  not_teleport: {len(neg_crops)}")

# ── stratified val split ──────────────────────────────────────────────────────
random.shuffle(pos_crops)
random.shuffle(neg_crops)

n_pos_val = max(4, int(len(pos_crops) * VAL_FRAC))
n_neg_val = max(4, int(len(neg_crops) * VAL_FRAC))

pos_val,  pos_train = pos_crops[:n_pos_val],  pos_crops[n_pos_val:]
neg_val,  neg_train = neg_crops[:n_neg_val],  neg_crops[n_neg_val:]

# ── oversample train positives to match train negatives ───────────────────────
def oversample_with_aug(pool, target):
    """Return `target` crops by cycling pool with minor augmentation."""
    out = list(pool)
    i   = 0
    while len(out) < target:
        aug = pool[i % len(pool)].copy()
        if random.random() < 0.5:
            aug = cv2.flip(aug, 1)
        hsv = cv2.cvtColor(aug, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:,:,2] = np.clip(hsv[:,:,2] + random.randint(-25, 25), 0, 255)
        aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        out.append(aug)
        i += 1
    return out

pos_train = oversample_with_aug(pos_train, len(neg_train))
print(f"After oversample — train: {len(pos_train)} tp / {len(neg_train)} neg  |  val: {len(pos_val)} tp / {len(neg_val)} neg")

# ── save ──────────────────────────────────────────────────────────────────────
if OUT_ROOT.exists():
    shutil.rmtree(OUT_ROOT)
for split in ("train", "val"):
    for cls in ("teleport", "not_teleport"):
        (OUT_ROOT / split / cls).mkdir(parents=True)

def save_crops(crops, folder):
    for i, crop in enumerate(crops):
        cv2.imwrite(str(folder / f"{i:05d}.jpg"), crop)

save_crops(pos_train, OUT_ROOT / "train" / "teleport")
save_crops(neg_train, OUT_ROOT / "train" / "not_teleport")
save_crops(pos_val,   OUT_ROOT / "val"   / "teleport")
save_crops(neg_val,   OUT_ROOT / "val"   / "not_teleport")

print("Done. Dataset saved to", OUT_ROOT)
