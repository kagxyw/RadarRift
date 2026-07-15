"""
Evaluate the teleport CNN over ALL labeled boxes in session_teleport
(train + val combined — every crop we have ground truth for).

Run from repo root:
  python tools/eval_tp_cls_full.py
"""
import torch
_orig = torch.load
def _p(*a, **k):
    k["weights_only"] = False
    return _orig(*a, **k)
torch.load = _p

import cv2
from pathlib import Path

_CP3      = Path(__file__).resolve().parent.parent
IMG_DIR   = _CP3 / "session_teleport/images"
LBL_DIR   = _CP3 / "session_teleport/labels"
WEIGHTS   = _CP3 / "tp_confirm_cls.pt"
CROP_SZ   = 64
PAD_FRAC  = 0.3
THRESHOLD = 0.55

NAMES  = ["ally", "enemy", "teleport", "recall", "champion_icon"]
TP_IDX = NAMES.index("teleport")


def pad_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * PAD_FRAC), int(bh * PAD_FRAC)
    crop = img[max(0,y1-py):min(h,y2+py), max(0,x1-px):min(w,x2+px)]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (CROP_SZ, CROP_SZ), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    from ultralytics import YOLO

    model  = YOLO(str(WEIGHTS))
    tp_cls_idx = None

    y_true, y_pred, y_prob, y_src = [], [], [], []

    img_paths = sorted(p for p in IMG_DIR.glob("*.png")
                       if not p.stem.startswith("synthv2"))

    for img_path in img_paths:
        lbl_path = LBL_DIR / (img_path.stem + ".txt")
        if not lbl_path.exists():
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

            res = model(crop, verbose=False)[0]
            if tp_cls_idx is None:
                tp_cls_idx = next(k for k, v in res.names.items() if v == "teleport")
            prob = float(res.probs.data[tp_cls_idx])
            pred = "teleport" if prob >= THRESHOLD else "not_teleport"
            true = "teleport" if cls == TP_IDX else "not_teleport"

            y_true.append(true)
            y_pred.append(pred)
            y_prob.append(prob)
            y_src.append((img_path.stem, NAMES[cls], prob))

    # ── metrics ──────────────────────────────────────────────────────────────
    tp = sum(t == "teleport"     and p == "teleport"     for t, p in zip(y_true, y_pred))
    fp = sum(t == "not_teleport" and p == "teleport"     for t, p in zip(y_true, y_pred))
    fn = sum(t == "teleport"     and p == "not_teleport" for t, p in zip(y_true, y_pred))
    tn = sum(t == "not_teleport" and p == "not_teleport" for t, p in zip(y_true, y_pred))

    total   = len(y_true)
    correct = tp + tn
    prec    = tp / (tp + fp) if tp + fp else 0.0
    rec     = tp / (tp + fn) if tp + fn else 0.0
    f1      = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    print(f"\nAll labeled boxes in session_teleport (excl. synthv2)")
    print(f"  Total crops   : {total}  ({tp+fn} teleport / {fp+tn} not_teleport)")
    print(f"  Threshold     : {THRESHOLD}")
    print(f"  Accuracy      : {correct/total:.3f}  ({correct}/{total})")
    print(f"  Precision     : {prec:.3f}")
    print(f"  Recall        : {rec:.3f}")
    print(f"  F1            : {f1:.3f}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")

    # ── per-class breakdown of false positives ───────────────────────────────
    from collections import Counter
    fp_classes = Counter(
        true_cls for t, p, (stem, true_cls, prob) in zip(y_true, y_pred, y_src)
        if t == "not_teleport" and p == "teleport"
    )
    print(f"\nFalse positive breakdown (which non-teleport classes fool the CNN):")
    for cls_name, count in fp_classes.most_common():
        print(f"  {cls_name}: {count}")

    print(f"\nFalse negatives (teleport crops with lowest CNN confidence):")
    fns = [(prob, stem) for t, p, (stem, cls_name, prob) in zip(y_true, y_pred, y_src)
           if t == "teleport" and p == "not_teleport"]
    for prob, stem in sorted(fns)[:10]:
        print(f"  {stem}  prob={prob:.3f}")
