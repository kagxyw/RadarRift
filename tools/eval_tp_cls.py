"""
Evaluate the binary teleport CNN on the validation set.

Run from repo root:
  python tools/eval_tp_cls.py
"""
import torch
_orig = torch.load
def _p(*a, **k):
    k["weights_only"] = False
    return _orig(*a, **k)
torch.load = _p

from pathlib import Path
import cv2

THRESHOLD = 0.55
VAL_DIR   = Path("dataset_tp_cls/val")
WEIGHTS   = Path("runs/classify/tp_confirm_cls/weights/best.pt")

if __name__ == "__main__":
    from ultralytics import YOLO

    model  = YOLO(str(WEIGHTS))
    tp_idx = None

    y_true, y_pred, y_prob = [], [], []

    for cls_dir in sorted(VAL_DIR.iterdir()):
        label = cls_dir.name  # "teleport" or "not_teleport"
        imgs  = sorted(cls_dir.glob("*.jpg"))
        print(f"  {label}: {len(imgs)} images")
        for img_path in imgs:
            img = cv2.imread(str(img_path))
            res = model(img, verbose=False)[0]
            if tp_idx is None:
                tp_idx = next(k for k, v in res.names.items() if v == "teleport")
            prob = float(res.probs.data[tp_idx])
            pred = "teleport" if prob >= THRESHOLD else "not_teleport"
            y_true.append(label)
            y_pred.append(pred)
            y_prob.append(prob)

    # ── confusion matrix ─────────────────────────────────────────────────────
    tp = sum(t == "teleport"     and p == "teleport"     for t, p in zip(y_true, y_pred))
    fp = sum(t == "not_teleport" and p == "teleport"     for t, p in zip(y_true, y_pred))
    fn = sum(t == "teleport"     and p == "not_teleport" for t, p in zip(y_true, y_pred))
    tn = sum(t == "not_teleport" and p == "not_teleport" for t, p in zip(y_true, y_pred))

    total   = len(y_true)
    correct = tp + tn
    prec    = tp / (tp + fp) if tp + fp else 0.0
    rec     = tp / (tp + fn) if tp + fn else 0.0
    f1      = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    print()
    print(f"Threshold   : {THRESHOLD}")
    print(f"Val images  : {total}  ({tp+fn} teleport / {fp+tn} not_teleport)")
    print(f"Accuracy    : {correct/total:.3f}  ({correct}/{total})")
    print(f"Precision   : {prec:.3f}  (of predicted teleport, how many were real)")
    print(f"Recall      : {rec:.3f}  (of real teleports, how many were caught)")
    print(f"F1          : {f1:.3f}")
    print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")

    # ── show worst mistakes ───────────────────────────────────────────────────
    items = list(zip(y_true, y_pred, y_prob,
                     [p for cls_dir in sorted(VAL_DIR.iterdir())
                        for p in sorted(cls_dir.glob("*.jpg"))]))

    print("\nWorst false positives (not_teleport predicted as teleport):")
    fps = [(prob, path) for t, p, prob, path in items if t == "not_teleport" and p == "teleport"]
    for prob, path in sorted(fps, reverse=True)[:5]:
        print(f"  {path.name}  prob={prob:.3f}")

    print("\nWorst false negatives (teleport missed):")
    fns = [(prob, path) for t, p, prob, path in items if t == "teleport" and p == "not_teleport"]
    for prob, path in sorted(fns)[:5]:
        print(f"  {path.name}  prob={prob:.3f}")
