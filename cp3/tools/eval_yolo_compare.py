"""
Compare CP2 (radarrift_final4) vs CP3 (continue_Teleport) YOLO per-class metrics
using the session_teleport validation split.

Run from repo root:
  python tools/eval_yolo_compare.py
"""
import torch, yaml, shutil
_orig = torch.load
def _p(*a, **k):
    k["weights_only"] = False
    return _orig(*a, **k)
torch.load = _p

from pathlib import Path
import numpy as np

_CP3    = Path(__file__).resolve().parent.parent
_ROOT   = _CP3.parent   # repo root (for CP2 model which lives outside cp3)

CLASSES = ["ally", "enemy", "teleport", "recall", "champion_icon"]
VAL_DIR = _CP3 / "session_teleport"

# Build a tiny eval yaml pointing at session_teleport (train=val — only val matters)
YAML_PATH = _CP3 / "_tmp_eval.yaml"
yaml.dump({
    "path":  str(VAL_DIR),
    "train": "images",
    "val":   "images",
    "nc":    len(CLASSES),
    "names": CLASSES,
}, open(YAML_PATH, "w"))

MODELS = {
    "CP2 radarrift_final4":  str(_ROOT / "runs/detect/radarrift_final4/weights/best.pt"),
    "CP3 continue_Teleport": str(_CP3 / "best.pt"),
}

if __name__ == "__main__":
    from ultralytics import YOLO

    results = {}
    for label, pt in MODELS.items():
        if not Path(pt).exists():
            print(f"SKIP {label} — {pt} not found")
            continue
        print(f"\n{'='*60}")
        print(f"Evaluating: {label}")
        model = YOLO(pt)
        m = model.val(data=str(YAML_PATH), imgsz=320, conf=0.25, iou=0.5, verbose=False)
        results[label] = m

    Path(YAML_PATH).unlink(missing_ok=True)

    for label, m in results.items():
        p  = np.array(m.box.p)
        r  = np.array(m.box.r)
        a  = np.array(m.box.ap50)
        f1 = np.where((p+r)>0, 2*p*r/(p+r), 0.0)
        names = [m.names[i] for i in range(len(p))]

        print(f"\n{'─'*60}")
        print(f"  {label}")
        print(f"{'─'*60}")
        print(f"  {'Class':<16} {'P':>7} {'R':>7} {'F1':>7} {'mAP50':>7}")
        print(f"  {'-'*46}")
        for n, pi, ri, fi, ai in zip(names, p, r, f1, a):
            print(f"  {n:<16} {pi:>7.3f} {ri:>7.3f} {fi:>7.3f} {ai:>7.3f}")
        print(f"  {'-'*46}")
        print(f"  {'mean':<16} {p.mean():>7.3f} {r.mean():>7.3f} {f1.mean():>7.3f} {a.mean():>7.3f}")
