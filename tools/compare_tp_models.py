#!/usr/bin/env python3
"""Compare the shipped TP classifier against the aug_faint variant at the
high-recall end of the curve, where the alert actually operates.

Run from repo root:
    python -m tools.compare_tp_models
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

_orig = torch.load


def _patched(*a, **k):
    k["weights_only"] = False
    return _orig(*a, **k)


torch.load = _patched

VAL = _ROOT / "dataset_tp_cls" / "val"

MODELS = {
    "CURRENT": (_ROOT / "runs/classify/tp_confirm_cls/weights/best.pt", 64),
    "AUG_FAINT": (_ROOT / "runs/classify/tp_exp_aug_faint/weights/best.pt", 128),
}


def score(weights: Path, imgsz: int):
    import cv2
    from ultralytics import YOLO

    m = YOLO(str(weights))
    tp = next(k for k, v in m.names.items() if v == "teleport")

    def run(d: Path) -> np.ndarray:
        fs = sorted(d.glob("*"))
        out = []
        for i in range(0, len(fs), 64):
            b = [cv2.resize(cv2.imread(str(f)), (imgsz, imgsz),
                            interpolation=cv2.INTER_AREA) for f in fs[i:i + 64]]
            for r in m(b, verbose=False, imgsz=imgsz):
                out.append(float(r.probs.data[tp]))
        return np.array(out)

    return run(VAL / "teleport"), run(VAL / "not_teleport")


if __name__ == "__main__":
    s = {name: score(p, z) for name, (p, z) in MODELS.items()}
    cp, cn = s["CURRENT"]
    ap, an = s["AUG_FAINT"]
    npos = len(cp)

    print(f"HIGH-RECALL OPERATING POINTS  ({npos} pos / {len(cn)} neg)")
    print(f"{'recall':>7} | {'CUR thr':>9} {'CUR FP':>7} | "
          f"{'AUG thr':>9} {'AUG FP':>7}")
    print("-" * 50)
    for target in (0.85, 0.90, 0.925, 0.94, 0.955, 0.97, 0.985, 1.0):
        k = int(np.ceil(target * npos))
        ct = np.sort(cp)[::-1][k - 1]
        at = np.sort(ap)[::-1][k - 1]
        print(f"{k/npos:>7.3f} | {ct:>9.4f} {int((cn >= ct).sum()):>7} | "
              f"{at:>9.4f} {int((an >= at).sum()):>7}")

    print("\n10 lowest-scoring TRUE teleports (the hard cases):")
    print("  CURRENT  :", np.round(np.sort(cp)[:10], 4))
    print("  AUG_FAINT:", np.round(np.sort(ap)[:10], 4))

    # Paired bootstrap: is the FP difference at 94% recall real, or noise?
    rng = np.random.default_rng(0)
    k = int(np.ceil(0.94 * npos))
    wins = 0
    trials = 2000
    for _ in range(trials):
        ip = rng.integers(0, npos, npos)
        ineg = rng.integers(0, len(cn), len(cn))
        ct = np.sort(cp[ip])[::-1][k - 1]
        at = np.sort(ap[ip])[::-1][k - 1]
        if (an[ineg] >= at).sum() < (cn[ineg] >= ct).sum():
            wins += 1
    print(f"\nPaired bootstrap @94% recall: AUG_FAINT has fewer false "
          f"positives in {wins/trials:.1%} of {trials} resamples")
