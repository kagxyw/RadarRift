#!/usr/bin/env python3
"""End-to-end check of the TP confirmer exactly as the app calls it.

Runs TpConfirmer.is_teleport() over dataset_tp_cls/val and reports the
confusion matrix at the configured THRESHOLD, so what is measured here is
the real shipping code path rather than a reimplementation of it.

Run from repo root:
    python -m tools.verify_tp_runtime
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

VAL = _ROOT / "dataset_tp_cls" / "val"


if __name__ == "__main__":
    import cv2

    from tp_confirm import CROP_SZ, THRESHOLD, get_confirmer

    c = get_confirmer()
    print(f"CROP_SZ={CROP_SZ}  THRESHOLD={THRESHOLD}  available={c.available}")
    c.is_teleport(cv2.imread(str(next((VAL / 'teleport').glob('*')))))
    print(f"backend = {'ONNX' if c._use_onnx else 'PyTorch'}")

    tp = fn = fp = tn = 0
    for f in sorted((VAL / "teleport").glob("*")):
        ok, _ = c.is_teleport(cv2.imread(str(f)))
        tp, fn = (tp + 1, fn) if ok else (tp, fn + 1)
    for f in sorted((VAL / "not_teleport").glob("*")):
        ok, _ = c.is_teleport(cv2.imread(str(f)))
        fp, tn = (fp + 1, tn) if ok else (fp, tn + 1)

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    print(f"\n  true teleports  caught {tp:>3} / {tp+fn:<3} -> recall    {rec:.3f}")
    print(f"  non-teleports   passed {fp:>3} / {fp+tn:<3} -> precision {prec:.3f}")
    print(f"  F1 {2*prec*rec/(prec+rec) if prec+rec else 0:.3f}")
