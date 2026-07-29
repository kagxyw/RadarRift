#!/usr/bin/env python3
"""Sweep the TP-confirmation CNN threshold on dataset_tp_cls/val.

Reports precision / recall / F1 per threshold for the ONNX path actually
used by the shipped executable, and shows what the old double-softmax bug
was doing to the same scores.

Run from repo root:
    python -m tools.tune_tp_threshold
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

VAL = _ROOT / "dataset_tp_cls" / "val"
ONNX = _ROOT / "runs" / "classify" / "tp_confirm_cls" / "weights" / "best.onnx"
CROP_SZ = 64
TP_IDX = 1


def _scores(sess, files: list[Path]) -> np.ndarray:
    iname = sess.get_inputs()[0].name
    out = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        r = cv2.resize(img, (CROP_SZ, CROP_SZ), interpolation=cv2.INTER_AREA)
        rgb = r[:, :, ::-1].astype(np.float32) / 255.0
        t = np.ascontiguousarray(rgb.transpose(2, 0, 1)[np.newaxis])
        p = sess.run(None, {iname: t})[0][0]
        out.append(float(p[TP_IDX]))
    return np.array(out)


def _double_softmax(p: np.ndarray) -> np.ndarray:
    """Reproduce the old bug: softmax applied to already-softmaxed output."""
    other = 1.0 - p
    stack = np.stack([other, p], axis=1)
    stack = stack - stack.max(axis=1, keepdims=True)
    e = np.exp(stack)
    return (e / e.sum(axis=1, keepdims=True))[:, 1]


def report(pos: np.ndarray, neg: np.ndarray, title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
    print(f"  teleport   n={len(pos):<4} mean={pos.mean():.3f} "
          f"median={np.median(pos):.3f} p10={np.percentile(pos,10):.3f}")
    print(f"  not_tele   n={len(neg):<4} mean={neg.mean():.3f} "
          f"median={np.median(neg):.3f} p90={np.percentile(neg,90):.3f}")
    print()
    print(f"  {'thr':>5} {'TP':>4} {'FN':>4} {'FP':>4} {'TN':>5} "
          f"{'prec':>7} {'recall':>7} {'F1':>7}")
    print("  " + "-" * 54)
    best = None
    for thr in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70,
                0.75, 0.80, 0.85, 0.90, 0.95]:
        tp = int((pos >= thr).sum())
        fn = len(pos) - tp
        fp = int((neg >= thr).sum())
        tn = len(neg) - fp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        mark = ""
        if best is None or f1 > best[1]:
            best = (thr, f1)
        print(f"  {thr:>5.2f} {tp:>4} {fn:>4} {fp:>4} {tn:>5} "
              f"{prec:>7.3f} {rec:>7.3f} {f1:>7.3f}{mark}")
    print(f"\n  best F1 at thr={best[0]:.2f} (F1={best[1]:.3f})")


if __name__ == "__main__":
    import onnxruntime as ort

    if not ONNX.is_file():
        raise SystemExit(f"missing {ONNX}")
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])

    pos_f = sorted((VAL / "teleport").glob("*"))
    neg_f = sorted((VAL / "not_teleport").glob("*"))
    print(f"scoring {len(pos_f)} teleport + {len(neg_f)} not_teleport crops…")

    pos = _scores(sess, pos_f)
    neg = _scores(sess, neg_f)

    report(pos, neg, "FIXED — graph softmax used directly (current code)")
    report(_double_softmax(pos), _double_softmax(neg),
           "BUGGY — double softmax (what shipped before this fix)")
