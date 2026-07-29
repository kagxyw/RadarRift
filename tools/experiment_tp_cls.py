#!/usr/bin/env python3
"""Train several TP-classifier variants and rank them by RECALL.

Each variant is trained then scored on dataset_tp_cls/val with a full
precision/recall curve, so we can compare "how much recall can this model
give me before the false positives become unacceptable".

Run from repo root:
    python -m tools.experiment_tp_cls
    python -m tools.experiment_tp_cls --only baseline_plus,bigger
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

_orig_load = torch.load


def _patched(*a, **k):
    k["weights_only"] = False
    return _orig_load(*a, **k)


torch.load = _patched

DATA = _ROOT / "dataset_tp_cls"
VAL = DATA / "val"

# name -> training kwargs
VARIANTS: dict[str, dict] = {
    # Current shipped config, but allowed to train to completion.
    "baseline_plus": dict(
        model="yolo11n-cls.pt", epochs=200, imgsz=64, batch=32,
        patience=60, lr0=5e-4, lrf=0.01,
    ),
    # Larger backbone — more capacity for the faint/ambiguous swirls.
    "bigger": dict(
        model="yolo11s-cls.pt", epochs=200, imgsz=64, batch=32,
        patience=60, lr0=5e-4, lrf=0.01,
    ),
    # Upscaled input: same source pixels, more spatial resolution in the convs.
    "upscale128": dict(
        model="yolo11n-cls.pt", epochs=200, imgsz=128, batch=32,
        patience=60, lr0=5e-4, lrf=0.01,
    ),
    # Augmentation aimed at the observed failure mode: faint, low-contrast,
    # partially-occluded teleport swirls.
    "aug_faint": dict(
        model="yolo11s-cls.pt", epochs=250, imgsz=128, batch=32,
        patience=80, lr0=8e-4, lrf=0.01,
        hsv_h=0.02, hsv_s=0.6, hsv_v=0.6,
        degrees=20.0, translate=0.15, scale=0.4,
        fliplr=0.5, flipud=0.2, erasing=0.3,
    ),
}


def train_variant(name: str, cfg: dict) -> Path:
    from ultralytics import YOLO

    cfg = dict(cfg)
    base = cfg.pop("model")
    run = f"tp_exp_{name}"
    print(f"\n{'='*72}\nTRAIN {name}  ({base}, imgsz={cfg.get('imgsz')})\n{'='*72}")
    t0 = time.time()
    m = YOLO(base)
    m.train(data=str(DATA), name=run, exist_ok=True, verbose=False,
            plots=False, **cfg)
    w = _ROOT / "runs" / "classify" / run / "weights" / "best.pt"
    print(f"  -> {w}   ({time.time()-t0:.0f}s)")
    return w


def score(weights: Path, imgsz: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (positive_scores, negative_scores) for the teleport class."""
    import cv2
    from ultralytics import YOLO

    m = YOLO(str(weights))
    names = m.names
    tp_idx = next((k for k, v in names.items() if v == "teleport"), 1)

    def run(folder: Path) -> np.ndarray:
        out = []
        files = sorted(folder.glob("*"))
        for i in range(0, len(files), 64):
            batch = [cv2.resize(cv2.imread(str(f)), (imgsz, imgsz),
                                interpolation=cv2.INTER_AREA)
                     for f in files[i:i + 64]]
            batch = [b for b in batch if b is not None]
            if not batch:
                continue
            for r in m(batch, verbose=False, imgsz=imgsz):
                out.append(float(r.probs.data[tp_idx]))
        return np.array(out)

    return run(VAL / "teleport"), run(VAL / "not_teleport")


def pr_analysis(pos: np.ndarray, neg: np.ndarray) -> dict:
    """Recall achievable at several precision floors, plus average precision."""
    ths = np.unique(np.concatenate([pos, neg, [0.0, 1.0]]))
    rows = []
    for t in ths:
        tp = int((pos >= t).sum())
        fp = int((neg >= t).sum())
        if tp == 0:
            continue
        rows.append((t, tp / (tp + fp), tp / len(pos), fp))

    out = {}
    for floor in (0.90, 0.85, 0.80, 0.70, 0.60):
        ok = [r for r in rows if r[1] >= floor]
        if ok:
            best = max(ok, key=lambda r: r[2])
            out[f"recall@P>={floor:.2f}"] = (best[2], best[0], best[3])
        else:
            out[f"recall@P>={floor:.2f}"] = (0.0, float("nan"), 0)

    # average precision (step-wise integration over recall)
    rows_sorted = sorted(rows, key=lambda r: r[2])
    ap, prev_r = 0.0, 0.0
    for _, p, r, _ in rows_sorted:
        ap += (r - prev_r) * p
        prev_r = r
    out["AP"] = ap
    out["max_recall"] = max((r[2] for r in rows), default=0.0)
    return out


if __name__ == "__main__":
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1].split(",")

    results: dict[str, dict] = {}

    # Score the currently shipped model as the reference point.
    cur = _ROOT / "runs" / "classify" / "tp_confirm_cls" / "weights" / "best.pt"
    if cur.is_file():
        print("Scoring CURRENT shipped model…")
        p, n = score(cur, 64)
        results["CURRENT (shipped)"] = pr_analysis(p, n)

    for name, cfg in VARIANTS.items():
        if only and name not in only:
            continue
        try:
            w = train_variant(name, cfg)
            p, n = score(w, cfg["imgsz"])
            results[name] = pr_analysis(p, n)
        except Exception as e:
            print(f"  !! {name} failed: {e}")

    print("\n\n" + "=" * 88)
    print("RESULTS — recall achievable at each precision floor")
    print("=" * 88)
    hdr = (f"{'variant':<22}{'AP':>7}{'maxRec':>8}"
           f"{'R@P.90':>9}{'R@P.85':>9}{'R@P.80':>9}{'R@P.70':>9}")
    print(hdr)
    print("-" * 88)
    for name, r in results.items():
        print(f"{name:<22}{r['AP']:>7.3f}{r['max_recall']:>8.3f}"
              f"{r['recall@P>=0.90'][0]:>9.3f}{r['recall@P>=0.85'][0]:>9.3f}"
              f"{r['recall@P>=0.80'][0]:>9.3f}{r['recall@P>=0.70'][0]:>9.3f}")

    print("\nOperating points (recall, threshold, #FP out of 380):")
    for name, r in results.items():
        print(f"\n  {name}")
        for k in ("recall@P>=0.90", "recall@P>=0.85",
                  "recall@P>=0.80", "recall@P>=0.70"):
            rec, thr, fp = r[k]
            print(f"     {k:<16} recall={rec:.3f}  thr={thr:.3f}  FP={fp}")
