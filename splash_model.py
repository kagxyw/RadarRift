"""
splash_model.py — YOLO-based loading screen splash card detection.

Trains / loads a YOLO model that detects champion splash card bounding boxes
in a LoL loading screen.  The output is simply a list of bounding boxes;
champion identification is handled by match_start.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).parent
DATASET_DIR = ROOT / "dataset_splash"
WEIGHTS_DIR = ROOT / "runs" / "detect" / "splash"

# Priority:
#   1. cache/splash_detection.pt  — bundled trained model (shipped with exe)
#   2. runs/detect/splash/…/best.pt — local training output
#   3. Any best.pt with "splash" in path (fallback scan)
_CACHE_PT = ROOT / "cache" / "splash_detection.pt"
BEST_PT   = _CACHE_PT

if not BEST_PT.exists():
    BEST_PT = WEIGHTS_DIR / "weights" / "best.pt"

if not BEST_PT.exists():
    candidates = sorted(ROOT.rglob("best.pt"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    candidates = [p for p in candidates if "splash" in str(p)]
    if candidates:
        BEST_PT = candidates[0]


# ── Training ──────────────────────────────────────────────────────────────────

def train(epochs: int = 200, lr: float = 0.05, imgsz: int = 1920) -> None:
    """Fine-tune the splash detector on the current dataset_splash annotations."""
    import shutil
    from ultralytics import YOLO

    data_yaml = DATASET_DIR / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_yaml}\n"
            "Annotate loading screen screenshots with annotation_tool.py first."
        )

    # Fine-tune from the best existing model; fall back to raw yolo11n
    base = _CACHE_PT if _CACHE_PT.exists() else "yolo11n.pt"
    print(f"Fine-tuning from: {base}")

    model = YOLO(str(base))
    model.train(
        data      = str(data_yaml),
        epochs    = epochs,
        imgsz     = imgsz,
        batch     = 2,
        project   = str(ROOT / "runs" / "detect"),
        name      = "splash",
        exist_ok  = True,
        half      = False,      # FP32 — required for stability at imgsz=1920
        device    = "0" if _cuda_available() else "cpu",
        lr0       = lr,
        freeze    = 0,          # unfreeze all layers for fine-tuning
        patience  = 40,
        workers   = 0,
        # augmentation — important for small datasets
        hsv_h     = 0.015,
        hsv_s     = 0.5,
        hsv_v     = 0.3,
        fliplr    = 0.5,
        scale     = 0.3,
        degrees   = 5.0,
        translate = 0.1,
        mosaic    = 0.0,        # off — full-res splash images don't tile well
    )

    # Copy best weights back to cache for app use
    best = ROOT / "runs" / "detect" / "splash" / "weights" / "best.pt"
    if best.exists():
        shutil.copy(best, _CACHE_PT)
        print(f"Copied best weights → {_CACHE_PT}")
    print(f"\nTrained model saved to {BEST_PT}")


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ── Inference ─────────────────────────────────────────────────────────────────

_model = None


def load_model():
    """Load (and cache) the best trained splash model."""
    global _model
    if _model is not None:
        return _model
    if not BEST_PT.exists():
        raise FileNotFoundError(
            f"Splash model not found at {BEST_PT}.\n"
            "Run  python splash_model.py train  first."
        )
    from ultralytics import YOLO
    print(f"Loading splash model from {BEST_PT} …")
    _model = YOLO(str(BEST_PT))
    return _model


def _iou(a: tuple[int,int,int,int], b: tuple[int,int,int,int]) -> float:
    """Intersection-over-union of two (x1,y1,x2,y2) boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter  = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1)


def _nms(dets: list, iou_thresh: float = 0.15) -> list:
    """
    Greedy NMS: sort by confidence descending, keep a box only if it has
    zero meaningful overlap with every already-kept box.
    iou_thresh=0.15 is intentionally tight — splash cards never overlap
    in a real loading screen, so any overlap means a duplicate detection.
    """
    dets = sorted(dets, key=lambda d: -d[0])
    kept: list = []
    for d in dets:
        if all(_iou(d[1], k[1]) < iou_thresh for k in kept):
            kept.append(d)
    return kept


def detect_cards(
    model,
    screenshot: "Image.Image",
    conf: float = 0.001,
    max_det: int = 50,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """Return (splash_boxes, name_boxes) — top-10 each by confidence.

    splash_boxes : champion_splash detections, sorted spatially (row then col)
    name_boxes   : summoner_name  detections, sorted spatially (row then col)

    Overlapping boxes are suppressed with tight NMS (iou=0.15) before
    selecting the top-10, so a double-firing on one card never sneaks through.
    """
    import numpy as np
    arr = np.array(screenshot.convert("RGB"))

    results = model(
        arr,
        imgsz  = 1920,
        conf   = conf,
        iou    = 0.3,    # tighter YOLO-internal NMS (default 0.7 is too loose)
        max_det= max_det,
        verbose= False,
        half   = False,
    )

    names      = model.names
    splash_cls = next((k for k, v in names.items() if v == "champion_splash"), 0)
    name_cls   = next((k for k, v in names.items() if v == "summoner_name"),   3)

    splash_dets, name_dets = [], []   # (conf, box)
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cid      = int(box.cls[0])
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            b = (x1, y1, x2, y2)
            if cid == splash_cls:
                splash_dets.append((conf_val, b))
            elif cid == name_cls:
                name_dets.append((conf_val, b))

    from match_start import process_splash_detections
    img_h, img_w = arr.shape[0], arr.shape[1]
    return process_splash_detections(splash_dets, name_dets, img_w, img_h)


def detect_all(
    model,
    screenshot: "Image.Image",
    conf: float = 0.001,
    max_det: int = 30,
) -> dict[str, list[tuple[int, int, int, int]]]:
    """Return all detections grouped by class name.

    Returns e.g. {'champion_splash': [...], 'summoner_name': [...], ...}
    """
    import numpy as np
    arr = np.array(screenshot.convert("RGB"))

    results = model(
        arr,
        imgsz  = 1920,
        conf   = conf,
        max_det= max_det,
        verbose= False,
        half   = False,
    )

    names = model.names
    out: dict[str, list] = {v: [] for v in names.values()}
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cname = names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            out[cname].append((x1, y1, x2, y2))
    return out


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2 or sys.argv[1] == "train":
        epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 200
        lr     = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
        train(epochs=epochs, lr=lr)
    else:
        from PIL import Image
        img   = Image.open(sys.argv[1]).convert("RGB")
        model = load_model()
        boxes = detect_cards(model, img)
        print(f"Detected {len(boxes)} cards:")
        for b in boxes:
            print(f"  {b}")
