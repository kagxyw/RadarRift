"""
yolo_champion.py — Train a single-class champion icon detector.

Uses dataset_champion/ which has every icon labelled as class 0 "champion_icon".
Fine-tunes from the existing minimap HF base model (same domain, same icon size).

Usage:
  python yolo_champion.py train              # 80 epochs from base
  python yolo_champion.py train --resume     # continue from best.pt
  python yolo_champion.py export-onnx        # export best to ONNX
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

ROOT       = Path(__file__).parent
DATA_YAML  = ROOT / "dataset_champion" / "dataset.yaml"
WEIGHTS    = ROOT / "runs" / "detect" / "radarrift_champion" / "weights" / "best.pt"
BASE_MODEL = ROOT / "cache" / "minimap_yolo11n.pt"
CP2_MODEL  = ROOT / "cp2" / "best.pt"


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _half() -> bool:
    return torch.cuda.is_available()


def train(epochs: int = 80, imgsz: int = 320, batch: int = 16,
          resume: bool = False) -> YOLO:
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"dataset.yaml not found at {DATA_YAML}")

    if resume and WEIGHTS.exists():
        model = YOLO(str(WEIGHTS))
        model.train(
            data         = str(DATA_YAML),
            epochs       = epochs,
            imgsz        = imgsz,
            batch        = batch,
            device       = _device(),
            half         = _half(),
            name         = "radarrift_champion",
            lr0          = 0.0005,
            freeze       = 0,
            workers      = 0,
            patience     = 30,
            cache        = True,
            close_mosaic = 10,
            verbose      = True,
            exist_ok     = True,
        )
    else:
        model = YOLO(str(BASE_MODEL))
        model.train(
            data         = str(DATA_YAML),
            epochs       = epochs,
            imgsz        = imgsz,
            batch        = batch,
            device       = _device(),
            half         = _half(),
            name         = "radarrift_champion",
            lr0          = 0.001,
            freeze       = 10,
            workers      = 0,
            patience     = 30,
            cache        = True,
            close_mosaic = 10,
            verbose      = True,
            exist_ok     = True,
        )

    return YOLO(str(WEIGHTS)) if WEIGHTS.exists() else model


def load_model(weights: str | Path | None = None) -> YOLO:
    """Load the champion icon detector for use by the tracker."""
    path = Path(weights) if weights else next(
        (p for p in (WEIGHTS, BASE_MODEL, CP2_MODEL) if p.exists()), BASE_MODEL)
    dev  = _device()
    model = YOLO(str(path))
    dummy = np.zeros((320, 320, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=320, device=dev, half=_half(), verbose=False)
    return model


def infer(
    model: YOLO,
    frame: np.ndarray,
    imgsz: int   = 320,
    conf:  float = 0.35,
    iou:   float = 0.45,
) -> list[dict]:
    """Run detection and return list of dicts with box/conf/class info."""
    results = model.predict(
        frame,
        imgsz   = imgsz,
        conf    = conf,
        iou     = iou,
        device  = _device(),
        half    = _half(),
        verbose = False,
    )
    name_table = getattr(model, "names", {0: "champion_icon"})
    detections = []
    for r in results:
        for box in r.boxes:
            cid  = int(box.cls[0])
            class_name = name_table.get(cid, "champion_icon")
            # cp2/best.pt also detects recall and teleport effects. They are
            # not champion portraits and must not enter identity matching.
            if class_name not in {"ally", "enemy", "champion_icon"}:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                "class_id":   cid,
                "class_name": class_name,
                "conf":       float(box.conf[0]),
                "box":        (x1, y1, x2, y2),
            })
    # class-agnostic NMS
    detections.sort(key=lambda d: d["conf"], reverse=True)
    kept = []
    for d in detections:
        def _iou(a, b):
            ix1, iy1 = max(a[0],b[0]), max(a[1],b[1])
            ix2, iy2 = min(a[2],b[2]), min(a[3],b[3])
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            if inter == 0: return 0.0
            aa = (a[2]-a[0])*(a[3]-a[1]); ab = (b[2]-b[0])*(b[3]-b[1])
            return inter / (aa + ab - inter)
        if all(_iou(d["box"], k["box"]) < iou for k in kept):
            kept.append(d)
    return kept


def export_onnx(imgsz: int = 320) -> Path:
    model = YOLO(str(WEIGHTS))
    out   = model.export(format="onnx", imgsz=imgsz, half=_half(), dynamic=False)
    dest  = ROOT / "cache" / "champion_yolo11n.onnx"
    shutil.copy(out, dest)
    print(f"ONNX exported -> {dest}")
    return dest


if __name__ == "__main__":
    cmd    = sys.argv[1] if len(sys.argv) > 1 else "help"
    resume = "--resume" in sys.argv
    epochs = next((int(sys.argv[i+1]) for i, a in enumerate(sys.argv)
                   if a == "--epochs" and i+1 < len(sys.argv)), 80)

    if cmd == "train":
        train(epochs=epochs, resume=resume)
    elif cmd == "export-onnx":
        export_onnx()
    else:
        print(__doc__)
