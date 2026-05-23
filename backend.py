"""
Inference backend selector.

When running from a PyInstaller bundle  (sys.frozen = True)  → ONNX Runtime.
When running from source                (python main.py)      → PyTorch/Ultralytics.

Override at any time:
    set RADARRIFT_BACKEND=onnx    # force ONNX even in source
    set RADARRIFT_BACKEND=pytorch # force PyTorch even in bundle
"""

import os
import sys

_env = os.environ.get("RADARRIFT_BACKEND", "").lower()
if _env == "onnx":
    USE_ONNX = True
elif _env == "pytorch":
    USE_ONNX = False
else:
    USE_ONNX = getattr(sys, "frozen", False)   # True only in .exe


def load_model():
    if USE_ONNX:
        from onnx_model import load_model as _f
    else:
        from yolo_champion import load_model as _f
    return _f()


def infer(model, frame, imgsz=320, conf=0.26, iou=0.45):
    if USE_ONNX:
        from onnx_model import infer as _f
    else:
        from yolo_champion import infer as _f
    return _f(model, frame, imgsz=imgsz, conf=conf, iou=iou)


def load_splash_model():
    if USE_ONNX:
        from onnx_model import load_splash_model as _f
    else:
        from splash_model import load_model as _f
    return _f()


def unload_splash_model():
    if USE_ONNX:
        from onnx_model import unload_splash_model as _f
        _f()
    # PyTorch splash_model has no separate unload; GC handles it when ref dropped.


def detect_cards(model, screenshot, conf=0.35, max_det=50):
    """Returns (splash_boxes, name_boxes) — top-10 each."""
    if USE_ONNX:
        from onnx_model import detect_cards as _f
    else:
        from splash_model import detect_cards as _f
    return _f(model, screenshot, conf=conf, max_det=max_det)
