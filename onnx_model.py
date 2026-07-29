"""
onnx_model.py — Lightweight ONNX Runtime inference for RadarRift.

Drop-in replacement for the PyTorch-based yolo_model.py / splash_model.py.
Using ONNX Runtime instead of PyTorch cuts the distributable from ~6 GB → ~400 MB
and supports any GPU on Windows via DirectML (AMD, Intel, Nvidia — no CUDA needed).

Providers tried in order:
  1. DmlExecutionProvider   — GPU via DirectML (Windows, any vendor)
  2. CPUExecutionProvider   — CPU fallback

Model files (auto-exported on first use from the .pt files in dev):
  cache/champion_yolo11n.onnx  — minimap champion-icon detector (aliases: minimap_yolo11n.onnx, yolo11n.onnx)
  cache/splash_detection.onnx  — loading-screen splash-card detector

Usage:
  from onnx_model import OnnxDetector, export_all

  export_all()   # one-time: converts .pt → .onnx if not already done

  det = OnnxDetector("cache/champion_yolo11n.onnx", conf=0.26)
  boxes = det.detect(bgr_frame)
  # boxes: list of (x1, y1, x2, y2, confidence, class_id)
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


def _frozen() -> bool:
    return getattr(sys, "frozen", False)


# Ultralytics may write a different stem than our target name; accept common aliases.
_MINIMAP_ONNX_NAMES = (
    "champion_yolo11n.onnx",
    "minimap_yolo11n.onnx",
    "yolo11n.onnx",
)
_SPLASH_ONNX_NAMES = ("splash_detection.onnx",)


def _onnx_search_roots() -> list[Path]:
    """Frozen: bundled _internal/cache, then writable cache next to exe. Dev: project cache."""
    if _frozen():
        return [
            Path(sys._MEIPASS) / "cache",
            Path(sys.executable).resolve().parent / "cache",
        ]
    return [Path(__file__).resolve().parent / "cache"]


def _bundle_cache() -> Path:
    """Read-only bundled cache (_internal/cache when frozen)."""
    if _frozen():
        return Path(sys._MEIPASS) / "cache"
    return Path(__file__).parent / "cache"


def _user_cache() -> Path:
    """Writable cache next to the exe (for downloaded/exported files)."""
    if _frozen():
        return Path(sys.executable).parent / "cache"
    return Path(__file__).parent / "cache"

# Both ONNX models bundled inside the exe — no download needed at runtime
MINIMAP_ONNX  = _bundle_cache() / "champion_yolo11n.onnx"   # single-class champion_icon
SPLASH_ONNX   = _bundle_cache() / "splash_detection.onnx"

# .pt files only used for training / re-export (source runs only)
MINIMAP_PT    = _user_cache() / "champion_yolo11n.pt"
SPLASH_PT     = _bundle_cache() / "splash_detection.pt"

_DEFAULT_IMGSZ = 640


# ── ONNX session factory ──────────────────────────────────────────────────────

def _make_session(onnx_path: str | Path):
    """Create an InferenceSession preferring DirectML GPU then CPU."""
    import os as _os
    if hasattr(sys, "_MEIPASS") and hasattr(_os, "add_dll_directory"):
        for _d in (
            _os.path.join(sys._MEIPASS, "onnxruntime", "capi"),
            sys._MEIPASS,
        ):
            if _os.path.isdir(_d):
                try:
                    _os.add_dll_directory(_d)
                except OSError:
                    pass
    import onnxruntime as ort

    path = str(onnx_path)
    # Try DirectML (GPU — any vendor, no CUDA required)
    try:
        sess = ort.InferenceSession(
            path,
            providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        )
        return sess
    except Exception:
        pass
    # CPU fallback
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


# ── Export helpers ────────────────────────────────────────────────────────────

def export_to_onnx(pt_path: Path, onnx_path: Path,
                   imgsz: int = _DEFAULT_IMGSZ, force: bool = False) -> bool:
    """
    Convert a .pt YOLO model to ONNX using Ultralytics' built-in export.
    Returns True on success, False on failure.
    If force=True, overwrites existing .onnx.
    """
    if onnx_path.exists() and not force:
        return True
    if force and onnx_path.exists():
        onnx_path.unlink()
    if not pt_path.is_file():
        print(f"  ONNX export skipped — weight file not found: {pt_path}")
        return False
    try:
        from ultralytics import YOLO

        model = YOLO(str(pt_path))
        out = model.export(format="onnx", imgsz=imgsz, simplify=True, opset=12)
        exported = Path(out)
        if not exported.is_file():
            # Fallback: same-dir stem (older Ultralytics)
            alt = pt_path.with_suffix(".onnx")
            exported = alt if alt.is_file() else exported
        if not exported.is_file():
            print(f"  ONNX export failed — output not found (expected near {pt_path})")
            return False
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        if exported.resolve() != onnx_path.resolve():
            if onnx_path.exists():
                onnx_path.unlink()
            shutil.move(str(exported), str(onnx_path))
        return True
    except Exception as e:
        print(f"  ONNX export failed for {pt_path.name}: {e}")
    return False


def resolve_minimap_onnx_path() -> Path:
    """Bundled cache, exe-adjacent cache, then dev export from .pt."""
    for root in _onnx_search_roots():
        for name in _MINIMAP_ONNX_NAMES:
            p = root / name
            if p.is_file():
                return p
    if not _frozen():
        if MINIMAP_PT.is_file():
            export_to_onnx(MINIMAP_PT, MINIMAP_ONNX, imgsz=320)
        for root in _onnx_search_roots():
            for name in _MINIMAP_ONNX_NAMES:
                p = root / name
                if p.is_file():
                    return p
    searched = [str(root / n) for root in _onnx_search_roots() for n in _MINIMAP_ONNX_NAMES]
    raise FileNotFoundError(
        "Minimap ONNX model not found. Expected one of "
        f"{_MINIMAP_ONNX_NAMES} under the cache folder (next to the .exe or inside the app bundle). "
        "Rebuild PyInstaller with .onnx files in project cache/, or copy the file into the "
        f"cache folder beside the executable. Checked: {searched!s}"
    )


def resolve_splash_onnx_path() -> Path:
    for root in _onnx_search_roots():
        for name in _SPLASH_ONNX_NAMES:
            p = root / name
            if p.is_file():
                return p
    if not _frozen():
        if SPLASH_PT.is_file():
            export_to_onnx(SPLASH_PT, SPLASH_ONNX, imgsz=1920)
        for root in _onnx_search_roots():
            for name in _SPLASH_ONNX_NAMES:
                p = root / name
                if p.is_file():
                    return p
    searched = [str(root / n) for root in _onnx_search_roots() for n in _SPLASH_ONNX_NAMES]
    raise FileNotFoundError(
        "Splash ONNX model not found. Expected splash_detection.onnx under cache. "
        f"Checked: {searched!s}"
    )


def export_all(force: bool = False) -> dict[str, bool]:
    """
    Export both models to ONNX from .pt. Returns {pt_name: success}.
    Uses imgsz 320 for minimap (champion), 1920 for splash.
    If force=True, overwrites existing .onnx files.
    """
    results = {}
    # (pt, onnx, imgsz) — must match runtime OnnxDetector imgsz
    for pt, onnx, imgsz in [
        (MINIMAP_PT, MINIMAP_ONNX, 320),
        (SPLASH_PT, SPLASH_ONNX, 1920),
    ]:
        if not pt.exists():
            results[pt.name] = False
            continue
        results[pt.name] = export_to_onnx(pt, onnx, imgsz=imgsz, force=force)
    return results


# ── Pre / post processing ─────────────────────────────────────────────────────

def _letterbox(img: np.ndarray, new_size: int = _DEFAULT_IMGSZ
               ) -> tuple[np.ndarray, float, int, int]:
    """Resize keeping aspect ratio, pad to square. Returns (img, scale, pad_x, pad_y)."""
    h, w = img.shape[:2]
    scale = new_size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_y = (new_size - nh) // 2
    pad_x = (new_size - nw) // 2
    canvas = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = img
    return canvas, scale, pad_x, pad_y


def _preprocess(bgr: np.ndarray, imgsz: int = _DEFAULT_IMGSZ
                ) -> tuple[np.ndarray, float, int, int]:
    """BGR → NCHW float32 [0,1] tensor + letterbox params."""
    lb, scale, px, py = _letterbox(bgr, imgsz)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.expand_dims(rgb.transpose(2, 0, 1), 0)   # NCHW
    return tensor, scale, px, py


def _nms(boxes: np.ndarray, scores: np.ndarray,
         iou_thr: float = 0.45) -> list[int]:
    """Simple NMS. boxes: (N,4) x1y1x2y2, scores: (N,)."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thr]
    return keep


def _postprocess(output: np.ndarray, scale: float, pad_x: int, pad_y: int,
                 orig_h: int, orig_w: int, conf_thr: float,
                 iou_thr: float = 0.45
                 ) -> list[tuple[int, int, int, int, float, int]]:
    """
    Decode YOLO11 ONNX output → list of (x1,y1,x2,y2,conf,cls).
    output shape: [1, 4+num_cls, num_anchors]
    """
    pred = output[0]                        # (4+C, N)
    pred = pred.T                           # (N, 4+C)
    box_raw  = pred[:, :4]                  # cx,cy,w,h
    cls_raw  = pred[:, 4:]                  # class logits

    conf     = cls_raw.max(axis=1)
    cls_id   = cls_raw.argmax(axis=1)
    mask     = conf >= conf_thr
    box_raw, conf, cls_id = box_raw[mask], conf[mask], cls_id[mask]

    if len(box_raw) == 0:
        return []

    # cx,cy,w,h → x1,y1,x2,y2  (still in letterboxed coords)
    x1 = box_raw[:, 0] - box_raw[:, 2] / 2
    y1 = box_raw[:, 1] - box_raw[:, 3] / 2
    x2 = box_raw[:, 0] + box_raw[:, 2] / 2
    y2 = box_raw[:, 1] + box_raw[:, 3] / 2

    # Remove padding and undo scale → original image coords
    x1 = np.clip((x1 - pad_x) / scale, 0, orig_w)
    y1 = np.clip((y1 - pad_y) / scale, 0, orig_h)
    x2 = np.clip((x2 - pad_x) / scale, 0, orig_w)
    y2 = np.clip((y2 - pad_y) / scale, 0, orig_h)

    boxes_xy = np.stack([x1, y1, x2, y2], axis=1)
    keep     = _nms(boxes_xy, conf, iou_thr=iou_thr)

    return [(int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i]),
             float(conf[i]), int(cls_id[i])) for i in keep]


# ── Public detector class ─────────────────────────────────────────────────────

class OnnxDetector:
    """
    Thin wrapper around an ONNX Runtime InferenceSession.

    Parameters
    ----------
    onnx_path : path to the .onnx model file
    conf      : confidence threshold (0–1)
    imgsz     : inference image size (must match the exported model)
    """

    def __init__(self, onnx_path: str | Path,
                 conf: float = 0.26,
                 imgsz: int  = _DEFAULT_IMGSZ) -> None:
        self._sess  = _make_session(onnx_path)
        self._iname = self._sess.get_inputs()[0].name
        self._conf  = conf
        self._imgsz = imgsz

        # Warm up
        dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
        self._sess.run(None, {self._iname: dummy})

    def detect(self, bgr: np.ndarray, iou: float = 0.45
               ) -> list[tuple[int, int, int, int, float, int]]:
        """
        Run detection on a BGR frame.
        Returns list of (x1, y1, x2, y2, confidence, class_id).
        """
        h, w = bgr.shape[:2]
        tensor, scale, px, py = _preprocess(bgr, self._imgsz)
        outputs = self._sess.run(None, {self._iname: tensor})
        return _postprocess(outputs[0], scale, px, py, h, w, self._conf,
                            iou_thr=iou)

    @property
    def conf(self) -> float:
        return self._conf

    @conf.setter
    def conf(self, v: float) -> None:
        self._conf = float(v)


# ── Convenience loaders ───────────────────────────────────────────────────────

def load_minimap_detector(conf: float = 0.26) -> OnnxDetector:
    """Load the minimap champion-icon detector. Exports .pt → .onnx in dev if needed."""
    return OnnxDetector(resolve_minimap_onnx_path(), conf=conf, imgsz=320)


def load_splash_detector(conf: float = 0.40) -> OnnxDetector:
    """Load the loading-screen splash-card detector. Exports .pt → .onnx in dev if needed."""
    return OnnxDetector(resolve_splash_onnx_path(), conf=conf)


# ── PyTorch-compatible interface (drop-in for yolo_model / splash_model) ──────

# Class names for the minimap detection model (radarrift_final4: 5-class)
_MINIMAP_CLASSES = {
    0: "ally",
    1: "enemy",
    2: "teleport",
    3: "recall",
    4: "champion_icon",
}

_minimap_det: OnnxDetector | None = None
_splash_det:  OnnxDetector | None = None


def load_model() -> OnnxDetector:
    """Load minimap detector — drop-in for yolo_model.load_model()."""
    global _minimap_det
    if _minimap_det is None:
        _minimap_det = OnnxDetector(
            resolve_minimap_onnx_path(), conf=0.26, imgsz=320,
        )
    return _minimap_det


def infer(model: OnnxDetector, frame: np.ndarray,
          imgsz: int = 320, conf: float = 0.26,
          iou: float = 0.45) -> list[dict]:
    """
    Drop-in for yolo_model.infer().
    Returns list of {class_id, class_name, conf, box:(x1,y1,x2,y2)}.
    """
    model.conf  = conf
    model._imgsz = imgsz
    raw = model.detect(frame, iou=iou)
    return [
        {
            "class_id":   cls_id,
            "class_name": _MINIMAP_CLASSES.get(cls_id, str(cls_id)),
            "conf":       conf_val,
            "box":        (x1, y1, x2, y2),
        }
        for x1, y1, x2, y2, conf_val, cls_id in raw
    ]


def load_splash_model() -> OnnxDetector:
    """Load splash detector — drop-in for splash_model.load_model()."""
    global _splash_det
    if _splash_det is None:
        _splash_det = OnnxDetector(
            resolve_splash_onnx_path(), conf=0.35, imgsz=1920,
        )
    return _splash_det


def unload_splash_model() -> None:
    """Drop splash ONNX session (e.g. after loading-screen scan window ends)."""
    global _splash_det
    _splash_det = None


def detect_cards(
    model: OnnxDetector,
    screenshot: "PIL.Image.Image",
    conf: float = 0.35,
    max_det: int = 50,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    import numpy as _np

    arr = _np.array(screenshot.convert("RGB"))
    bgr = arr[:, :, ::-1].copy()
    model.conf = conf
    raw = model.detect(bgr)

    splash_dets, name_dets = [], []
    for x1, y1, x2, y2, conf_val, cls_id in raw:
        cls_id = int(cls_id)
        b = (int(x1), int(y1), int(x2), int(y2))
        if cls_id == 0:
            splash_dets.append((float(conf_val), b))
        elif cls_id == 3:
            name_dets.append((float(conf_val), b))

    splash_dets = sorted(splash_dets, key=lambda d: -d[0])[:max_det]
    name_dets   = sorted(name_dets, key=lambda d: -d[0])[:max_det]

    from match_start import process_splash_detections
    img_h, img_w = arr.shape[:2]
    return process_splash_detections(splash_dets, name_dets, img_w, img_h)


# ── CLI helper ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv or "-f" in sys.argv
    print("Exporting models to ONNX…" + (" (force overwrite)" if force else ""))
    results = export_all(force=force)
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'}  {name}")
    if not all(results.values()):
        print("\nPut .pt weights in cache/ then run again:")
        print("  cache/champion_yolo11n.pt   → champion_yolo11n.onnx (imgsz 320)")
        print("  cache/splash_detection.pt   → splash_detection.onnx (imgsz 1920)")
