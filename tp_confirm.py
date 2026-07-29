"""
tp_confirm.py — lightweight CNN teleport confirmation stage.

Tries ONNX Runtime first (works in the PyInstaller exe), then falls back to
ultralytics YOLO (dev / source). If neither is available the confirmer passes
every detection through unchanged.

Usage::
    from tp_confirm import TpConfirmer
    confirmer = TpConfirmer()            # lazy-loads on first call
    if confirmer.is_teleport(crop_bgr):
        ...
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent

# ── weight search order ────────────────────────────────────────────────────────
# ONNX candidates (preferred — works in exe without torch/ultralytics)
_ONNX_CANDIDATES = [
    _ROOT / "runs" / "classify" / "tp_confirm_cls" / "weights" / "best.onnx",
]
# PT candidates (dev fallback) — keep in sync with the ONNX above, otherwise
# running from source behaves differently from the shipped executable.
_PT_CANDIDATES = [
    _ROOT / "runs" / "classify" / "tp_exp_aug_faint" / "weights" / "best.pt",
    _ROOT / "runs" / "classify" / "tp_confirm_cls" / "weights" / "best.pt",
]

if getattr(sys, "frozen", False):
    _exe_dir = Path(sys.executable).parent
    # PyInstaller 6+ puts "." datas inside _internal/
    _ONNX_CANDIDATES.insert(0, _exe_dir / "_internal" / "tp_confirm_cls.onnx")
    _ONNX_CANDIDATES.insert(1, _exe_dir / "tp_confirm_cls.onnx")
    _PT_CANDIDATES.insert(0, _exe_dir / "_internal" / "tp_confirm_cls.pt")
    _PT_CANDIDATES.insert(1, _exe_dir / "tp_confirm_cls.pt")

# Input size — MUST match the size the active weights were exported at.
# Re-export with tools/export_tp_cls_onnx.py after changing this.
CROP_SZ   = 128

# CNN probability above this → confirmed teleport.
# Tuned on dataset_tp_cls/val (67 pos / 380 neg); see tools/compare_tp_models.py.
# Recall-first setting: 0.29 → 94% per-frame recall, 27 false positives / 380.
# Other points on the same curve:
#     0.40 → 91.0% recall, 25 FP
#     0.03 → 98.5% recall, 42 FP   (practical ceiling)
#     0.75 → 67.2% recall, 11 FP   (precision-first)
THRESHOLD = 0.29
PAD_FRAC  = 0.3   # padding added around the YOLO box before feeding to CNN

# Class index for "teleport" in the ONNX model output (alphabetical YOLO-cls order:
#   0 = not_teleport, 1 = teleport)
_TP_CLASS_IDX = 1


def _resolve_onnx() -> Path | None:
    for p in _ONNX_CANDIDATES:
        if p.is_file():
            return p
    return None


def _resolve_pt() -> Path | None:
    for p in _PT_CANDIDATES:
        if p.is_file():
            return p
    return None


class TpConfirmer:
    """
    Singleton-friendly teleport CNN confirmer.
    Prefers ONNX Runtime (exe-safe); falls back to ultralytics YOLO in dev.
    Thread-safe for read (predict) after initial load.
    """

    def __init__(self, threshold: float = THRESHOLD):
        self._model = None        # onnxruntime.InferenceSession or YOLO
        self._use_onnx = False
        self._threshold = threshold
        self._available: bool | None = None

    # ── public ────────────────────────────────────────────────────────────────

    def is_teleport(self, crop_bgr: np.ndarray) -> tuple[bool, float]:
        """
        Return (confirmed: bool, prob: float).
        Falls through as (True, 1.0) if no weights are available.
        """
        model = self._load()
        if model is None:
            return True, 1.0

        prob = (self._predict_onnx(model, crop_bgr)
                if self._use_onnx
                else self._predict_pt(model, crop_bgr))
        return prob >= self._threshold, prob

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = (
                _resolve_onnx() is not None or _resolve_pt() is not None
            )
        return self._available

    # ── internal ──────────────────────────────────────────────────────────────

    def _load(self):
        if self._model is not None:
            return self._model

        # 1. Try ONNX Runtime
        onnx_path = _resolve_onnx()
        if onnx_path is not None:
            try:
                import onnxruntime as ort
                sess = ort.InferenceSession(
                    str(onnx_path),
                    providers=["CPUExecutionProvider"],
                )
                self._model    = sess
                self._use_onnx = True
                self._available = True
                print(f"[TpConfirmer] Loaded ONNX: {onnx_path.name}")
                return self._model
            except Exception as exc:
                print(f"[TpConfirmer] ONNX load failed: {exc}")

        # 2. Fall back to ultralytics YOLO (.pt)
        pt_path = _resolve_pt()
        if pt_path is not None:
            try:
                import torch
                _orig = torch.load
                def _p(*a, **k):
                    k["weights_only"] = False
                    return _orig(*a, **k)
                torch.load = _p
                from ultralytics import YOLO
                self._model    = YOLO(str(pt_path))
                self._use_onnx = False
                self._available = True
                print(f"[TpConfirmer] Loaded PT: {pt_path.name}")
                return self._model
            except Exception as exc:
                print(f"[TpConfirmer] PT load failed: {exc}")

        self._available = False
        return None

    def _predict_onnx(self, sess, crop_bgr: np.ndarray) -> float:
        """ONNX Runtime inference — returns teleport probability."""
        try:
            resized = cv2.resize(crop_bgr, (CROP_SZ, CROP_SZ),
                                 interpolation=cv2.INTER_AREA)
            # BGR → RGB, HWC → NCHW float32 normalised [0,1]
            rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
            tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[np.newaxis])
            iname  = sess.get_inputs()[0].name
            out    = np.asarray(sess.run(None, {iname: tensor})[0][0],
                                dtype=np.float64)
            # Ultralytics classification exports bake softmax into the graph,
            # so the output is already a probability vector. Only normalise
            # when a raw-logit graph is detected.
            if abs(out.sum() - 1.0) > 1e-3 or out.min() < 0.0:
                out = out - out.max()
                exp = np.exp(out)
                out = exp / exp.sum()
            return float(out[_TP_CLASS_IDX])
        except Exception:
            return 1.0

    def _predict_pt(self, model, crop_bgr: np.ndarray) -> float:
        """Ultralytics YOLO-cls inference — returns teleport probability."""
        try:
            resized = cv2.resize(crop_bgr, (CROP_SZ, CROP_SZ),
                                 interpolation=cv2.INTER_AREA)
            results = model(resized, verbose=False)
            probs   = results[0].probs
            names   = results[0].names
            tp_idx  = next(
                (k for k, v in names.items() if v == "teleport"), None
            )
            if tp_idx is None:
                return 1.0
            return float(probs.data[tp_idx])
        except Exception:
            return 1.0


# Module-level singleton
_confirmer: TpConfirmer | None = None


def get_confirmer() -> TpConfirmer:
    global _confirmer
    if _confirmer is None:
        _confirmer = TpConfirmer()
    return _confirmer
