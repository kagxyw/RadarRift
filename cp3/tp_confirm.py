"""
tp_confirm.py — lightweight CNN teleport confirmation stage.

Loads runs/classify/tp_confirm_cls/weights/best.pt (YOLO-cls) and,
given a BGR crop of a YOLO teleport detection, returns a confidence
score in [0, 1] that the crop is a genuine teleport animation.

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

# Where to look for the classifier weights (in priority order)
_CLS_CANDIDATES = [
    _ROOT / "runs" / "classify" / "tp_confirm_cls" / "weights" / "best.pt",
]
# Also support being bundled next to the .exe
if getattr(sys, "frozen", False):
    _exe_dir = Path(sys.executable).parent
    _CLS_CANDIDATES.insert(0, _exe_dir / "_internal" / "tp_confirm_cls.pt")
    _CLS_CANDIDATES.insert(1, _exe_dir / "tp_confirm_cls.pt")

CROP_SZ   = 64    # must match training size
THRESHOLD = 0.55  # CNN probability above this → confirmed teleport
PAD_FRAC  = 0.3   # padding added around the YOLO box before feeding to CNN


def _resolve_weights() -> Path | None:
    for p in _CLS_CANDIDATES:
        if p.is_file():
            return p
    return None


class TpConfirmer:
    """
    Singleton-friendly teleport CNN confirmer.
    Thread-safe for read (predict) after initial load.
    """

    def __init__(self, threshold: float = THRESHOLD):
        self._model = None
        self._threshold = threshold
        self._available: bool | None = None   # None = not yet tried

    # ── public ────────────────────────────────────────────────────────────────

    def is_teleport(self, crop_bgr: np.ndarray) -> tuple[bool, float]:
        """
        Return (confirmed: bool, prob: float).
        If the model weights are not found, returns (True, 1.0) so behaviour
        is unchanged (pass-through).
        """
        model = self._load()
        if model is None:
            return True, 1.0          # no classifier → always pass

        prob = self._predict(model, crop_bgr)
        return prob >= self._threshold, prob

    @property
    def available(self) -> bool:
        """True if classifier weights were found."""
        if self._available is None:
            self._available = _resolve_weights() is not None
        return self._available

    # ── internal ──────────────────────────────────────────────────────────────

    def _load(self):
        if self._model is not None:
            return self._model
        weights = _resolve_weights()
        if weights is None:
            self._available = False
            return None
        try:
            import torch
            _orig = torch.load
            def _p(*a, **k):
                k["weights_only"] = False
                return _orig(*a, **k)
            torch.load = _p
            from ultralytics import YOLO
            self._model = YOLO(str(weights))
            self._available = True
        except Exception as exc:
            print(f"[TpConfirmer] Failed to load CNN: {exc}")
            self._model = None
            self._available = False
        return self._model

    def _predict(self, model, crop_bgr: np.ndarray) -> float:
        """Run a 64×64 crop through the classifier, return teleport probability."""
        try:
            resized = cv2.resize(crop_bgr, (CROP_SZ, CROP_SZ),
                                 interpolation=cv2.INTER_AREA)
            results = model(resized, verbose=False)
            probs = results[0].probs
            # classes: not_teleport=0, teleport=1  (alphabetical in YOLO-cls)
            tp_idx = results[0].names  # dict {0: 'not_teleport', 1: 'teleport'}
            # find which index maps to 'teleport'
            tp_class_idx = next(
                (k for k, v in tp_idx.items() if v == "teleport"), None
            )
            if tp_class_idx is None:
                return 1.0   # unknown layout → pass through
            return float(probs.data[tp_class_idx])
        except Exception:
            return 1.0   # on any error → pass through


# Module-level singleton
_confirmer: TpConfirmer | None = None


def get_confirmer() -> TpConfirmer:
    global _confirmer
    if _confirmer is None:
        _confirmer = TpConfirmer()
    return _confirmer
