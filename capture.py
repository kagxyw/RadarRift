"""
capture.py — Screen region selection and frame grabbing.

Uses dxcam (DXGI Desktop Duplication) for low-latency GPU-direct capture.
Falls back to mss if dxcam is unavailable.
"""

from __future__ import annotations

import threading
import numpy as np
from PIL import Image

# ── backend selection ─────────────────────────────────────────────────────────

# try:
#     import dxcam as _dxcam
#     _DXCAM_AVAILABLE = True
# except ImportError:
#     _DXCAM_AVAILABLE = False
_DXCAM_AVAILABLE = False

try:
    import mss as _mss
    _MSS_AVAILABLE = True
except ImportError:
    _MSS_AVAILABLE = False

if not _MSS_AVAILABLE:
    raise ImportError("mss is not installed. Run: pip install mss")

_BACKEND = "mss"


# ── capture backends ──────────────────────────────────────────────────────────

_DXCAM_SINGLETON: "_dxcam.DXCamera | None" = None
_DXCAM_LOCK = threading.Lock()


def _get_dxcam() -> "_dxcam.DXCamera":
    """
    Return the single shared DXCamera for output 0.

    dxcam silently reuses an existing instance when you call create() twice,
    but prints a noisy warning.  This singleton avoids the double-create
    entirely so the warning never appears.
    """
    global _DXCAM_SINGLETON
    with _DXCAM_LOCK:
        if _DXCAM_SINGLETON is None:
            _DXCAM_SINGLETON = _dxcam.create(output_color="RGB")
    return _DXCAM_SINGLETON


class _DxcamCapture:
    """
    Frame grabber backed by dxcam (DXGI Desktop Duplication).

    dxcam grabs from the GPU framebuffer directly — no compositor round-trip.
    The underlying DXCamera is a process-wide singleton so multiple Capture
    objects don't trigger dxcam's "already created" warning.

    dxcam region format: (left, top, right, bottom)  ← note: right/bottom, not w/h
    """

    def __init__(self, region: tuple[int, int, int, int]):
        self.region      = region
        left, top, w, h  = region
        self._dx_region  = (left, top, left + w, top + h)
        self._cam        = _get_dxcam()
        self._lock       = _DXCAM_LOCK
        self._last: np.ndarray | None = None

    def grab(self) -> Image.Image:
        """Return the latest frame as a PIL RGB Image."""
        with self._lock:
            frame = self._cam.grab(region=self._dx_region)
        if frame is None:
            # dxcam returns None when the screen hasn't updated since last grab;
            # reuse the previous frame so the caller always gets a valid image.
            if self._last is not None:
                return Image.fromarray(self._last)
            left, top, w, h = self.region
            return Image.new("RGB", (w, h), (0, 0, 0))
        self._last = frame
        return Image.fromarray(frame)

    def grab_fresh(self, timeout_s: float = 0.05) -> Image.Image:
        """
        Discard any cached frame and block until dxcam returns a genuinely
        new frame (non-None) or timeout_s elapses.

        Use this after toggling WDA_EXCLUDEFROMCAPTURE so the grab always
        reflects the updated composited desktop rather than a stale buffer.
        """
        import time
        with self._lock:
            self._last = None   # invalidate cache
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            with self._lock:
                frame = self._cam.grab(region=self._dx_region)
            if frame is not None:
                self._last = frame
                return Image.fromarray(frame)
            time.sleep(0.003)
        # Timeout — return a blank frame rather than stale data
        left, top, w, h = self.region
        return Image.new("RGB", (w, h), (0, 0, 0))

    def close(self) -> None:
        pass


class _MssCapture:
    """
    Frame grabber using mss (GDI/BitBlt).

    GDI BitBlt respects WDA_EXCLUDEFROMCAPTURE, so windows hidden from capture
    appear as black/absent.  Used as the YOLO inference grabber when the overlay
    must be excluded from the captured frame.

    mss stores Win32 DC handles in thread-local storage, so each thread that
    calls grab() gets its own mss context created on first use.
    """

    def __init__(self, region: tuple[int, int, int, int]):
        self.region  = region
        left, top, w, h = region
        self._monitor = {"left": left, "top": top, "width": w, "height": h}
        self._local  = threading.local()

    def _sct(self) -> "_mss.base.MSSBase":
        if not hasattr(self._local, "sct"):
            self._local.sct = _mss.mss()
        return self._local.sct

    def grab(self) -> Image.Image:
        shot = self._sct().grab(self._monitor)
        return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

    def close(self) -> None:
        if hasattr(self._local, "sct"):
            self._local.sct.close()
            del self._local.sct


# ── public Capture class ──────────────────────────────────────────────────────

class Capture:
    """
    Thread-safe screen region grabber.

    Automatically uses dxcam when available (faster, GPU-direct),
    otherwise falls back to mss.

    Usage:
        cap = Capture(region=(x, y, w, h))
        img = cap.grab()    # safe to call from any thread
        cap.close()
    """

    def __init__(self, region: tuple[int, int, int, int]):
        self.region = region
        if _DXCAM_AVAILABLE:
            try:
                self._backend = _DxcamCapture(region)
                self._name    = "dxcam"
            except Exception as exc:
                # print(f"[capture] dxcam init failed ({exc}), falling back to mss")
                self._backend = _MssCapture(region)
                self._name    = "mss"
        else:
            self._backend = _MssCapture(region)
            self._name    = "mss"
        # print(f"[capture] Capture({region}) using {self._name}")

    def grab(self) -> Image.Image:
        return self._backend.grab()

    def grab_fresh(self, timeout_s: float = 0.05) -> Image.Image:
        """Block until a new frame is available (invalidates the cache first)."""
        if hasattr(self._backend, "grab_fresh"):
            return self._backend.grab_fresh(timeout_s)
        return self._backend.grab()

    def grab_gdi(self) -> Image.Image:
        """
        Grab via GDI/BitBlt (mss).  Unlike dxcam, GDI BitBlt respects
        WDA_EXCLUDEFROMCAPTURE, so set the overlay hidden before calling this
        to get a clean frame without the overlay markers.
        """
        if not hasattr(self, "_mss_backend"):
            self._mss_backend = _MssCapture(self.region)
        return self._mss_backend.grab()

    def close(self) -> None:
        if hasattr(self, "_mss_backend"):
            self._mss_backend.close()
        self._backend.close()
