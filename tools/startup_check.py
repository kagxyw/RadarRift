"""
startup_check.py — Verify and download required data files before launch.

Checks on every launch:
  1. cache/icons/*.png          — minimap champion icons
  2. cache/thumb/hist matrices  — skin identification data
  3. bundled ONNX models        — detection models

If anything is missing:
  Phase 1 — confirmation dialog lists missing items
  Phase 2 — download progress with byte counter
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path


# ── path helpers ──────────────────────────────────────────────────────────────

def _bundle_cache() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "cache"
    return Path(__file__).resolve().parent.parent / "cache"


def _user_cache() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "cache"
    return Path(__file__).resolve().parent.parent / "cache"


# ── missing-file checker ───────────────────────────────────────────────────────

class _MissingItem:
    def __init__(self, label: str, key: str) -> None:
        self.label = label
        self.key   = key


_MIN_ICONS       = 150
_MIN_MODEL_BYTES = 1_000_000

_MATRIX_MIN_KB = {
    "thumb_matrix.npy": 1_000,
    "hist_matrix.npy":  500,
    "thumb_index.json": 50,
    "hist_index.json":  10,
}


def _icon_issues(icons_dir: Path) -> str | None:
    if not icons_dir.exists():
        return "folder missing"
    pngs = list(icons_dir.glob("*.png"))
    if not pngs:
        return "no icons found"
    corrupt = [p for p in pngs if p.stat().st_size < 500]
    if corrupt:
        for p in corrupt:
            try:
                p.unlink()
            except Exception:
                pass
        return f"{len(corrupt)} corrupt icon(s) removed — need re-download"
    if len(pngs) < _MIN_ICONS:
        return f"only {len(pngs)} icons cached (expected {_MIN_ICONS}+)"
    return None


def _matrix_issues(cache: Path) -> str | None:
    for name, min_kb in _MATRIX_MIN_KB.items():
        p = cache / name
        if not p.exists():
            return f"{name} missing"
        if p.stat().st_size < min_kb * 1024:
            return (f"{name} appears corrupt "
                    f"({p.stat().st_size // 1024} KB, expected ≥{min_kb} KB)")
    return None


def _model_issues(path: Path) -> str | None:
    if not path.exists():
        return "file missing"
    if path.stat().st_size < _MIN_MODEL_BYTES:
        return f"file too small ({path.stat().st_size // 1024} KB)"
    return None


def _missing() -> list[_MissingItem]:
    bc    = _bundle_cache()
    items: list[_MissingItem] = []

    if _icon_issues(bc / "icons"):
        items.append(_MissingItem("Champion minimap icons", "icons"))

    if _matrix_issues(bc):
        items.append(_MissingItem("Skin identification matrix", "matrix"))

    for onnx_name, label in (
        ("minimap_yolo11n.onnx",  "Minimap detection model"),
        ("splash_detection.onnx", "Splash detection model"),
    ):
        if _model_issues(bc / onnx_name):
            items.append(_MissingItem(label, "splash_missing"))

    return items


# ── downloader ────────────────────────────────────────────────────────────────

def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024 ** 2:.1f} MB"


def _download_url(url: str, dest: Path,
                  on_progress: "callable | None" = None) -> None:
    import urllib.request
    with urllib.request.urlopen(url, timeout=30) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done  = 0
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if on_progress:
                    on_progress(done, total)


def _run_downloads(items: list[_MissingItem],
                   on_step: "callable",
                   on_progress: "callable",
                   on_done: "callable") -> None:
    bc = _bundle_cache()
    uc = _user_cache()

    try:
        from . import rebuild_cache as _rc
        _rc.CACHE_DIR    = bc
        _rc.THUMB_MATRIX = bc / "thumb_matrix.npy"
        _rc.HIST_MATRIX  = bc / "hist_matrix.npy"
        _rc.THUMB_INDEX  = bc / "thumb_index.json"
    except Exception as e:
        on_done(False, f"Could not load rebuild_cache module: {e}")
        return

    keys = {i.key for i in items}

    try:
        if "icons" in keys:
            on_step("Downloading champion minimap icons…")
            (bc / "icons").mkdir(parents=True, exist_ok=True)
            _rc.download_icons(verbose=False, on_progress=on_progress)
            on_step("  Icons done.")

        if "matrix" in keys:
            if not any(bc.glob("*.jpg")):
                on_step("Downloading champion skin portraits…")
                _rc.download_skins_wiki(verbose=False, on_progress=on_progress)
                on_step("  Portraits done.")
            on_step("Building identification matrix…")
            _rc.build_matrix(verbose=False, on_progress=on_progress)
            on_step("  Matrix built.")

        if "hf_minimap" in keys:
            on_step("Downloading minimap detection model…")
            dest = uc / "minimap_yolo11n.pt"
            uc.mkdir(parents=True, exist_ok=True)
            try:
                from huggingface_hub import hf_hub_download
                import shutil
                tmp = hf_hub_download(
                    repo_id="boboyes/leagueoflegends-minimap-detection",
                    filename="yolo11n-minimap.pt",
                )
                shutil.copy(tmp, dest)
            except Exception:
                _download_url(
                    "https://huggingface.co/boboyes/"
                    "leagueoflegends-minimap-detection/resolve/main/"
                    "yolo11n-minimap.pt",
                    dest, on_progress=on_progress,
                )
            on_step("  Minimap model done.")

        if "splash_missing" in keys:
            on_done(False,
                    "A bundled detection model is missing or corrupt.\n"
                    "Please re-download the full RadarRift package.")
            return

    except Exception as e:
        on_done(False, str(e))
        return

    on_done(True, "")


# ── Qt dialog ─────────────────────────────────────────────────────────────────

def _show_dialog(items: list[_MissingItem]) -> bool:
    from PyQt6.QtCore    import Qt, QTimer, pyqtSignal, QObject
    from PyQt6.QtWidgets import (
        QApplication, QDialog, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QProgressBar, QScrollArea, QFrame,
        QStackedWidget,
    )

    app = QApplication.instance() or QApplication(sys.argv)

    BG  = "#1e1e2e"
    BG2 = "#313244"
    FG  = "#cdd6f4"
    DIM = "#6c7086"
    RED = "#f38ba8"
    GRN = "#a6e3a1"
    YEL = "#f9e2af"

    BASE_STYLE = f"""
        QDialog, QWidget {{ background: {BG}; color: {FG};
                            font-family: "Segoe UI"; font-size: 13px; }}
        QPushButton {{ border-radius: 8px; padding: 8px 18px; font-weight: bold; border: none; }}
        QProgressBar {{ background: {BG2}; border-radius: 4px; height: 8px; text-align: center; }}
        QProgressBar::chunk {{ background: {GRN}; border-radius: 4px; }}
        QScrollArea, QScrollArea > QWidget > QWidget {{ background: {BG2}; }}
    """

    # ── worker signals (cross-thread safe) ────────────────────────────────────
    class _Signals(QObject):
        step     = pyqtSignal(str)
        progress = pyqtSignal(int, int)   # downloaded, total
        done     = pyqtSignal(bool, str)  # ok, error_msg

    sig = _Signals()

    # ── dialog ────────────────────────────────────────────────────────────────
    dlg = QDialog()
    dlg.setWindowTitle("RadarRift — Setup Required")
    dlg.setFixedWidth(540)
    dlg.setStyleSheet(BASE_STYLE)
    dlg.setWindowFlags(
        dlg.windowFlags()
        | Qt.WindowType.WindowStaysOnTopHint
    )

    stack  = QStackedWidget()
    result = [False]

    # ── PAGE 1: confirmation ──────────────────────────────────────────────────
    p1 = QWidget()
    v1 = QVBoxLayout(p1)
    v1.setContentsMargins(24, 20, 24, 20)
    v1.setSpacing(10)

    title1 = QLabel("RadarRift — Setup Required")
    title1.setStyleSheet(f"color: {FG}; font-size: 15px; font-weight: bold;")
    title1.setAlignment(Qt.AlignmentFlag.AlignCenter)
    v1.addWidget(title1)

    sub = QLabel("The following files are missing and must be downloaded\n"
                 "before RadarRift can run:")
    sub.setStyleSheet(f"color: {DIM}; font-size: 11px;")
    sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
    v1.addWidget(sub)

    # scrollable list of missing items
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setMaximumHeight(160)

    list_widget = QWidget()
    list_widget.setStyleSheet(f"background: {BG2}; border-radius: 6px;")
    lv = QVBoxLayout(list_widget)
    lv.setContentsMargins(12, 8, 12, 8)
    lv.setSpacing(4)
    for item in items:
        row = QLabel(f"  •  {item.label}")
        row.setStyleSheet(f"color: {RED}; font-size: 12px;")
        lv.addWidget(row)
    lv.addStretch()
    scroll.setWidget(list_widget)
    v1.addWidget(scroll)

    note = QLabel("An internet connection is required.\n"
                  "This only runs when files are missing.")
    note.setStyleSheet(f"color: {DIM}; font-size: 10px;")
    note.setAlignment(Qt.AlignmentFlag.AlignCenter)
    v1.addWidget(note)

    v1.addStretch()

    btn_row1 = QWidget()
    bl1 = QHBoxLayout(btn_row1)
    bl1.setContentsMargins(0, 0, 0, 0)
    bl1.setSpacing(12)
    bl1.addStretch()

    cancel_btn = QPushButton("Cancel")
    cancel_btn.setStyleSheet("background: #45475a; color: #cdd6f4;")
    cancel_btn.clicked.connect(dlg.reject)

    install_btn = QPushButton("Download & Setup")
    install_btn.setStyleSheet(f"background: {GRN}; color: #1e1e2e;")

    bl1.addWidget(cancel_btn)
    bl1.addWidget(install_btn)
    v1.addWidget(btn_row1)

    # ── PAGE 2: download progress ─────────────────────────────────────────────
    p2 = QWidget()
    v2 = QVBoxLayout(p2)
    v2.setContentsMargins(24, 20, 24, 20)
    v2.setSpacing(10)

    title2 = QLabel("RadarRift — Downloading…")
    title2.setStyleSheet(f"color: {FG}; font-size: 15px; font-weight: bold;")
    title2.setAlignment(Qt.AlignmentFlag.AlignCenter)
    v2.addWidget(title2)

    step_lbl = QLabel("Starting…")
    step_lbl.setStyleSheet(f"color: {YEL}; font-size: 12px;")
    step_lbl.setWordWrap(True)
    v2.addWidget(step_lbl)

    prog_bar = QProgressBar()
    prog_bar.setRange(0, 100)
    prog_bar.setValue(0)
    v2.addWidget(prog_bar)

    prog_lbl = QLabel("")
    prog_lbl.setStyleSheet(f"color: {DIM}; font-size: 10px;")
    prog_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
    v2.addWidget(prog_lbl)

    v2.addStretch()

    status_lbl = QLabel("")
    status_lbl.setStyleSheet(f"color: {FG}; font-size: 13px; font-weight: bold;")
    status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    status_lbl.setWordWrap(True)
    v2.addWidget(status_lbl)

    close_btn = QPushButton("Please wait…")
    close_btn.setStyleSheet("background: #45475a; color: #6c7086;")
    close_btn.setEnabled(False)
    v2.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignCenter)

    # ── assemble stack ────────────────────────────────────────────────────────
    stack.addWidget(p1)
    stack.addWidget(p2)

    root_layout = QVBoxLayout(dlg)
    root_layout.setContentsMargins(0, 0, 0, 0)
    root_layout.addWidget(stack)

    # ── signal handlers (run on Qt main thread) ───────────────────────────────
    def _on_step(msg: str) -> None:
        step_lbl.setText(msg)
        prog_lbl.setText("")

    def _on_progress(downloaded: int, total: int) -> None:
        if total > 0:
            pct = min(100, downloaded * 100 // total)
            prog_bar.setValue(pct)
            prog_lbl.setText(
                f"{_fmt_bytes(downloaded)} / {_fmt_bytes(total)}  ({pct}%)")
        else:
            prog_lbl.setText(f"{_fmt_bytes(downloaded)} downloaded")

    def _on_done(ok: bool, err: str) -> None:
        prog_bar.setValue(100 if ok else prog_bar.value())
        if ok:
            prog_lbl.setText("Complete")
            status_lbl.setText("✓  All files ready — launching RadarRift…")
            status_lbl.setStyleSheet(
                f"color: {GRN}; font-size: 13px; font-weight: bold;")
            close_btn.setText("Launch")
            close_btn.setStyleSheet(f"background: {GRN}; color: #1e1e2e;")
            close_btn.setEnabled(True)

            def _launch():
                result[0] = True
                dlg.accept()
            close_btn.clicked.connect(_launch)
        else:
            status_lbl.setText(f"✗  Error: {err}")
            status_lbl.setStyleSheet(
                f"color: {RED}; font-size: 12px; font-weight: bold;")
            close_btn.setText("Close")
            close_btn.setStyleSheet(f"background: {RED}; color: #1e1e2e;")
            close_btn.setEnabled(True)
            close_btn.clicked.connect(dlg.reject)

    sig.step.connect(_on_step)
    sig.progress.connect(_on_progress)
    sig.done.connect(_on_done)

    def _start_download() -> None:
        stack.setCurrentIndex(1)
        dlg.setWindowTitle("RadarRift — Downloading…")
        threading.Thread(
            target=_run_downloads,
            args=(
                items,
                lambda msg, **_kw: sig.step.emit(msg),
                lambda d, t:       sig.progress.emit(d, t),
                lambda ok, err:    sig.done.emit(ok, err),
            ),
            daemon=True,
        ).start()

    install_btn.clicked.connect(_start_download)

    dlg.exec()
    return result[0]


# ── public entry point ────────────────────────────────────────────────────────

def ensure_ready() -> bool:
    """
    Called on every launch.
    Nothing missing  → returns True immediately (silent).
    Something missing → shows Qt confirm → download dialog.
    Returns True only if all downloads succeeded.
    """
    absent = _missing()
    if not absent:
        return True
    return _show_dialog(absent)
