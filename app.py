"""
app.py — App(AppWindow): tracking logic wired to the Qt UI.

Inherits all widget-building from AppWindow (ui.py).
QApplication is created in main.py before App() is instantiated.
"""

from __future__ import annotations

import ctypes
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageGrab

from PyQt6.QtCore    import (
    QEvent,
    QFileSystemWatcher,
    QObject,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui     import QColor, QImage, QKeyEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QWidget, QFileDialog,
    QRadioButton, QFrame, QLineEdit, QSpinBox,
    QComboBox, QCompleter, QTextEdit, QPlainTextEdit,
)

from capture  import Capture
from champions import Champion, ChampionRoster, STATUS_ON_MAP, STATUS_OFF_MAP
from constants import BG, FG, DIM, ALLY, ENE, ACT, ASSETS_DIR, _POS_FILE, _ROSTER_FILE
from overlay_qt import QtOverlay
from select_minimap import (
    auto_minimap_region,
    default_persisted_settings_path,
    read_minimap_persisted,
)
from ui       import AppWindow
from win_lol  import lol_client_is_foreground


# ── process helpers ────────────────────────────────────────────────────────────

def _lol_running() -> bool:
    try:
        out = subprocess.check_output(
            ["tasklist", "/FI", "IMAGENAME eq League of Legends.exe", "/NH"],
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        ).decode(errors="ignore")
        return "League of Legends.exe" in out
    except Exception:
        return False


def _win_vk_down(vk: int) -> bool:
    """True if Windows virtual key is currently held (works even when LoL is focused)."""
    if vk <= 0:
        return False
    try:
        return (ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000) != 0
    except Exception:
        return False


def _vk_display(vk: int) -> str:
    """Short label for a Windows virtual key (for the Set key button)."""
    if vk <= 0:
        return "—"
    if 0x41 <= vk <= 0x5A:
        return chr(vk)
    if 0x30 <= vk <= 0x39:
        return chr(vk)
    if 0x70 <= vk <= 0x87:
        return f"F{vk - 0x6F}"
    _names = {
        0x20: "Space",
        0x1B: "Esc",
        0x09: "Tab",
        0x0D: "Enter",
        0x10: "Shift",
        0x11: "Ctrl",
        0x12: "Alt",
    }
    return _names.get(vk, f"0x{vk:X}")


def _grab_fullscreen() -> Image.Image:
    return ImageGrab.grab()


def _is_black_screen(img: Image.Image, threshold: float = 15.0) -> bool:
    arr = np.array(img.resize((64, 36), Image.LANCZOS), dtype=np.float32)
    return float(arr.mean()) < threshold


_PREVIEW_BG = ASSETS_DIR / "屏幕截图 2026-03-07 060822.png"


def _load_preview_bg(size: int) -> "np.ndarray | None":
    """Load the bundled minimap screenshot as a (size×size, RGB) numpy array."""
    import cv2 as _cv2
    try:
        raw = _PREVIEW_BG.read_bytes()
        bgr = _cv2.imdecode(np.frombuffer(raw, np.uint8), _cv2.IMREAD_COLOR)
        if bgr is not None:
            bgr = _cv2.resize(bgr, (size, size))
            return _cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    return None


def _blend_ghost_icon_circle_rgb(
    base: np.ndarray,
    icon_rgb: np.ndarray,
    cx: int,
    cy: int,
    size: int,
    alpha: float,
    border_color: tuple[int, int, int] = (128, 40, 40),
) -> None:
    """Blend champion *icon_rgb* into *base* (H×W×RGB uint8), clipped to a circle."""
    import cv2 as _cv2

    H, W = base.shape[:2]
    if size < 4:
        return
    half = size // 2
    x0, y0 = cx - half, cy - half
    x1, y1 = max(0, x0), max(0, y0)
    x2, y2 = min(W, x0 + size), min(H, y0 + size)
    sx0, sy0 = x1 - x0, y1 - y0
    rw, rh = x2 - x1, y2 - y1
    if rw < 1 or rh < 1:
        return
    icon_sq = _cv2.resize(icon_rgb, (size, size), interpolation=_cv2.INTER_AREA)
    icon_part = icon_sq[sy0 : sy0 + rh, sx0 : sx0 + rw]
    mask_full = np.zeros((size, size), np.uint8)
    _cv2.circle(
        mask_full,
        (size // 2, size // 2),
        size // 2,
        255,
        -1,
        lineType=_cv2.LINE_AA,
    )
    mask = mask_full[sy0 : sy0 + rh, sx0 : sx0 + rw]
    m = (mask.astype(np.float32) / 255.0)[..., None]
    a = float(np.clip(alpha, 0.0, 1.0))
    roi = base[y1:y2, x1:x2].astype(np.float32)
    blended = roi * (1.0 - a * m) + icon_part.astype(np.float32) * (a * m)
    base[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    if 0 <= cx < W and 0 <= cy < H:
        _cv2.circle(base, (cx, cy), size // 2, border_color, 1, _cv2.LINE_AA)


# ── Qt region selectors ────────────────────────────────────────────────────────

def _selection_help_icon(tooltip: str) -> QPushButton:
    """Small circled i chip.

    Uses QPushButton (not QLabel): on WA_TranslucentBackground windows, buttons
    still paint background-color via QStyle::drawControl(CE_PushButtonBevel); many
    QLabel styles are skipped. #:hover / #:pressed give clear affordance.
    """
    btn = QPushButton("i")
    btn.setObjectName("SelectionHelpIcon")
    btn.setFixedSize(22, 22)
    btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    btn.setStyleSheet(
        "QPushButton#SelectionHelpIcon {"
        "  color: #1e1e2e;"
        "  background-color: #89b4fa;"
        "  border-radius: 11px;"
        "  border: none;"
        "  font-weight: bold;"
        "  font-size: 12px;"
        "  padding: 0;"
        "}"
        "QPushButton#SelectionHelpIcon:hover { background-color: #b4d0fa; }"
        "QPushButton#SelectionHelpIcon:pressed { background-color: #74a7e8; }"
    )
    btn.setToolTip(tooltip)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    return btn


def _fill_dim_outside_rect(
    p: QPainter,
    sw: int,
    sh: int,
    rx: int,
    ry: int,
    rw: int,
    rh: int,
    dim: QColor,
) -> None:
    """Dim the screen but leave the selection rectangle fully clear (no dark overlay on it)."""
    rx = max(0, min(rx, sw))
    ry = max(0, min(ry, sh))
    rw = max(0, min(rw, sw - rx))
    rh = max(0, min(rh, sh - ry))
    if rw <= 0 or rh <= 0:
        p.fillRect(0, 0, sw, sh, dim)
        return
    p.fillRect(0, 0, sw, ry, dim)
    p.fillRect(0, ry + rh, sw, max(0, sh - ry - rh), dim)
    p.fillRect(0, ry, rx, rh, dim)
    p.fillRect(rx + rw, ry, max(0, sw - rx - rw), rh, dim)


class _RegionSelectorDlg(QWidget):
    """Full-screen overlay: pick minimap corner + size."""

    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        _scr      = QApplication.primaryScreen()
        screen    = _scr.geometry()
        self._dpr = _scr.devicePixelRatio()
        self._sw  = screen.width()
        self._sh  = screen.height()
        self.setGeometry(screen)

        self.result: tuple[int, int, int, int] | None = None

        self._size   = max(150, int(self._sh * 0.185))
        self._corner = read_minimap_persisted()[1]
        self._min_sz = max(80,  int(self._sh * 0.08))
        self._max_sz = min(self._sw // 2, int(self._sh * 0.45))

        # ── floating control panel ────────────────────────────────────────────
        panel = QFrame(self)
        panel.setStyleSheet("""
            QFrame      { background:#1e1e2e; border-radius:12px; padding:10px; }
            QLabel      { color:#cdd6f4; }
            QRadioButton{ color:#cdd6f4; }
            QPushButton { border-radius:6px; padding:6px 14px; font-weight:bold; }
            QSlider::groove:horizontal {
                background:#313244; height:4px; border-radius:2px; }
            QSlider::handle:horizontal {
                background:#cba6f7; width:14px; height:14px;
                border-radius:7px; margin:-5px 0; }
        """)
        pl = QVBoxLayout(panel)
        pl.setSpacing(8)

        title = QLabel("Select Minimap Region")
        title.setStyleSheet("color:#cba6f7; font-size:14px; font-weight:bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_row = QWidget()
        tr = QHBoxLayout(title_row)
        tr.setContentsMargins(0, 0, 0, 0)
        tr.setSpacing(4)
        tr.addWidget(title, 1)
        tr.addWidget(
            _selection_help_icon(
                "Choose which corner the minimap sits in (bottom-left or bottom-right) "
                "and the square capture size. Default corner follows League’s FlipMiniMap "
                "in PersistedSettings (0 = right). The green rectangle previews what will "
                "be saved. Drag the slider to resize (width equals height). "
                "Confirm saves; Cancel or Escape discards.",
            ),
            0,
            Qt.AlignmentFlag.AlignTop,
        )
        pl.addWidget(title_row)

        corner_row = QWidget()
        cr = QHBoxLayout(corner_row)
        cr.setContentsMargins(0, 0, 0, 0)
        self._left_rb  = QRadioButton("Bottom-Left")
        self._right_rb = QRadioButton("Bottom-Right")
        if self._corner == "left":
            self._left_rb.setChecked(True)
        else:
            self._right_rb.setChecked(True)
        self._left_rb.toggled.connect(
            lambda c: self._on_corner("left")  if c else None)
        self._right_rb.toggled.connect(
            lambda c: self._on_corner("right") if c else None)
        cr.addWidget(self._left_rb)
        cr.addWidget(self._right_rb)
        pl.addWidget(corner_row)

        sz_lbl = QLabel("Size")
        sz_lbl.setStyleSheet("color:#585b70; font-size:11px;")
        sz_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pl.addWidget(sz_lbl)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(self._min_sz, self._max_sz)
        self._slider.setValue(self._size)
        self._slider.setFixedWidth(260)
        self._slider.valueChanged.connect(self._on_size)
        pl.addWidget(self._slider)

        _phys0 = round(self._size * self._dpr)
        self._size_lbl = QLabel(f"{_phys0} × {_phys0} px")
        self._size_lbl.setStyleSheet(
            "color:#00ff88; font-weight:bold; font-size:12px;")
        self._size_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pl.addWidget(self._size_lbl)

        btn_row = QWidget()
        bl = QHBoxLayout(btn_row)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(8)
        confirm = QPushButton("✓  Confirm")
        confirm.setStyleSheet("background:#a6e3a1; color:#1e1e2e;")
        confirm.clicked.connect(self._confirm)
        cancel = QPushButton("✕  Cancel")
        cancel.setStyleSheet("background:#f38ba8; color:#1e1e2e;")
        cancel.clicked.connect(self.close)
        bl.addWidget(confirm)
        bl.addWidget(cancel)
        pl.addWidget(btn_row)

        panel.adjustSize()
        panel.move((self._sw - panel.sizeHint().width()) // 2, 24)

        self.show()

    def _on_corner(self, corner: str) -> None:
        self._corner = corner
        self.update()

    def _on_size(self, val: int) -> None:
        self._size = val
        phys = round(val * self._dpr)
        self._size_lbl.setText(f"{phys} × {phys} px")
        self.update()

    def _confirm(self) -> None:
        sz  = self._size
        x   = (self._sw - sz - 8) if self._corner == "right" else 8
        y   = self._sh - sz - 8
        self.result = (round(x * self._dpr), round(y * self._dpr),
                       round(sz * self._dpr), round(sz * self._dpr))
        self.close()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        sz = self._size
        x  = (self._sw - sz - 8) if self._corner == "right" else 8
        y  = self._sh - sz - 8
        dim = QColor(0, 0, 0, 140)
        _fill_dim_outside_rect(p, self._sw, self._sh, x, y, sz, sz, dim)
        p.setPen(QPen(QColor(0, 255, 136), 3))
        p.setBrush(QColor(0, 255, 136, 30))
        p.drawRect(x, y, sz, sz)
        p.end()

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key.Key_Escape:
            self.close()
        elif e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._confirm()


class _FreeRegionSelectorDlg(QWidget):
    """Full-screen overlay: drag any rectangle."""

    def __init__(self, title: str = "Drag to select region") -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        _scr     = QApplication.primaryScreen()
        screen   = _scr.geometry()
        self._dpr = _scr.devicePixelRatio()
        self._sw = screen.width()
        self._sh = screen.height()
        self.setGeometry(screen)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self.result: tuple[int, int, int, int] | None = None
        self._x0 = self._y0 = self._x1 = self._y1 = 0.0
        self._dragging = self._has_sel = False

        panel = QFrame(self)
        panel.setStyleSheet(
            "QFrame{background:#1e1e2e;border-radius:10px;}"
            "QLabel{color:#cdd6f4;}"
        )
        pl = QVBoxLayout(panel)
        t_lbl = QLabel(title)
        t_lbl.setStyleSheet(
            "color:#cba6f7; font-size:13px; font-weight:bold;")
        t_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_row = QWidget()
        tr = QHBoxLayout(title_row)
        tr.setContentsMargins(0, 0, 0, 0)
        tr.setSpacing(4)
        tr.addWidget(t_lbl, 1)
        tr.addWidget(
            _selection_help_icon(
                "Click and drag on the dimmed screen to draw any rectangle. "
                "Confirm saves it (Enter also works once the drag is large enough). "
                "Cancel or Escape closes without saving.",
            ),
            0,
            Qt.AlignmentFlag.AlignTop,
        )
        pl.addWidget(title_row)

        i_lbl = QLabel("Click and drag to draw the region")
        i_lbl.setStyleSheet("color:#585b70; font-size:11px;")
        i_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        btn_row = QWidget()
        bl = QHBoxLayout(btn_row)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(8)
        self._confirm_btn = QPushButton("✓  Confirm")
        self._confirm_btn.setStyleSheet(
            "background:#a6e3a1; color:#1e1e2e; border-radius:6px;"
            " padding:5px 12px; font-weight:bold;")
        self._confirm_btn.setEnabled(False)
        self._confirm_btn.clicked.connect(self._confirm)
        cancel_btn = QPushButton("✕  Cancel")
        cancel_btn.setStyleSheet(
            "background:#f38ba8; color:#1e1e2e; border-radius:6px;"
            " padding:5px 12px; font-weight:bold;")
        cancel_btn.clicked.connect(self.close)
        bl.addWidget(self._confirm_btn)
        bl.addWidget(cancel_btn)

        pl.addWidget(i_lbl)
        pl.addWidget(btn_row)
        panel.adjustSize()
        panel.move((self._sw - panel.sizeHint().width()) // 2, 16)

        self.show()

    def mousePressEvent(self, e) -> None:
        pos = e.position()
        self._x0 = self._x1 = pos.x()
        self._y0 = self._y1 = pos.y()
        self._dragging = True
        self._has_sel  = False
        self._confirm_btn.setEnabled(False)
        self.update()

    def mouseMoveEvent(self, e) -> None:
        if self._dragging:
            self._x1 = e.position().x()
            self._y1 = e.position().y()
            self.update()

    def mouseReleaseEvent(self, e) -> None:
        self._x1 = e.position().x()
        self._y1 = e.position().y()
        self._dragging = False
        if abs(self._x1 - self._x0) > 4 and abs(self._y1 - self._y0) > 4:
            self._has_sel = True
            self._confirm_btn.setEnabled(True)
        self.update()

    def _confirm(self) -> None:
        if self._has_sel:
            x = int(min(self._x0, self._x1))
            y = int(min(self._y0, self._y1))
            w = int(abs(self._x1 - self._x0))
            h = int(abs(self._y1 - self._y0))
            self.result = (round(x * self._dpr), round(y * self._dpr),
                           round(w * self._dpr), round(h * self._dpr))
        self.close()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        dim = QColor(0, 0, 0, 120)
        if self._dragging or self._has_sel:
            x = int(min(self._x0, self._x1))
            y = int(min(self._y0, self._y1))
            w = int(abs(self._x1 - self._x0))
            h = int(abs(self._y1 - self._y0))
            _fill_dim_outside_rect(p, self._sw, self._sh, x, y, w, h, dim)
            p.setPen(QPen(QColor(0, 255, 136), 2))
            p.setBrush(QColor(0, 255, 136, 30))
            p.drawRect(x, y, w, h)
            p.setPen(QColor(0, 255, 136))
            p.drawText(x + 4, y + h + 18, f"{w} × {h} px")
        else:
            p.fillRect(self.rect(), dim)
        p.end()

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key.Key_Escape:
            self.close()
        elif e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._confirm()


def _wait_for_close(widget: QWidget) -> None:
    """Spin the Qt event loop until the widget is no longer visible."""
    app = QApplication.instance()
    while widget.isVisible():
        app.processEvents()
        time.sleep(0.016)


# ── App ────────────────────────────────────────────────────────────────────────

class App(AppWindow):
    """Main application: Qt UI (AppWindow) + tracking logic."""

    # Thread-safe signals — emitted from background threads, received on main thread
    _sig_status         = pyqtSignal(str, str)    # (message, phase)
    _sig_refresh_roster = pyqtSignal()
    _sig_auto_id        = pyqtSignal(object, object, object)  # (img, splash_boxes, name_boxes)
    _sig_roster_done    = pyqtSignal(object)      # ChampionRoster
    _sig_game_ended     = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()

        # ── tracking state ────────────────────────────────────────────────────
        self.capture:        Capture | None        = None
        self._death_capture: Capture | None        = None
        self._death_region:  tuple   | None        = None
        self.roster:         ChampionRoster | None = None
        self._library                              = None
        self._track_model                          = None
        self._splash_model                         = None

        self.running              = False
        self._tracking_starting   = False   # True while _load_models_then_run hasn't finished
        self._cancel_tracking_start = False  # set by _stop to abort a load before infer threads start
        self._last_persisted_minimap_key: tuple[float | None, str] | None = None
        self.alert_radius: int    = 0
        self.alert_sound:  str    = ""
        self._last_frame: np.ndarray | None = None
        self._infer_results: list = []
        self._infer_seq:     int  = 0

        # Form overrides for transforming champions: key → current form id
        self._champ_forms: dict[str, str] = {}
        # Form toggle buttons per champion: key → {form_id: QPushButton}
        self._form_btn_map: dict[str, dict[str, "QPushButton"]] = {}

        # debug snapshots for preview dialog (updated every N inference cycles)
        self._dbg_minimap_rgb: np.ndarray | None = None
        self._dbg_results:     list              = []
        self._preview_dlg                        = None

        self._enemies_in_radius: set[str]        = set()
        self._alert_exit_time:   dict[str, float] = {}
        self._pygame_ready = False

        # thread-safe setting caches
        self._fps_cap:        int | None = 30
        self._yolo_conf:      float      = 0.35
        self._cooldown:       float      = 20.0
        self._volume:         float      = 0.80
        self._off_timeout:    float      = 2.0
        self._champ_size:     int        = 36
        self._ghost_alpha:    float      = 0.50
        self._timer_size:     int        = 10
        self._timer_alpha:    float      = 0.50
        self._arrow_size:     int        = 2
        self._arrow_alpha:    float      = 0.50
        self._dot_size:       int        = 5
        self._ring_thickness: int        = 2

        # Minimap overlay (ghosts, dots, alert ring): always / never / hold / toggle
        self._ghost_marker_mode:   str  = "always"
        self._ghost_hotkey_vk:     int  = 0x56   # 'V' — default when using hold/toggle
        self._ghost_toggle_visible: bool = False
        self._ghost_key_prev_down:  bool = False
        self._ghost_key_listen:     bool = False

        self._stop_watch   = threading.Event()
        self._watch_thread: threading.Thread | None = None
        self._init_done    = False
        self._persisted_settings_mtime: float | None = None

        # overlay — QApplication already running from main.py
        self._overlay = QtOverlay()

        # connect cross-thread signals → main-thread slots
        self._sig_status.connect(self._set_status)
        self._sig_refresh_roster.connect(self._refresh_roster_display)
        self._sig_auto_id.connect(self._auto_identify)
        self._sig_roster_done.connect(self._on_roster_identified)
        self._sig_game_ended.connect(self._on_game_ended)

        # wire up UI signal → cached setting → save
        self.fps_combo.currentTextChanged.connect(self._on_fps_change)
        self.cooldown_spin.valueChanged.connect(self._on_cooldown_change)
        self.volume_slider.valueChanged.connect(self._on_volume_change)
        self.off_timeout_spin.valueChanged.connect(self._on_off_timeout_change)
        self.champ_size_spin.valueChanged.connect(self._on_champ_size_change)
        self.ghost_alpha_spin.valueChanged.connect(self._on_ghost_alpha_change)
        self.timer_size_spin.valueChanged.connect(self._on_timer_size_change)
        self.timer_alpha_spin.valueChanged.connect(self._on_timer_alpha_change)
        self.arrow_size_spin.valueChanged.connect(self._on_arrow_size_change)
        self.arrow_alpha_spin.valueChanged.connect(self._on_arrow_alpha_change)
        self.dot_size_spin.valueChanged.connect(self._on_dot_size_change)
        self.ring_thickness_spin.valueChanged.connect(self._on_ring_thickness_change)

        self.ghost_marker_mode_combo.currentIndexChanged.connect(
            self._on_ghost_marker_mode_changed)
        self.ghost_marker_key_btn.clicked.connect(self._begin_ghost_key_capture)

        self._sync_settings_from_widgets()
        self._restore_pos()
        self._prime_persisted_settings_mtime()
        self._sync_run_button_state()
        self._init_done = True

        self._persisted_watch_debounce = QTimer(self)
        self._persisted_watch_debounce.setSingleShot(True)
        self._persisted_watch_debounce.setInterval(400)
        self._persisted_watch_debounce.timeout.connect(self._persisted_reload_after_write)

        self._persisted_watcher = QFileSystemWatcher(self)
        self._persisted_watcher.fileChanged.connect(self._schedule_persisted_reload)
        self._persisted_watcher.directoryChanged.connect(
            self._on_persisted_config_dir_changed)
        self._refresh_persisted_watch_paths()

        QTimer.singleShot(50, self._apply_capture_exclusion)
        threading.Thread(target=self._prewarm_splash_model, daemon=True).start()
        self._start_watcher()

        qa = QApplication.instance()
        if qa is not None:
            qa.installEventFilter(self)

    def _widget_under_main_window(self, w: QWidget | None) -> bool:
        while w is not None:
            if w is self:
                return True
            w = w.parentWidget()
        return False

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """Grab Tab for hotkey capture; block Tab focus navigation in the main UI."""
        if event.type() != QEvent.Type.KeyPress or not isinstance(event, QKeyEvent):
            return super().eventFilter(obj, event)
        if not isinstance(obj, QWidget) or not self._widget_under_main_window(obj):
            return super().eventFilter(obj, event)
        # Let real text fields keep Tab (if any are added under this window later)
        if isinstance(obj, (QLineEdit, QTextEdit, QPlainTextEdit)):
            return super().eventFilter(obj, event)

        ke = event
        if self._ghost_key_listen:
            if ke.key() == Qt.Key.Key_Escape:
                self._ghost_key_listen = False
                self._refresh_ghost_key_button_label()
                return True
            if ke.key() in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab):
                vk = int(ke.nativeVirtualKey())
                if vk == 0:
                    vk = 0x09
                self._complete_ghost_hotkey_assign(vk)
                return True
            if ke.key() in (
                Qt.Key.Key_Shift,
                Qt.Key.Key_Control,
                Qt.Key.Key_Alt,
                Qt.Key.Key_Meta,
                Qt.Key.Key_unknown,
            ):
                return True
            vk = int(ke.nativeVirtualKey())
            if vk == 0:
                return True
            self._complete_ghost_hotkey_assign(vk)
            return True

        if ke.key() in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab):
            return True

        return super().eventFilter(obj, event)

    # ── settings sync ─────────────────────────────────────────────────────────

    def _sync_settings_from_widgets(self) -> None:
        self._on_fps_change(self.fps_combo.currentText())
        self._on_cooldown_change(self.cooldown_spin.value())
        self._on_volume_change(self.volume_slider.value())
        self._on_off_timeout_change(self.off_timeout_spin.value())
        self._on_champ_size_change(self.champ_size_spin.value())
        self._on_ghost_alpha_change(self.ghost_alpha_spin.value())
        self._on_timer_size_change(self.timer_size_spin.value())
        self._on_timer_alpha_change(self.timer_alpha_spin.value())
        self._on_arrow_size_change(self.arrow_size_spin.value())
        self._on_arrow_alpha_change(self.arrow_alpha_spin.value())
        self._on_dot_size_change(self.dot_size_spin.value())
        self._on_ring_thickness_change(self.ring_thickness_spin.value())
        self._on_ghost_marker_mode_changed()
        self._refresh_ghost_key_button_label()

    def _on_ghost_marker_mode_changed(self, _index: int | None = None) -> None:
        data = self.ghost_marker_mode_combo.currentData()
        self._ghost_marker_mode = data if isinstance(data, str) else "always"
        use_key = self._ghost_marker_mode in ("hold", "toggle")
        self.ghost_marker_key_btn.setEnabled(use_key)
        if self._ghost_marker_mode == "toggle":
            # Avoid treating an already-held key as a fresh press when switching mode
            self._ghost_key_prev_down = _win_vk_down(self._ghost_hotkey_vk)
        else:
            self._ghost_key_prev_down = False
        self._save_pos()

    def _begin_ghost_key_capture(self) -> None:
        self._ghost_key_listen = True
        self.ghost_marker_key_btn.setText("Press a key… (Esc = cancel)")
        self.ghost_marker_key_btn.setStyleSheet(
            "background: #fab387; color: #1e1e2e; min-width: 160px;")
        self.activateWindow()
        self.raise_()
        self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)

    def _complete_ghost_hotkey_assign(self, vk: int) -> None:
        if vk <= 0:
            return
        self._ghost_hotkey_vk = vk
        self._ghost_key_listen = False
        self._refresh_ghost_key_button_label()
        self._save_pos()

    def _refresh_ghost_key_button_label(self) -> None:
        if self._ghost_key_listen:
            return
        self.ghost_marker_key_btn.setText(f"Key: {_vk_display(self._ghost_hotkey_vk)}")
        self.ghost_marker_key_btn.setStyleSheet(
            "background: #45475a; color: #cdd6f4; min-width: 120px;")

    def _ghost_hotkey_tick(self) -> None:
        """Call from render thread each frame so toggle mode sees key edges."""
        if self._ghost_marker_mode != "toggle":
            return
        vk = self._ghost_hotkey_vk
        if vk <= 0:
            return
        down = _win_vk_down(vk)
        if down and not self._ghost_key_prev_down:
            self._ghost_toggle_visible = not self._ghost_toggle_visible
        self._ghost_key_prev_down = down

    def _overlay_markers_visible_for_paint(self) -> bool:
        """Ghosts, detection dots, and alert ring — same hold/toggle/always/never."""
        mode = self._ghost_marker_mode
        if mode == "always":
            return True
        if mode == "never":
            return False
        vk = self._ghost_hotkey_vk
        if vk <= 0:
            return True
        if mode == "hold":
            return _win_vk_down(vk)
        return self._ghost_toggle_visible

    def keyPressEvent(self, e) -> None:
        if self._ghost_key_listen:
            if e.key() == Qt.Key.Key_Escape:
                self._ghost_key_listen = False
                self._refresh_ghost_key_button_label()
                e.accept()
                return
            if e.key() in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab):
                vk = int(e.nativeVirtualKey())
                if vk == 0:
                    vk = 0x09
                self._complete_ghost_hotkey_assign(vk)
                e.accept()
                return
            if e.key() in (
                Qt.Key.Key_Shift,
                Qt.Key.Key_Control,
                Qt.Key.Key_Alt,
                Qt.Key.Key_Meta,
                Qt.Key.Key_unknown,
            ):
                e.accept()
                return
            vk = int(e.nativeVirtualKey())
            if vk == 0:
                e.accept()
                return
            self._complete_ghost_hotkey_assign(vk)
            e.accept()
            return
        super().keyPressEvent(e)

    def _on_fps_change(self, val=None) -> None:
        txt = val if isinstance(val, str) else self.fps_combo.currentText()
        try:
            self._fps_cap = (None if txt.lower() == "unlimited"
                             else max(1, int(txt)))
        except Exception:
            pass
        self._save_pos()

    def _on_cooldown_change(self, val=None) -> None:
        try:
            self._cooldown = max(0.0, float(
                val if val is not None else self.cooldown_spin.value()))
        except Exception:
            pass
        self._save_pos()

    def _on_volume_change(self, val=None) -> None:
        try:
            v = int(val if val is not None else self.volume_slider.value())
            self._volume = max(0.0, min(1.0, v / 100.0))
            self._vol_lbl.setText(f"{v}%")
        except Exception:
            pass
        self._save_pos()

    def _on_off_timeout_change(self, val=None) -> None:
        try:
            self._off_timeout = max(0.5, float(
                val if val is not None else self.off_timeout_spin.value()))
        except Exception:
            pass
        self._save_pos()

    def _on_champ_size_change(self, val=None) -> None:
        try:
            self._champ_size = max(0, int(
                val if val is not None else self.champ_size_spin.value()))
        except Exception:
            pass
        self._save_pos()

    def _on_ghost_alpha_change(self, val=None) -> None:
        try:
            self._ghost_alpha = max(0, int(
                val if val is not None else self.ghost_alpha_spin.value())) / 100.0
        except Exception:
            pass
        self._save_pos()

    def _on_timer_size_change(self, val=None) -> None:
        try:
            self._timer_size = max(0, int(
                val if val is not None else self.timer_size_spin.value()))
        except Exception:
            pass
        self._save_pos()

    def _on_timer_alpha_change(self, val=None) -> None:
        try:
            self._timer_alpha = max(0, int(
                val if val is not None else self.timer_alpha_spin.value())) / 100.0
        except Exception:
            pass
        self._save_pos()

    def _on_arrow_size_change(self, val=None) -> None:
        try:
            self._arrow_size = max(0, int(
                val if val is not None else self.arrow_size_spin.value()))
        except Exception:
            pass
        self._save_pos()

    def _on_arrow_alpha_change(self, val=None) -> None:
        try:
            self._arrow_alpha = max(0, int(
                val if val is not None else self.arrow_alpha_spin.value())) / 100.0
        except Exception:
            pass
        self._save_pos()

    def _on_dot_size_change(self, val=None) -> None:
        try:
            self._dot_size = max(0, int(
                val if val is not None else self.dot_size_spin.value()))
        except Exception:
            pass
        self._save_pos()

    def _on_ring_thickness_change(self, val=None) -> None:
        try:
            self._ring_thickness = max(0, int(
                val if val is not None else self.ring_thickness_spin.value()))
        except Exception:
            pass
        self._save_pos()

    # ── UI action implementations (override AppWindow stubs) ──────────────────

    def _primary_screen_dpr(self) -> tuple[int, int, float]:
        """Logical screen size + devicePixelRatio for the primary screen.

        Returns (logical_w, logical_h, dpr). To get physical coordinates
        multiply logical values by dpr (same as _RegionSelectorDlg._confirm).
        """
        scr = QApplication.primaryScreen()
        if scr is None:
            return 1920, 1080, 1.0
        dpr = scr.devicePixelRatio()
        g   = scr.geometry()
        return g.width(), g.height(), dpr

    def _sync_run_button_state(self, *, starting: bool = False) -> None:
        """starting=True while models load (running not yet True) — show Stop like legacy UI."""
        show_stop = self.running or starting
        if show_stop:
            self.run_btn.setText("■  Stop")
            self.run_btn.setStyleSheet("background: #f38ba8; color: #1e1e2e;")
            self.run_btn.setEnabled(True)
        else:
            self.run_btn.setText("▶  Start")
            self.run_btn.setStyleSheet(f"background: {ACT}; color: #1e1e2e;")
            self.run_btn.setEnabled(self.capture is not None)

    def _toggle_run(self) -> None:
        if self.running or self._tracking_starting:
            self._stop()
        else:
            self._start()

    def _set_status(self, msg: str, phase: str = "wait") -> None:
        if phase == "sync_btn":
            self._sync_run_button_state()
            return
        super()._set_status(msg, phase)
        # Model load failure emits stop before running becomes True — reset Start/Stop toggle.
        if self._init_done and phase == "stop" and not self.running:
            self._sync_run_button_state()

    def _prime_persisted_settings_mtime(self) -> None:
        """Baseline mtime before we react to QFileSystemWatcher events."""
        p = default_persisted_settings_path()
        if p and p.is_file():
            try:
                self._persisted_settings_mtime = p.stat().st_mtime
            except OSError:
                self._persisted_settings_mtime = None
        else:
            self._persisted_settings_mtime = None

    def _refresh_persisted_watch_paths(self) -> None:
        """Watch PersistedSettings.json, or its parent Config folder if the file is absent."""
        w = self._persisted_watcher
        for p in list(w.files()):
            w.removePath(p)
        for p in list(w.directories()):
            w.removePath(p)
        path = default_persisted_settings_path()
        if path is None:
            return
        fp = path.resolve()
        try:
            if fp.is_file():
                w.addPath(str(fp))
            elif fp.parent.is_dir():
                w.addPath(str(fp.parent.resolve()))
        except Exception:
            pass

    def _schedule_persisted_reload(self, _path: str = "") -> None:
        """LoL wrote PersistedSettings.json — debounce (save is often save-to-temp + rename)."""
        if not self._init_done:
            return
        self._persisted_watch_debounce.start()

    def _on_persisted_config_dir_changed(self, _path: str) -> None:
        """File may have been created; re-attach file watch and try reload."""
        if not self._init_done:
            return
        self._refresh_persisted_watch_paths()
        self._persisted_watch_debounce.start()

    def _persisted_reload_after_write(self) -> None:
        if not self._init_done:
            return
        p = default_persisted_settings_path()
        m: float | None = None
        if p and p.is_file():
            try:
                m = p.stat().st_mtime
            except OSError:
                m = None
        if (
            m is not None
            and self._persisted_settings_mtime is not None
            and m == self._persisted_settings_mtime
        ):
            self._refresh_persisted_watch_paths()
            return
        if m is not None:
            self._persisted_settings_mtime = m
        changed = self._apply_auto_minimap_region(
            quiet=True, notify_if_missing_scale=False)
        if changed:
            self._set_status("PersistedSettings changed — minimap region updated.", "id")
        # Windows often drops single-file watches after a write; re-register.
        self._refresh_persisted_watch_paths()

    def _apply_auto_minimap_region(
        self,
        *,
        quiet: bool = False,
        notify_if_missing_scale: bool = True,
    ) -> bool:
        """
        Recompute square capture from PersistedSettings + screen size.
        Closes and reopens Capture (dxcam/mss) like a manual region change so the
        grab stream matches the new minimap scale. Restarts tracking if it was
        active or still loading models.
        Returns True if capture was recreated.
        """
        sw_log, sh_log, dpr = self._primary_screen_dpr()
        scale, corner = read_minimap_persisted()
        persisted_key: tuple[float | None, str] = (scale, corner)
        if scale is None:
            if notify_if_missing_scale and not quiet:
                self._set_status("MinimapScale not in PersistedSettings — using 1.0", "wait")
            s = 1.0
        else:
            s = scale
        x_l, y_l, w_l, h_l = auto_minimap_region(sw_log, sh_log, s, margin=8, corner=corner)
        region = (round(x_l * dpr), round(y_l * dpr), round(w_l * dpr), round(h_l * dpr))
        same_region = self.capture is not None and tuple(self.capture.region) == region
        same_key = self._last_persisted_minimap_key == persisted_key
        if same_region and same_key:
            return False

        was_active = self.running or self._tracking_starting
        if was_active:
            self._stop()
        x, y, w, h = region
        if self.capture is not None:
            try:
                self.capture.close()
            except Exception:
                pass
        self.capture = Capture(region)
        self._last_persisted_minimap_key = persisted_key
        self.region_lbl.setText(f"{w}×{h}  at ({x}, {y})  [auto]")
        self._sync_run_button_state()
        self._sync_death_capture()
        self._save_pos()
        if was_active:
            self._start()
        if not quiet:
            self._set_status("Minimap region set from PersistedSettings.", "id")
        return True

    def _apply_capture_exclusion(self) -> None:
        hidden = self.capture_hidden_cb.isChecked()
        try:
            WDA_NONE               = 0x00
            WDA_EXCLUDEFROMCAPTURE = 0x11
            hwnd = int(self.winId())
            ctypes.windll.user32.SetWindowDisplayAffinity(
                hwnd, WDA_EXCLUDEFROMCAPTURE if hidden else WDA_NONE)
        except Exception:
            pass
        self._overlay.set_capture_hidden(hidden)   # mirror the main-window checkbox
        self._save_pos()

    def _select_region(self) -> None:
        sel = _RegionSelectorDlg()
        _wait_for_close(sel)
        if sel.result:
            was_running = self.running
            if was_running:
                self._stop()
            x, y, w, h = sel.result
            self.capture = Capture(sel.result)
            self.region_lbl.setText(f"{w}×{h}  at ({x}, {y})")
            self._sync_run_button_state()
            self._sync_death_capture()
            self._save_pos()
            if was_running:
                self._start()

    def _select_death_region(self) -> None:
        sel = _FreeRegionSelectorDlg("Drag to select Death Strip")
        _wait_for_close(sel)
        if sel.result:
            x, y, w, h = sel.result
            self._death_region = sel.result
            if self._death_capture:
                self._death_capture.close()
            self._death_capture = Capture(sel.result)
            self.death_region_lbl.setText(
                f"{w}×{h}  at ({x}, {y})  [manual]")
            self._save_pos()

    def _auto_death_region(self) -> None:
        if self.capture is None:
            self._set_status("Select minimap region first.", "stop")
            return
        self._death_region = None
        self._sync_death_capture()

    def _select_alert_radius(self) -> None:
        # Small inline dialog: show minimap + circle preview
        dlg = QDialog(self)
        dlg.setWindowTitle("Set Alert Radius")
        dlg.setModal(True)
        dlg.setStyleSheet(f"QWidget{{background:{BG};color:{FG};}}"
                          "QPushButton{border-radius:6px;padding:6px 14px;"
                          "font-weight:bold;}")
        vl = QVBoxLayout(dlg)
        vl.addWidget(QLabel("Drag the slider to set the alert radius (0 = off)."))

        preview_lbl = QLabel()
        preview_lbl.setFixedSize(300, 300)
        preview_lbl.setStyleSheet("background:black;")
        vl.addWidget(preview_lbl, alignment=Qt.AlignmentFlag.AlignCenter)

        sl_row = QWidget()
        sl_layout = QHBoxLayout(sl_row)
        sl_layout.addWidget(QLabel("Radius:"))
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 400)
        slider.setValue(self.alert_radius)
        sl_layout.addWidget(slider, 1)
        r_lbl = QLabel(f"{self.alert_radius} px" if self.alert_radius else "Off")
        r_lbl.setFixedWidth(70)
        sl_layout.addWidget(r_lbl)
        vl.addWidget(sl_row)

        _radius_holder = [self.alert_radius]

        def _update_preview(val: int) -> None:
            import cv2
            _radius_holder[0] = val
            r_lbl.setText(f"{val} px" if val else "Off")
            sz = 300

            # Always use the bundled minimap screenshot as background so the
            # preview is stable regardless of whether the game is running.
            bg_tmp = _load_preview_bg(sz)
            arr = bg_tmp if bg_tmp is not None else np.zeros((sz, sz, 3), np.uint8)

            # Scale radius: val is in physical-pixel minimap units;
            # cap_sz is the capture width in physical pixels.
            cap_sz = self.capture.region[2] if self.capture is not None else sz
            r_scaled = int(val * sz / max(cap_sz, 1))

            cx, cy = sz // 2, sz // 2
            if r_scaled > 0:
                cv2.circle(arr, (cx, cy), r_scaled, (200, 180, 60), 2)
            cv2.circle(arr, (cx, cy), max(2, self._dot_size), (255, 220, 50), -1)

            h_, w_, c_ = arr.shape
            qimg = QImage(arr.tobytes(), w_, h_, w_ * c_,
                          QImage.Format.Format_RGB888)
            preview_lbl.setPixmap(QPixmap.fromImage(qimg))

        slider.valueChanged.connect(_update_preview)
        _update_preview(self.alert_radius)

        btn_row = QWidget()
        bl = QHBoxLayout(btn_row)
        bl.setContentsMargins(0, 0, 0, 0)
        ok = QPushButton("✓  Confirm")
        ok.setStyleSheet("background:#a6e3a1; color:#1e1e2e;")
        ok.clicked.connect(dlg.accept)
        cancel = QPushButton("✕  Cancel")
        cancel.setStyleSheet("background:#f38ba8; color:#1e1e2e;")
        cancel.clicked.connect(dlg.reject)
        bl.addWidget(ok)
        bl.addWidget(cancel)
        vl.addWidget(btn_row)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.alert_radius = _radius_holder[0]
            self._radius_lbl.setText(
                f"{self.alert_radius} px" if self.alert_radius else "Off")
            self._save_pos()

    def _pick_alert_sound(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Alert Sound", str(Path.home()),
            "Audio Files (*.wav *.mp3 *.ogg);;All Files (*)",
        )
        if path:
            self.alert_sound = path
            self._sound_lbl.setText("✓")
            self._sound_lbl.setStyleSheet(f"color: {ACT}; font-size:12px;")
        self._save_pos()

    def _test_alert_sound(self) -> None:
        threading.Thread(target=self._play_alert, daemon=True).start()

    def _preview_overlay(self) -> None:
        import cv2

        # Bring existing dialog to front rather than opening a second one
        if self._preview_dlg is not None:
            try:
                self._preview_dlg.raise_()
                self._preview_dlg.activateWindow()
                return
            except RuntimeError:
                self._preview_dlg = None

        SZ = 380   # single-panel preview size

        _DEMO_COLORS = {
            "player": (255, 220,  50),
            "ally":   ( 80, 220, 255),
            "enemy":  (255,  80,  80),
        }

        def _arr_to_pixmap(arr_rgb: np.ndarray) -> QPixmap:
            h, w, c = arr_rgb.shape
            return QPixmap.fromImage(
                QImage(arr_rgb.tobytes(), w, h, w * c,
                       QImage.Format.Format_RGB888))

        # Pre-load Kayle icon for demo enemy
        _kayle_icon: np.ndarray | None = None
        try:
            _kayle_path = Path(__file__).parent / "cache" / "icons" / "Kayle.png"
            raw = _kayle_path.read_bytes()
            _bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if _bgr is not None:
                _kayle_icon = cv2.cvtColor(_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            pass

        dlg = QDialog(self)
        dlg.setWindowTitle("Overlay Preview")
        dlg.setModal(False)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.setStyleSheet(f"background:{BG}; color:{FG};")

        vl = QVBoxLayout(dlg)
        vl.setSpacing(8)

        hdr = QLabel("Updates every 500 ms while tracking is active")
        hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hdr.setStyleSheet(f"color:{DIM}; font-size:11px;")
        vl.addWidget(hdr)

        map_img = QLabel()
        map_img.setFixedSize(SZ, SZ)
        map_img.setStyleSheet("background:#11111b; border-radius:4px;")
        map_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vl.addWidget(map_img, alignment=Qt.AlignmentFlag.AlignCenter)

        status_lbl = QLabel()
        status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_lbl.setStyleSheet(f"color:{DIM};")
        vl.addWidget(status_lbl)

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(
            "background:#585b70; color:#cdd6f4; border-radius:6px;"
            " padding:6px 14px;")
        close_btn.clicked.connect(dlg.close)
        vl.addWidget(close_btn)

        def _draw_icon(canvas: np.ndarray, icon_rgb: np.ndarray,
                       cx: int, cy: int, sz: int,
                       border_col: tuple) -> None:
            """Paste a circular-clipped champion icon (matches in-game overlay)."""
            _blend_ghost_icon_circle_rgb(
                canvas, icon_rgb, cx, cy, sz, 1.0, border_col)

        def _refresh() -> None:
            dbg_frame = self._dbg_minimap_rgb
            dbg_res   = self._dbg_results

            _bg_tmp = _load_preview_bg(SZ)
            bg_base = (_bg_tmp if _bg_tmp is not None
                       else np.zeros((SZ, SZ, 3), np.uint8))

            if dbg_frame is not None and self.running:
                # ── Live mode: mirror actual overlay_qt paintEvent ────────────
                # Use the real minimap frame as background.
                base  = cv2.resize(dbg_frame, (SZ, SZ))
                scale = SZ / max(dbg_frame.shape[1], 1)
                dot   = max(2, self._dot_size)
                csz   = max(8, self._champ_size)
                g_alpha = self._ghost_alpha   # 0-1 float

                lib = self._library

                # Ghost markers — off-map / dead enemies (icon + timer)
                if lib is not None and self._overlay_markers_visible_for_paint():
                    for ci, ch in enumerate(lib.all):
                        if lib.team[ci] in ("ally", "player"):
                            continue
                        st = lib._state.get(ch.key)
                        if st is None or st.pos is None:
                            continue
                        if not getattr(st, "ghost_active", False) and \
                                not getattr(st, "dead", False):
                            continue
                        gx = int(st.pos[0] * scale)
                        gy = int(st.pos[1] * scale)
                        icon = lib.icon_imgs.get(ch.key)
                        if icon is not None:
                            _blend_ghost_icon_circle_rgb(
                                base, icon, gx, gy, csz, g_alpha)
                        half = csz // 2
                        elapsed = time.perf_counter() - st.last_seen
                        label   = (f"{int(elapsed)}s" if elapsed < 60
                                   else f"{int(elapsed//60)}m{int(elapsed%60):02d}s")
                        cv2.putText(base, label, (gx - half, gy + half + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                    (200, 100, 100), 1, cv2.LINE_AA)

                vis = self._overlay_markers_visible_for_paint()
                # Active detection dots (what overlay draws for on-map champs)
                if vis:
                    for r in dbg_res:
                        cx_r = int(r["location"][0] * scale)
                        cy_r = int(r["location"][1] * scale)
                        col  = _DEMO_COLORS.get(r.get("team", ""), (200, 200, 200))
                        cv2.circle(base, (cx_r, cy_r), dot, col, -1)

                    # Alert radius ring around player
                    thick = self._ring_thickness or 2
                    if self.alert_radius > 0 and lib is not None:
                        pkey = lib.roster.player.key
                        pst  = lib._state.get(pkey)
                        if pst and pst.pos:
                            cap_sz = (self.capture.region[2]
                                      if self.capture is not None else 1)
                            r_sc = int(self.alert_radius * SZ / max(cap_sz, 1))
                            px_s = int(pst.pos[0] * scale)
                            py_s = int(pst.pos[1] * scale)
                            cv2.circle(base, (px_s, py_s), r_sc,
                                       (200, 180, 60), thick)

                map_img.setPixmap(_arr_to_pixmap(base))
                n = len(dbg_res)
                status_lbl.setText(
                    f"Tracking — {n} champion{'s' if n != 1 else ''} detected")
                status_lbl.setStyleSheet(f"color:{ACT};")

            else:
                # ── Demo mode: replicate overlay appearance with static data ──
                demo  = bg_base.copy()
                s     = SZ / 400.0
                dot   = max(2, self._dot_size)
                thick = self._ring_thickness or 2
                csz   = max(20, self._champ_size or 36)
                g_alpha = min(1.0, max(0.0, self._ghost_alpha))

                player_pos = (int(90  * s), int(310 * s))
                enemy_pos  = (int(300 * s), int(100 * s))
                demo_r     = max(20, self.alert_radius or int(SZ * 0.22))

                vis_demo = self._overlay_markers_visible_for_paint()
                if vis_demo:
                    # Alert radius ring (gold)
                    cv2.circle(demo, player_pos, demo_r, (200, 180, 60), thick)

                    # Player dot (gold) + ally dots (cyan)
                    cv2.circle(demo, player_pos, dot, _DEMO_COLORS["player"], -1)
                    for pos in [(int(130*s), int(270*s)), (int(70*s), int(240*s))]:
                        cv2.circle(demo, pos, dot, _DEMO_COLORS["ally"], -1)

                # Kayle as ghost enemy: circular icon + timer (matches overlay)
                ex, ey = enemy_pos
                half = csz // 2
                if vis_demo:
                    if _kayle_icon is not None:
                        _blend_ghost_icon_circle_rgb(
                            demo, _kayle_icon, ex, ey, csz, g_alpha)
                    cv2.putText(demo, "8s",
                                (ex - half, ey + half + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                (200, 100, 100), 1, cv2.LINE_AA)
                    cv2.putText(demo, "Kayle",
                                (ex - half, ey - half - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                (200, 80, 80), 1, cv2.LINE_AA)

                map_img.setPixmap(_arr_to_pixmap(demo))

                if self.running:
                    status_lbl.setText(
                        "Tracking active — waiting for first detections…")
                    status_lbl.setStyleSheet(f"color:{ACT};")
                else:
                    status_lbl.setText("Not tracking — demo mode")
                    status_lbl.setStyleSheet(f"color:{DIM};")

        _refresh()

        timer = QTimer(dlg)
        timer.timeout.connect(_refresh)
        timer.start(500)

        dlg.show()
        self._preview_dlg = dlg
        dlg.destroyed.connect(lambda: setattr(self, "_preview_dlg", None))

    @staticmethod
    def _load_champ_keys() -> dict[str, str]:
        """
        Returns {display_name: champion_key} for manual roster entry / completer.
        Uses cache/champion_registry.json, then adds cache/icons/*.png not listed.
        """
        import re

        display: dict[str, str] = {}
        reg_path = Path(__file__).parent / "cache" / "champion_registry.json"
        if reg_path.exists():
            try:
                raw = json.loads(reg_path.read_text(encoding="utf-8"))
                data = raw.get("data", {})
                if isinstance(data, dict):
                    for key, meta in data.items():
                        if isinstance(meta, dict) and "name" in meta:
                            display[str(meta["name"])] = key
            except Exception:
                pass

        icons_dir = Path(__file__).parent / "cache" / "icons"
        if icons_dir.exists():
            for p in sorted(icons_dir.glob("*.png")):
                k = p.stem
                if k in display.values():
                    continue
                pretty = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", k)
                if pretty not in display:
                    display[pretty] = k

        return display

    def _enter_manually(self) -> None:
        from PyQt6.QtCore import Qt as _Qt

        # ── load champion list from local cache / registry ─
        champ_map = self._load_champ_keys()   # {display_name: key}
        names_sorted = sorted(champ_map.keys())

        dlg = QDialog(self)
        dlg.setWindowTitle("Enter Champions Manually")
        dlg.setMinimumWidth(460)
        dlg.setStyleSheet(
            f"QWidget{{background:{BG};color:{FG};}}"
            "QPushButton{border-radius:6px;padding:6px 12px;font-weight:bold;}"
            f"QComboBox{{background:#313244;border:1px solid #45475a;"
            f" border-radius:6px;padding:3px 6px;color:{FG};}}"
            "QComboBox QAbstractItemView{"
            f"  background:#313244;color:{FG};selection-background-color:#45475a;}}"
        )

        vl = QVBoxLayout(dlg)
        vl.setSpacing(6)

        # ── side selector ─────────────────────────────────────────────────────
        from PyQt6.QtWidgets import QRadioButton, QButtonGroup
        side_row = QWidget()
        side_lay = QHBoxLayout(side_row)
        side_lay.setContentsMargins(0, 0, 0, 4)
        side_lay.setSpacing(12)
        side_lay.addWidget(QLabel("You are on:"))
        _rb_blue = QRadioButton("Blue side")
        _rb_red  = QRadioButton("Red side")
        _rb_blue.setChecked(True)
        _side_grp = QButtonGroup(dlg)
        _side_grp.addButton(_rb_blue, 0)
        _side_grp.addButton(_rb_red,  1)
        side_lay.addWidget(_rb_blue)
        side_lay.addWidget(_rb_red)
        side_lay.addStretch()
        vl.addWidget(side_row)

        # section header labels (mutated when side flips)
        _your_team_lbl = QLabel()
        _enemy_lbl     = QLabel()

        def _apply_side_labels() -> None:
            on_blue = _rb_blue.isChecked()
            if on_blue:
                _your_team_lbl.setText("── Blue side  (your team)")
                _your_team_lbl.setStyleSheet(
                    f"color:{ALLY}; font-size:11px; font-weight:bold; padding-top:6px;")
                _enemy_lbl.setText("── Red side  (enemies)")
                _enemy_lbl.setStyleSheet(
                    f"color:{ENE}; font-size:11px; font-weight:bold; padding-top:6px;")
            else:
                _your_team_lbl.setText("── Red side  (your team)")
                _your_team_lbl.setStyleSheet(
                    f"color:{ENE}; font-size:11px; font-weight:bold; padding-top:6px;")
                _enemy_lbl.setText("── Blue side  (enemies)")
                _enemy_lbl.setStyleSheet(
                    f"color:{ALLY}; font-size:11px; font-weight:bold; padding-top:6px;")

        _rb_blue.toggled.connect(lambda _: _apply_side_labels())

        def _make_combo() -> "QComboBox":
            cb = QComboBox()
            cb.setEditable(True)
            cb.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
            cb.addItem("")                    # blank = "not selected"
            cb.addItems(names_sorted)
            comp = QCompleter(names_sorted, cb)
            comp.setCaseSensitivity(_Qt.CaseSensitivity.CaseInsensitive)
            comp.setFilterMode(_Qt.MatchFlag.MatchContains)
            cb.setCompleter(comp)
            return cb

        # ── your team: player + 4 allies ──────────────────────────────────────
        vl.addWidget(_your_team_lbl)
        blue_grid = QWidget()
        bg_lay = QHBoxLayout(blue_grid)
        bg_lay.setContentsMargins(0, 0, 0, 0)
        bg_lay.setSpacing(6)

        blue_combos: list[QComboBox] = []
        blue_labels = ["Player", "Ally 1", "Ally 2", "Ally 3", "Ally 4"]
        for lbl_txt in blue_labels:
            col = QWidget()
            cl  = QVBoxLayout(col)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(2)
            l = QLabel(lbl_txt)
            l.setStyleSheet(f"color:{DIM}; font-size:10px;")
            cb = _make_combo()
            cl.addWidget(l)
            cl.addWidget(cb)
            bg_lay.addWidget(col)
            blue_combos.append(cb)
        vl.addWidget(blue_grid)

        # ── opposing side: 5 enemies ──────────────────────────────────────────
        vl.addWidget(_enemy_lbl)
        red_grid = QWidget()
        rg_lay = QHBoxLayout(red_grid)
        rg_lay.setContentsMargins(0, 0, 0, 0)
        rg_lay.setSpacing(6)

        red_combos: list[QComboBox] = []
        for i in range(1, 6):
            col = QWidget()
            cl  = QVBoxLayout(col)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(2)
            l = QLabel(f"Enemy {i}")
            l.setStyleSheet(f"color:{DIM}; font-size:10px;")
            cb = _make_combo()
            cl.addWidget(l)
            cl.addWidget(cb)
            rg_lay.addWidget(col)
            red_combos.append(cb)
        vl.addWidget(red_grid)

        _apply_side_labels()  # initialise labels to blue-side defaults

        # ── auto-fill from existing roster ────────────────────────────────────
        def _set_combo(cb: "QComboBox", name: str) -> None:
            """Select name in combo; fall back to free-text if not in list."""
            if not name:
                return
            idx = cb.findText(name, _Qt.MatchFlag.MatchFixedString)
            if idx >= 0:
                cb.setCurrentIndex(idx)
            else:
                # try case-insensitive fallback
                idx = cb.findText(name, _Qt.MatchFlag.MatchFixedString)
                if idx >= 0:
                    cb.setCurrentIndex(idx)
                else:
                    cb.setCurrentText(name)

        if self.roster is not None:
            r = self.roster
            # side radio
            if r.enemy_side == "red":
                _rb_blue.setChecked(True)
            else:
                _rb_red.setChecked(True)
            _apply_side_labels()

            # player + allies
            _set_combo(blue_combos[0], r.player.name)
            for i, ally in enumerate(r.allies[:4]):
                _set_combo(blue_combos[i + 1], ally.name)

            # enemies
            for i, enemy in enumerate(r.enemies[:5]):
                _set_combo(red_combos[i], enemy.name)

        # ── buttons ───────────────────────────────────────────────────────────
        btn_row = QWidget()
        bl = QHBoxLayout(btn_row)
        bl.setContentsMargins(0, 6, 0, 0)
        ok     = QPushButton("✓  Confirm")
        ok.setStyleSheet("background:#a6e3a1; color:#1e1e2e;")
        cancel = QPushButton("✕  Cancel")
        cancel.setStyleSheet("background:#f38ba8; color:#1e1e2e;")
        ok.clicked.connect(dlg.accept)
        cancel.clicked.connect(dlg.reject)
        bl.addWidget(ok)
        bl.addWidget(cancel)
        vl.addWidget(btn_row)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        def _combo_to_champ(cb: "QComboBox") -> Champion | None:
            txt = cb.currentText().strip()
            if not txt:
                return None
            dd_key = champ_map.get(txt, txt.replace(" ", ""))
            return Champion(name=txt, key=dd_key)

        player_c = _combo_to_champ(blue_combos[0])
        if player_c is None:
            return   # player is required

        allies  = [c for cb in blue_combos[1:] if (c := _combo_to_champ(cb))]
        enemies = [c for cb in red_combos    if (c := _combo_to_champ(cb))]

        was_running = self.running
        if was_running:
            self._stop()
        self.roster   = ChampionRoster(player=player_c, allies=allies,
                                       enemies=enemies)
        self._library = None
        self._build_roster_display()
        self._set_status("Manual roster set — start tracker when ready.", "id")
        threading.Thread(target=self._rebuild_library, daemon=True).start()
        if was_running:
            self._start()

    # ── tracking start / stop ─────────────────────────────────────────────────

    def _start(self, auto: bool = False) -> None:
        if not self.capture:
            return
        self._cancel_tracking_start = False
        if not auto:
            self._set_status("Loading models…", "id")
        self._sync_run_button_state(starting=True)
        self._overlay.start(self.capture.region)
        self._apply_capture_exclusion()
        self._tracking_starting = True
        threading.Thread(target=self._load_models_then_run, daemon=True).start()

    def _stop(self) -> None:
        self._cancel_tracking_start = True
        self.running = False
        self._overlay.stop()
        self._sync_run_button_state()
        self._set_status("Stopped.", "stop")

    def _rebuild_library(self) -> None:
        try:
            if not self.roster:
                return
            from tracker import ChampionLibrary
            minimap_px  = self.capture.region[2] if self.capture else 400
            self._library = ChampionLibrary(self.roster, minimap_size=minimap_px)
        except Exception:
            pass

    def _load_models_then_run(self) -> None:
        try:
            try:
                self._rebuild_library()
                if self._track_model is None:
                    from backend import load_model
                    self._track_model = load_model()
            except Exception as e:
                self._sig_status.emit(f"Model load error: {e}", "stop")
                return
            if self._cancel_tracking_start:
                return
            self.running        = True
            self._infer_results = []
            threading.Thread(target=self._infer_loop,  daemon=True).start()
            threading.Thread(target=self._render_loop, daemon=True).start()
        finally:
            self._tracking_starting = False
            if not self.running:
                self._sig_status.emit("", "sync_btn")

    # ── inference loop ────────────────────────────────────────────────────────

    def _infer_loop(self) -> None:
        from tracker import track_frame
        _death_t: float = 0.0
        while self.running:
            t0      = time.perf_counter()
            roster  = self.roster
            library = self._library
            model_  = self._track_model

            if not roster or not library or not model_:
                time.sleep(0.01)
                continue

            if not lol_client_is_foreground():
                self._infer_results = []
                time.sleep(0.05)
                continue

            try:
                frame = self.capture.grab()
            except Exception:
                time.sleep(0.05)
                continue
            arr = np.array(frame)
            self._last_frame = arr

            try:
                results = track_frame(arr, library, model=model_, now=t0,
                                      conf=self._yolo_conf,
                                      off_timeout=self._off_timeout)
            except Exception:
                results = []

            self._infer_results = results
            self._infer_seq    += 1

            # update preview-dialog snapshot every 10 frames (cheap copy)
            if self._infer_seq % 10 == 0:
                self._dbg_minimap_rgb = arr.copy()
                self._dbg_results     = list(results)

            # death panel — temporarily disabled
            # try:
            #     dc = self._death_capture
            #     if dc is not None and (t0 - _death_t) >= 0.5:
            #         death_frame = np.array(dc.grab())
            #         from death_panel import scan_death_panel
            #         scan_death_panel(death_frame, library, now=t0)
            #         _death_t = t0
            # except Exception:
            #     pass

            # roster status
            roster_changed = False
            for c in roster.allies + roster.enemies + [roster.player]:
                on     = library.is_visible(c.key, t0)
                new_st = STATUS_ON_MAP if on else STATUS_OFF_MAP
                if c.status != new_st:
                    c.status      = new_st
                    roster_changed = True
            if roster_changed and self.running:
                self._sig_refresh_roster.emit()

            # alert proximity
            player_key    = library.roster.player.key
            player_on_map = any(r["key"] == player_key for r in results)
            if self.alert_radius > 0 and player_on_map:
                pst = library._state.get(player_key)
                if pst and pst.pos:
                    px, py   = pst.pos
                    cooldown = self._cooldown
                    now_in: set[str] = set()
                    for r in results:
                        if r["team"] == "enemy":
                            ex, ey = r["location"]
                            if ((ex - px)**2 + (ey - py)**2)**0.5 <= self.alert_radius:
                                now_in.add(r["key"])
                    for key in (self._enemies_in_radius - now_in):
                        self._alert_exit_time[key] = t0
                    for key in (now_in - self._enemies_in_radius):
                        if t0 - self._alert_exit_time.get(key, 0.0) >= cooldown:
                            threading.Thread(target=self._play_alert,
                                             daemon=True).start()
                    self._enemies_in_radius = now_in

            cap      = self._fps_cap
            target_s = (1.0 / cap) if cap else 0.0
            rem      = target_s - (time.perf_counter() - t0)
            if rem > 0:
                time.sleep(rem)

    # ── render loop ────────────────────────────────────────────────────────────

    def _render_loop(self) -> None:
        _RENDER_FPS = 30
        _FRAME_T    = 1.0 / _RENDER_FPS
        _ui_t       = time.perf_counter()

        while self.running:
            t0 = time.perf_counter()

            try:
                self._ghost_hotkey_tick()
                show_overlay = self._overlay_markers_visible_for_paint()
                # When LoL is not foreground: remove all drawing (overlay clears to transparent).
                # When back in foreground: pass real results/library so overlay redraws.
                if not lol_client_is_foreground():
                    self._overlay.update_state(
                        [], None, t0,
                        alert_radius   = self.alert_radius,
                        champ_size     = self._champ_size,
                        ghost_alpha    = self._ghost_alpha,
                        timer_size     = self._timer_size,
                        timer_alpha    = self._timer_alpha,
                        arrow_size     = self._arrow_size,
                        arrow_alpha    = self._arrow_alpha,
                        dot_size       = self._dot_size,
                        ring_thickness = self._ring_thickness,
                        show_overlay_markers = show_overlay,
                    )
                else:
                    self._overlay.update_state(
                        self._infer_results,
                        self._library,
                        t0,
                        alert_radius   = self.alert_radius,
                        champ_size     = self._champ_size,
                        ghost_alpha    = self._ghost_alpha,
                        timer_size     = self._timer_size,
                        timer_alpha    = self._timer_alpha,
                        arrow_size     = self._arrow_size,
                        arrow_alpha    = self._arrow_alpha,
                        dot_size       = self._dot_size,
                        ring_thickness = self._ring_thickness,
                        show_overlay_markers = show_overlay,
                    )
            except Exception:
                pass

            now = time.perf_counter()
            if now - _ui_t >= 0.25:
                if self.running:
                    self._sig_status.emit("Tracking", "run")
                _ui_t = now

            rem = _FRAME_T - (time.perf_counter() - t0)
            if rem > 0:
                time.sleep(rem)

    # ── alert sound ────────────────────────────────────────────────────────────

    def _play_alert(self) -> None:
        if not self.alert_sound or not Path(self.alert_sound).exists():
            return
        try:
            import pygame
            if not self._pygame_ready:
                pygame.mixer.init()
                self._pygame_ready = True
            snd = pygame.mixer.Sound(self.alert_sound)
            snd.set_volume(self._volume)
            snd.play()
        except Exception:
            pass

    # ── auto-watcher ───────────────────────────────────────────────────────────

    def _prewarm_splash_model(self) -> None:
        try:
            from backend import load_splash_model
            self._splash_model = load_splash_model()
        except Exception:
            self._splash_model = None

    def _start_watcher(self) -> None:
        self._stop_watch.clear()
        self._watch_thread = threading.Thread(
            target=self._watcher_loop, daemon=True)
        self._watch_thread.start()

    def _watcher_loop(self) -> None:
        while not self._stop_watch.is_set():
            # ── phase 1: wait for LoL to launch ──────────────────────────────
            while not self._stop_watch.is_set():
                if _lol_running():
                    break
                time.sleep(2)
            if self._stop_watch.is_set():
                return

            self._sig_status.emit("League detected — waiting for loading screen…", "scan")

            splash_model = self._splash_model
            if splash_model is None:
                try:
                    from backend import load_splash_model
                    splash_model = self._splash_model = load_splash_model()
                except Exception:
                    pass

            # Only count seconds when we actually ran YOLO — alt-tab /
            # transition frames don't eat into the budget.
            _SPLASH_TIMEOUT = 20
            consecutive  = 0
            scan_elapsed = 0.0
            _lol_check_t = 0.0   # throttle the expensive tasklist call

            # ── phase 2: scan for loading screen ─────────────────────────────
            detected = False
            while not self._stop_watch.is_set():
                # Check LoL still running at most once every 5 s (tasklist is slow)
                now_t = time.perf_counter()
                if now_t - _lol_check_t >= 5.0:
                    if not _lol_running():
                        self._sig_status.emit(
                            "Waiting for League of Legends…", "wait")
                        break
                    _lol_check_t = now_t

                try:
                    img = _grab_fullscreen()
                except Exception:
                    time.sleep(0.5)
                    continue

                # Skip black/transition frames but do NOT count them against
                # the timeout — they are not real "scan" cycles.
                if _is_black_screen(img):
                    consecutive = 0
                    time.sleep(0.3)
                    continue

                if scan_elapsed >= _SPLASH_TIMEOUT:
                    self._sig_status.emit(
                        "Loading screen not detected — enter manually", "wait")
                    return

                n_cards      = 0
                splash_boxes: list = []
                name_boxes:   list = []
                if splash_model is not None:
                    try:
                        from backend import detect_cards
                        result = detect_cards(splash_model, img)
                        if isinstance(result, tuple):
                            splash_boxes, name_boxes = result
                        else:
                            splash_boxes = result
                        n_cards = len(splash_boxes)
                    except Exception:
                        n_cards = 0

                consecutive   = (consecutive + 1) if n_cards == 10 else 0
                # Only advance elapsed when we actually attempted a scan
                scan_elapsed += 0.5
                remaining     = max(0, int(_SPLASH_TIMEOUT - scan_elapsed))
                self._sig_status.emit(
                    f"Scanning… ({n_cards}/10 cards)  {remaining}s", "scan")

                if consecutive >= 3:
                    self._sig_auto_id.emit(img, splash_boxes, name_boxes)
                    detected = True
                    break
                time.sleep(0.5)

            if detected or self._stop_watch.is_set():
                return
            # LoL closed mid-scan — loop back to wait for it again

    def _auto_identify(self, screenshot: Image.Image,
                       splash_boxes: list | None = None,
                       name_boxes:   list | None = None) -> None:
        self._set_status("Identifying champions…", "id")
        threading.Thread(target=self._do_identify,
                         args=(screenshot, splash_boxes, name_boxes),
                         daemon=True).start()

    def _do_identify(self, screenshot: Image.Image,
                     splash_boxes: list | None = None,
                     name_boxes:   list | None = None) -> None:
        try:
            from match_start import SkinDatabase, identify_all
            db     = SkinDatabase()
            roster = identify_all(screenshot, db,
                                  splash_boxes=splash_boxes,
                                  name_boxes=name_boxes)
        except Exception as e:
            self._sig_status.emit(f"Identify error: {e}", "stop")
            return

        if roster is None:
            self._sig_status.emit("Detection failed — enter manually.", "stop")
            return

        try:
            _ROSTER_FILE.write_text(json.dumps({
                "player":  {"name": roster.player.name,  "key": roster.player.key},
                "allies":  [{"name": c.name, "key": c.key} for c in roster.allies],
                "enemies": [{"name": c.name, "key": c.key} for c in roster.enemies],
            }, indent=2))
        except Exception:
            pass

        self._sig_roster_done.emit(roster)
        threading.Thread(target=self._watch_game_end, daemon=True).start()

    def _on_roster_identified(self, roster) -> None:
        """Main-thread slot: apply identified roster and start tracker."""
        self.roster        = roster
        self._library      = None
        self._champ_forms  = {}   # reset every new game — all forms start at base
        self._form_btn_map = {}
        self._build_roster_display()
        self._set_status("Champions identified — starting tracker…", "id")
        QTimer.singleShot(3000, self._auto_start_tracking)

    def _auto_start_tracking(self) -> None:
        if self.capture is None:
            sw_log, sh_log, dpr = self._primary_screen_dpr()
            scale, corner = read_minimap_persisted()
            if scale is None:
                scale = 1.0
            x_l, y_l, w_l, h_l = auto_minimap_region(sw_log, sh_log, scale, margin=8, corner=corner)
            x, y, w, h = round(x_l * dpr), round(y_l * dpr), round(w_l * dpr), round(h_l * dpr)
            self.capture = Capture((x, y, w, h))
            self.region_lbl.setText(f"{w}×{h}  at ({x}, {y})  [auto]")
            self._sync_run_button_state()
            self._set_status("Auto minimap from PersistedSettings — starting tracker…", "id")
        if self._death_capture is None:
            self._sync_death_capture()
        self._start(auto=True)

    def _sync_death_capture(self) -> None:
        if self._death_region:
            region = self._death_region
        elif self.capture is not None:
            from death_panel import death_panel_region
            region = death_panel_region(self.capture.region)
            x, y, w, h = region
            self.death_region_lbl.setText(f"{w}×{h}  at ({x}, {y})  [auto]")
        else:
            return
        if self._death_capture is not None:
            try:
                self._death_capture.close()
            except Exception:
                pass
        self._death_capture = Capture(region)

    def _watch_game_end(self) -> None:
        while not self._stop_watch.is_set():
            time.sleep(3)
            if not _lol_running():
                self._sig_game_ended.emit()
                return

    def _on_game_ended(self) -> None:
        self._stop()
        self.roster   = None
        self._library = None
        self._build_roster_display()
        self._set_status("Game ended — waiting for League of Legends…", "wait")
        self._start_watcher()

    # ── splash debug ───────────────────────────────────────────────────────────

    def _save_splash_debug(self, img: Image.Image,
                           splash_boxes: list, name_boxes: list) -> None:
        try:
            import cv2
            arr = np.array(img.convert("RGB"))
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            for i, (x1, y1, x2, y2) in enumerate(splash_boxes):
                cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 80), 3)
                cv2.putText(bgr, f"card {i+1}", (x1 + 4, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 2)
            for i, (x1, y1, x2, y2) in enumerate(name_boxes):
                cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 180, 255), 2)
                cv2.putText(bgr, f"name {i+1}", (x1 + 4, y1 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)
            cv2.imwrite("debug_splash_latest.jpg", bgr)
        except Exception:
            pass

    # ── persistence ────────────────────────────────────────────────────────────

    # ── Kayn form selector ─────────────────────────────────────────────────────

    # Transforming champion form data: key → [(form_id, label, local_filename|None)]
    _FORM_CHAMPIONS: dict[str, list] = {
        "Kayn": [
            ("base", "Base",            None),
            ("slay", "Rhaast",          "kayn_slay_square.png"),
            ("ass",  "Shadow Assassin", "kayn_ass_square.png"),
        ],
        "Kayle": [
            ("base",  "Base (Lv1–10)", None),
            ("lvl11", "Lv11+",         "kayle_square_lvl11.png"),
        ],
    }

    def _after_roster_row(self, c, row) -> None:
        """Inject form picker for transforming champions (Kayn, Kayle)."""
        if c.key not in self._FORM_CHAMPIONS:
            return
        from PyQt6.QtWidgets import QButtonGroup, QHBoxLayout
        from PyQt6.QtGui     import QIcon
        from PyQt6.QtCore    import QSize

        wrap = QWidget(self._roster_container)
        hl   = QHBoxLayout(wrap)
        hl.setContentsMargins(22, 0, 0, 2)
        hl.setSpacing(6)

        lbl = QLabel("Form:")
        lbl.setStyleSheet(f"color:{DIM}; font-size:11px;")
        hl.addWidget(lbl)

        grp  = QButtonGroup(wrap)
        btns: dict[str, QPushButton] = {}
        for form, label, fname in self._FORM_CHAMPIONS[c.key]:
            btn = QPushButton()
            btn.setCheckable(True)
            btn.setFixedSize(52, 52)
            btn.setToolTip(label)
            if fname:
                px = QPixmap(str(ASSETS_DIR / fname))
                if not px.isNull():
                    btn.setIcon(QIcon(px))
                    btn.setIconSize(QSize(42, 42))
                    btn.setText("")
                else:
                    btn.setText(label[:4])
            else:
                btn.setText("—")
            btn.setStyleSheet(
                "QPushButton{"
                "  border:2px solid #45475a; border-radius:6px;"
                "  background:#1e1e2e; padding:0;}"
                "QPushButton:checked{"
                "  border:3px solid #cba6f7; background:#313244;}"
                "QPushButton:hover{"
                "  border:2px solid #7f849c; background:#24243a;}"
            )
            btn.clicked.connect(
                lambda _c, k=c.key, f=form: self._set_champ_form(k, f))
            grp.addButton(btn)
            hl.addWidget(btn)
            btns[form] = btn

        hl.addStretch()
        self._roster_layout.addWidget(wrap)
        self._form_btn_map[c.key] = btns

        # Current form — always "base" on a fresh game (dict will be empty)
        current = self._champ_forms.get(c.key, "base")
        btns.get(current, btns["base"]).setChecked(True)

    def _set_champ_form(self, champ_key: str, form: str) -> None:
        """Swap a transforming champion's icon + identification features."""
        self._champ_forms[champ_key] = form

        # Update highlight on the form buttons
        btns = self._form_btn_map.get(champ_key, {})
        for fid, btn in btns.items():
            btn.setChecked(fid == form)

        lib = self._library
        if lib is None:
            self._save_pos()
            return

        idx = next((i for i, c in enumerate(lib.all)
                    if c.key == champ_key), None)
        if idx is None:
            self._save_pos()
            return

        from PIL import Image as _Image

        # Resolve the local filename for the chosen form (None → cache/icons base)
        fname = next(
            (fn for fid, _lbl, fn in self._FORM_CHAMPIONS.get(champ_key, [])
             if fid == form),
            None,
        )

        if fname is None:
            try:
                from tracker import fetch_icon, _img_hsv_hist, _img_orb_desc, _to_orb_gray
                icon = fetch_icon(champ_key)
                lib.icon_imgs[champ_key] = np.array(icon.resize((64, 64), _Image.LANCZOS))
                lib.sqrt_hist[idx] = np.sqrt(_img_hsv_hist(icon))
                lib.orb_refs[idx]  = _img_orb_desc(_to_orb_gray(icon))
            except Exception:
                pass
        else:
            img_path = ASSETS_DIR / fname
            try:
                import cv2 as _cv2
                raw = img_path.read_bytes()
                bgr = _cv2.imdecode(np.frombuffer(raw, np.uint8), _cv2.IMREAD_COLOR)
                if bgr is None:
                    self._save_pos()
                    return
                rgb  = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB)
                lib.icon_imgs[champ_key] = _cv2.resize(rgb, (64, 64))
                pil = _Image.fromarray(rgb)
                from tracker import _img_hsv_hist, _img_orb_desc, _to_orb_gray
                lib.sqrt_hist[idx] = np.sqrt(_img_hsv_hist(pil))
                lib.orb_refs[idx]  = _img_orb_desc(_to_orb_gray(pil))
            except Exception:
                self._save_pos()
                return

        # Invalidate death-panel template cache so it rebuilds with the new icon
        if hasattr(lib, "_tmpl_refs"):
            del lib._tmpl_refs  # type: ignore[attr-defined]

        self._save_pos()

    def _save_pos(self) -> None:
        if not self._init_done:
            return
        try:
            data = {
                "geo_x": self.x(), "geo_y": self.y(),
                "region":       list(self.capture.region) if self.capture else None,
                "death_region": list(self._death_region) if self._death_region else None,
                "fps":                self.fps_combo.currentText(),
                "alert_radius":       self.alert_radius,
                "alert_sound":        self.alert_sound,
                "alert_cooldown":     self.cooldown_spin.value(),
                "alert_volume":       self.volume_slider.value(),
                "off_timeout":        self.off_timeout_spin.value(),
                "champ_size":         self.champ_size_spin.value(),
                "ghost_alpha":        self.ghost_alpha_spin.value(),
                "timer_size":         self.timer_size_spin.value(),
                "timer_alpha":        self.timer_alpha_spin.value(),
                "arrow_size":         self.arrow_size_spin.value(),
                "arrow_alpha":        self.arrow_alpha_spin.value(),
                "dot_size":           self.dot_size_spin.value(),
                "ring_thickness":     self.ring_thickness_spin.value(),
                "capture_hidden":     self.capture_hidden_cb.isChecked(),
                "champ_forms":        self._champ_forms,
                "ghost_marker_mode":  (self.ghost_marker_mode_combo.currentData()
                                       or "always"),
                "ghost_hotkey_vk":    int(self._ghost_hotkey_vk),
            }
            _POS_FILE.write_text(json.dumps(data))
        except Exception:
            pass

    @staticmethod
    def _virtual_screen() -> tuple[int, int, int, int]:
        try:
            u = ctypes.windll.user32
            l = u.GetSystemMetrics(76)
            t = u.GetSystemMetrics(77)
            w = u.GetSystemMetrics(78)
            h = u.GetSystemMetrics(79)
            return l, t, l + w, t + h
        except Exception:
            return 0, 0, 3840, 2160

    def _restore_pos(self) -> None:
        try:
            if not _POS_FILE.exists():
                return
            data   = json.loads(_POS_FILE.read_text())
            vl, vt, vr, vb = self._virtual_screen()

            gx, gy = data.get("geo_x"), data.get("geo_y")
            if gx is not None and gy is not None:
                self.move(max(vl, min(int(gx), vr - 100)),
                          max(vt, min(int(gy), vb - 50)))

            r = data.get("alert_radius", 0)
            if isinstance(r, int) and r >= 0:
                self.alert_radius = r
                self._radius_lbl.setText(f"{r} px" if r else "Off")

            snd = data.get("alert_sound", "")
            if snd and Path(snd).exists():
                self.alert_sound = snd
                self._sound_lbl.setText("✓")
                self._sound_lbl.setStyleSheet(f"color:{ACT}; font-size:12px;")
            else:
                self._sound_lbl.setText("need file")
                self._sound_lbl.setStyleSheet(f"color:{DIM}; font-size:12px;")

            for attr, key, default in (
                ("cooldown_spin",      "alert_cooldown", 20.0),
                ("off_timeout_spin",   "off_timeout",     2.5),
            ):
                try:
                    getattr(self, attr).setValue(
                        float(data.get(key, default)))
                except Exception:
                    pass

            for attr, key, default in (
                ("volume_slider",       "alert_volume",   80),
                ("champ_size_spin",     "champ_size",     36),
                ("ghost_alpha_spin",    "ghost_alpha",    50),
                ("timer_size_spin",     "timer_size",     10),
                ("timer_alpha_spin",    "timer_alpha",   50),
                ("arrow_size_spin",     "arrow_size",      3),
                ("arrow_alpha_spin",    "arrow_alpha",    50),
                ("dot_size_spin",       "dot_size",        3),
                ("ring_thickness_spin", "ring_thickness",  2),
            ):
                try:
                    getattr(self, attr).setValue(
                        int(data.get(key, default)))
                except Exception:
                    pass

            fps = str(data.get("fps", "30"))
            if fps in {"1", "5", "15", "30", "Unlimited"}:
                self.fps_combo.setCurrentText(fps)

            self.capture_hidden_cb.setChecked(
                bool(data.get("capture_hidden", True)))
            QTimer.singleShot(60, self._apply_capture_exclusion)

            region = data.get("region")
            if region and len(region) == 4:
                x, y, w, h = region
                self.capture = Capture((x, y, w, h))
                self.region_lbl.setText(f"{w}×{h}  at ({x}, {y})")
                self._sync_run_button_state()

            death_region = data.get("death_region")
            if death_region and len(death_region) == 4:
                dr = tuple(death_region)
                self._death_region  = dr
                dx, dy, dw, dh = dr
                self._death_capture = Capture(dr)
                self.death_region_lbl.setText(
                    f"{dw}×{dh}  at ({dx}, {dy})  [manual]")
            elif region and len(region) == 4:
                self._sync_death_capture()

            cf = data.get("champ_forms", {})
            if isinstance(cf, dict):
                self._champ_forms = {k: v for k, v in cf.items()
                                     if isinstance(k, str) and isinstance(v, str)}
            # backward-compat with old "kayn_form" key
            kf = data.get("kayn_form")
            if kf and "Kayn" not in self._champ_forms and kf in ("base", "slay", "ass"):
                self._champ_forms["Kayn"] = kf

            gm = data.get("ghost_marker_mode", "always")
            if isinstance(gm, str):
                cb = self.ghost_marker_mode_combo
                for i in range(cb.count()):
                    if cb.itemData(i) == gm:
                        cb.setCurrentIndex(i)
                        break
            vk = data.get("ghost_hotkey_vk")
            if isinstance(vk, int) and 0 < vk < 256:
                self._ghost_hotkey_vk = vk
            self._on_ghost_marker_mode_changed()
            self._refresh_ghost_key_button_label()

        except Exception:
            pass

    # ── window events ─────────────────────────────────────────────────────────

    def moveEvent(self, event) -> None:
        super().moveEvent(event)
        self._save_pos()

    def closeEvent(self, event) -> None:
        self._stop_watch.set()
        self.running = False
        if self.capture:
            self.capture.close()
        self._overlay.stop()
        event.accept()

    def run(self) -> None:
        self.show()
        sys.exit(QApplication.instance().exec())
