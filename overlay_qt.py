"""
overlay_qt.py — Transparent click-through Qt overlay for RadarRift.

    The QApplication is created and exec()'d by main.py / App.run().
    This module only manages the overlay widget itself.

    Thread-safety: update_state() is called from the render background thread.
    It only sets Python attributes (GIL-protected). The widget repaints via a
    QTimer(16 ms) on the Qt main thread — no cross-thread Qt calls.
"""

from __future__ import annotations

import ctypes
import time
from typing import Any

import cv2
import numpy as np

from PyQt6.QtCore    import Qt, QTimer
from PyQt6.QtGui     import (
    QPainter,
    QPainterPath,
    QPen,
    QColor,
    QFont,
    QFontMetrics,
    QImage,
    QPixmap,
)
from PyQt6.QtWidgets import QWidget

from win_lol import lol_client_is_foreground


# ── team colours ──────────────────────────────────────────────────────────────

_COLORS: dict[str, QColor] = {
    "player": QColor(255, 220,  50),
    "ally":   QColor( 80, 220, 255),
    "enemy":  QColor(255,  80,  80),
}
_COLOR_DEFAULT = QColor(200, 200, 200)


def _team_color(team: str) -> QColor:
    return _COLORS.get(team, _COLOR_DEFAULT)


# ── overlay widget ─────────────────────────────────────────────────────────────

class _QtOverlayWidget(QWidget):
    """
    Frameless, transparent, always-on-top window over the minimap.

    Drawing data is written from the render thread (GIL-protected).
    A QTimer(16 ms) on the Qt main thread triggers repaints — no cross-thread
    Qt calls are ever made from background threads.
    """

    def __init__(self, region: tuple[int, int, int, int]) -> None:
        super().__init__(None)

        # ── drawing state (written from render thread, read in paintEvent) ───
        self.results: list = []
        self.library       = None
        self.now:  float   = 0.0
        self.params: dict  = {}

        self._px_cache: dict[str, QPixmap] = {}

        # DPI ratio: capture region is in physical pixels; Qt geometry is logical
        from PyQt6.QtWidgets import QApplication
        self._dpr: float = QApplication.primaryScreen().devicePixelRatio()

        # ── window flags ─────────────────────────────────────────────────────
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # Convert physical pixel region → logical pixels for Qt geometry
        dpr = self._dpr
        x, y, w, h = region
        self.setGeometry(round(x / dpr), round(y / dpr),
                         round(w / dpr), round(h / dpr))
        self.show()

        # WA_TransparentForMouseEvents only tells Qt to ignore events; the
        # underlying Win32 HWND still intercepts clicks at the OS level.
        # WS_EX_TRANSPARENT makes Windows pass all hit-tests through to whatever
        # is below, giving true OS-level click-through.
        self._apply_click_through()

        # Repaint timer: 16 ms ≈ 62 Hz, fires on the Qt main thread (safe)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(16)

    def _apply_click_through(self) -> None:
        """Set WS_EX_TRANSPARENT on the Win32 HWND for true OS-level click-through."""
        try:
            _GWL_EXSTYLE      = -20
            _WS_EX_TRANSPARENT = 0x00000020
            _WS_EX_LAYERED     = 0x00080000
            hwnd  = int(self.winId())
            style = ctypes.windll.user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(
                hwnd, _GWL_EXSTYLE,
                style | _WS_EX_TRANSPARENT | _WS_EX_LAYERED,
            )
        except Exception:
            pass

    # ── pixmap cache ──────────────────────────────────────────────────────────

    def _get_pixmap(self, key: str, icon_rgb: np.ndarray, sz: int) -> QPixmap:
        cache_key = f"{key}_{sz}"
        if cache_key not in self._px_cache:
            resized = cv2.resize(icon_rgb, (sz, sz), interpolation=cv2.INTER_AREA)
            h, w, _ = resized.shape
            qimg = QImage(
                resized.tobytes(), w, h,
                resized.strides[0], QImage.Format.Format_RGB888,
            )
            self._px_cache[cache_key] = QPixmap.fromImage(qimg)
        return self._px_cache[cache_key]

    # ── paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:
        results      = self.results
        library      = self.library
        now          = self.now or time.perf_counter()
        params       = self.params

        # Clear whenever LoL is unfocused (paint runs ~62 Hz; render thread ~30 Hz,
        # so we must not rely on stale library/results for a frame after alt-tab).
        if library is None or not lol_client_is_foreground():
            p = QPainter(self)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            p.fillRect(self.rect(), Qt.GlobalColor.transparent)
            p.end()
            return

        dot_size     = params.get("dot_size",        5)
        champ_size   = params.get("champ_size",     36)
        ghost_alpha  = params.get("ghost_alpha",    0.5)
        timer_size   = params.get("timer_size",     10)
        arrow_size   = params.get("arrow_size",      2)
        alert_radius = params.get("alert_radius",    0)
        ring_thick   = params.get("ring_thickness",  2)
        # One gate for ghosts, detection dots, and alert radius (hold/toggle/always/never).
        show_overlay = params.get(
            "show_overlay_markers",
            params.get("show_ghost_markers", True),
        )

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Clear to transparent — game shows through
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        p.fillRect(self.rect(), Qt.GlobalColor.transparent)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        # YOLO/tracker coordinates are in physical pixel space (the captured frame).
        # Qt's paint device uses logical pixels.  Applying 1/dpr scale makes
        # physical-pixel drawing positions map correctly onto the logical widget.
        dpr = self._dpr
        if dpr != 1.0:
            p.scale(1.0 / dpr, 1.0 / dpr)

        # ── ghost markers (off-map enemies) ───────────────────────────────────
        if library is not None and show_overlay:
            for ci, ch in enumerate(library.all):
                if library.team[ci] in ("ally", "player"):
                    continue
                st = library._state.get(ch.key)
                if st is None or st.pos is None:
                    continue
                if not getattr(st, "ghost_active", False) and not getattr(st, "dead", False):
                    continue

                cx, cy = library.enemy_base_px if st.dead else st.pos
                src = library.icon_imgs.get(ch.key)
                if src is None:
                    continue

                sz     = max(8, champ_size)
                pixmap = self._get_pixmap(ch.key, src, sz)
                ix, iy = cx - sz // 2, cy - sz // 2

                col   = _team_color(library.team[ci])
                faded = QColor(max(5, col.red()   * 55 // 100),
                               max(5, col.green() * 55 // 100),
                               max(5, col.blue()  * 55 // 100))

                # Circular clip — draw icon inside a circle, not a full square tile
                clip = QPainterPath()
                clip.addEllipse(float(ix), float(iy), float(sz), float(sz))
                p.save()
                p.setClipPath(clip)
                p.setOpacity(ghost_alpha)
                p.drawPixmap(ix, iy, pixmap)
                p.restore()
                p.setOpacity(1.0)

                p.setPen(QPen(faded, 1))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(ix, iy, sz, sz)

                # timer / DEAD label
                if st.dead:
                    label  = "DEAD"
                    tcolor = QColor(255, 80, 80)
                else:
                    elapsed = now - st.last_seen
                    label   = (f"{int(elapsed)}s" if elapsed < 60
                               else f"{int(elapsed//60)}m{int(elapsed%60):02d}s")
                    tcolor  = col

                font = QFont("Arial", max(7, timer_size), QFont.Weight.Bold)
                p.setFont(font)
                fm = QFontMetrics(font)
                tw = fm.horizontalAdvance(label)
                tx = cx - tw // 2
                ty = iy + sz + fm.height()

                p.setPen(QColor(0, 0, 0, 180))
                p.drawText(tx + 1, ty + 1, label)
                p.setPen(tcolor)
                p.drawText(tx, ty, label)

                if st.dead:
                    continue

                # direction arrow — use cached unit vector (persists after going off-map)
                _dir = getattr(st, "last_dir", None)
                if _dir is None:
                    trail = list(st.pos_trail)
                    if len(trail) >= 2:
                        dx = trail[-1][0] - trail[0][0]
                        dy = trail[-1][1] - trail[0][1]
                        dist = (dx * dx + dy * dy) ** 0.5
                        if dist > 3:
                            _dir = (dx / dist, dy / dist)
                if _dir is not None:
                    nx, ny = _dir
                    a = max(1, int(arrow_size))
                    # Short stem past the icon rim; length grows mildly with arrow slider only.
                    rim       = sz // 2
                    shaft_len = rim + 6 + a * 2
                    head_len  = max(5, 5 + a * 2)
                    head_w    = max(4, 3 + a + a // 2)
                    tip_x  = int(cx + nx * (shaft_len + head_len))
                    tip_y  = int(cy + ny * (shaft_len + head_len))
                    base_x = int(cx + nx * shaft_len)
                    base_y = int(cy + ny * shaft_len)
                    px_side = -ny * head_w
                    py_side =  nx * head_w
                    bright = QColor(
                        min(255, faded.red()   + 80),
                        min(255, faded.green() + 80),
                        min(255, faded.blue()  + 80),
                    )
                    from PyQt6.QtCore import QPoint
                    from PyQt6.QtGui  import QPolygon
                    p.setPen(QPen(bright, max(1, arrow_size)))
                    p.drawLine(cx, cy, base_x, base_y)
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(bright)
                    head = QPolygon([
                        QPoint(tip_x, tip_y),
                        QPoint(int(base_x + px_side), int(base_y + py_side)),
                        QPoint(int(base_x - px_side), int(base_y - py_side)),
                    ])
                    p.drawPolygon(head)
                    p.setBrush(Qt.BrushStyle.NoBrush)

        # ── active detection dots ─────────────────────────────────────────────
        if show_overlay and dot_size > 0:
            p.setPen(Qt.PenStyle.NoPen)
            for r in results:
                cx, cy = r["location"]
                col = _team_color(r.get("team", ""))
                p.setBrush(col)
                p.drawEllipse(cx - dot_size, cy - dot_size,
                              dot_size * 2, dot_size * 2)

        # ── alert radius ring ─────────────────────────────────────────────────
        if show_overlay and alert_radius > 0 and ring_thick > 0 and library is not None:
            player_key    = library.roster.player.key
            pst           = library._state.get(player_key)
            player_on_map = any(r.get("key") == player_key for r in results)
            if pst and pst.pos and player_on_map:
                px, py = pst.pos
                enemy_inside = any(
                    r.get("team") == "enemy"
                    and ((r["location"][0] - px) ** 2 +
                         (r["location"][1] - py) ** 2) ** 0.5 <= alert_radius
                    for r in results
                )
                ring_col  = QColor(255, 60, 60) if enemy_inside else QColor(200, 180, 60)
                pen_width = (ring_thick + 2) if enemy_inside else ring_thick
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(QPen(ring_col, pen_width))
                p.drawEllipse(px - alert_radius, py - alert_radius,
                              alert_radius * 2, alert_radius * 2)

        p.end()


# ── public wrapper ─────────────────────────────────────────────────────────────

class QtOverlay:
    """
    Manages a transparent Qt overlay widget.

    Assumes QApplication already exists (created by main.py before App()).
    update_state() is safe to call from any thread — it only writes Python
    attributes (GIL-protected); the widget repaints via its own 60 Hz timer.
    """

    def __init__(self) -> None:
        self._widget: _QtOverlayWidget | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self, region: tuple[int, int, int, int]) -> None:
        if self._widget is not None:
            self._widget.hide()
            self._widget.deleteLater()
            self._widget = None
        self._widget = _QtOverlayWidget(region)

    def stop(self) -> None:
        if self._widget is not None:
            self._widget._timer.stop()
            self._widget.hide()
            self._widget.deleteLater()
            self._widget = None

    # ── state update (safe from any thread) ───────────────────────────────────

    def update_state(
        self,
        results: list,
        library: Any,
        now: float,
        **params,
    ) -> None:
        w = self._widget
        if w is None:
            return
        # Only plain attribute assignment — GIL makes this thread-safe.
        # The repaint QTimer picks up the new data on the next tick.
        w.results = list(results)
        w.library = library
        w.now     = now
        w.params  = params

    # ── window management ─────────────────────────────────────────────────────

    def reposition(self, region: tuple[int, int, int, int]) -> None:
        w = self._widget
        if w is not None:
            dpr = w._dpr
            x, y, wd, ht = region
            w.setGeometry(round(x / dpr), round(y / dpr),
                          round(wd / dpr), round(ht / dpr))

    def set_capture_hidden(self, hidden: bool) -> None:
        w = self._widget
        if w is None:
            return
        try:
            WDA_NONE               = 0x00
            WDA_EXCLUDEFROMCAPTURE = 0x11
            hwnd = int(w.winId())
            ctypes.windll.user32.SetWindowDisplayAffinity(
                hwnd, WDA_EXCLUDEFROMCAPTURE if hidden else WDA_NONE)
        except Exception:
            pass
