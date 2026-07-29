"""Persistent full-display PyQt6 overlay for the directional indicator.

The overlay never creates a QApplication. RadarRift will supply its existing
application instance. The development preview at the bottom creates an
application only when no QApplication exists.

Preview:

    python -m directional_indicator.overlay_qt --preview
    python -m directional_indicator.overlay_qt --preview --screen 1 --seconds 12
"""

from __future__ import annotations

import argparse
import ctypes
import sys

import numpy as np

from PyQt6.QtCore import PYQT_VERSION_STR, QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import (
    QColor,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from PyQt6.QtWidgets import QApplication, QWidget

from .config import (
    DIRECTIONAL_INDICATOR_CONFIG,
    DirectionalIndicatorConfig,
)
from .controller import (
    DirectionalIndicatorController,
    IndicatorDisplayState,
)
from .geometry import (
    MinimapPoint,
    ScreenBounds,
)


class DirectionalIndicatorOverlay(QWidget):
    """One persistent, transparent, non-activating, click-through window."""

    def __init__(
        self,
        config: DirectionalIndicatorConfig = DIRECTIONAL_INDICATOR_CONFIG,
    ) -> None:
        if QApplication.instance() is None:
            raise RuntimeError(
                "DirectionalIndicatorOverlay requires an existing QApplication"
            )
        super().__init__(None)
        self._config = config
        self._bounds: ScreenBounds | None = None
        # Per-slot state and cached pixmaps
        self._slots: dict[str, IndicatorDisplayState] = {}
        self._icon_pixmaps: dict[str, QPixmap | None] = {}
        self._icon_refs:    dict[str, np.ndarray | None] = {}
        self._shutting_down = False

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    @property
    def screen_bounds(self) -> ScreenBounds | None:
        return self._bounds

    @property
    def safe_indicator_radius(self) -> float:
        return self._config.indicator_radius

    def update_bounds(self, bounds: ScreenBounds) -> None:
        """Apply desktop bounds while all painting remains overlay-local."""

        if self._shutting_down:
            return
        if not isinstance(bounds, ScreenBounds):
            raise ValueError("bounds must be ScreenBounds")
        self._bounds = bounds
        self.setGeometry(
            round(bounds.x),
            round(bounds.y),
            round(bounds.width),
            round(bounds.height),
        )

    def show_indicator(self, slot_id: str, state: IndicatorDisplayState) -> None:
        if self._shutting_down:
            return
        self._accept_slot(slot_id, state)
        if self._bounds != state.screen_bounds:
            self.update_bounds(state.screen_bounds)
        self.show()
        self.raise_()
        self._apply_native_click_through()
        self.update()

    def update_indicator(self, states: dict[str, IndicatorDisplayState]) -> None:
        if self._shutting_down:
            return
        # Remove slots that are no longer active
        for gone in set(self._slots) - set(states):
            self._slots.pop(gone, None)
            self._icon_pixmaps.pop(gone, None)
            self._icon_refs.pop(gone, None)
        for slot_id, state in states.items():
            self._accept_slot(slot_id, state)
            if self._bounds != state.screen_bounds:
                self.update_bounds(state.screen_bounds)
        if not self.isVisible():
            self.show()
            self.raise_()
            self._apply_native_click_through()
        self.update()

    def _tick(self) -> None:
        """Called by the feature timer to keep glow animating during hold."""
        if self._slots and not self._shutting_down:
            self.update()

    def hide_indicator(self) -> None:
        self._slots.clear()
        self._icon_pixmaps.clear()
        self._icon_refs.clear()
        self.hide()

    def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        self._slots.clear()
        self._icon_pixmaps.clear()
        self._icon_refs.clear()
        self.hide()
        self.deleteLater()

    def set_config(self, config: DirectionalIndicatorConfig) -> None:
        """Apply a live config change (icon size) and rebuild cached pixmaps."""
        self._config = config
        self._icon_pixmaps.clear()
        for slot_id, state in self._slots.items():
            self._icon_refs[slot_id] = state.champion_icon
            self._icon_pixmaps[slot_id] = self._prepare_icon_pixmap(
                state.champion_icon
            )
        if self._slots:
            self.update()

    def _accept_slot(self, slot_id: str, state: IndicatorDisplayState) -> None:
        """Install state and prepare icon pixmap for a slot if it changed."""
        icon = state.champion_icon
        if icon is not self._icon_refs.get(slot_id):
            self._icon_refs[slot_id]    = icon
            self._icon_pixmaps[slot_id] = self._prepare_icon_pixmap(icon)
        self._slots[slot_id] = state

    def _prepare_icon_pixmap(
        self,
        icon: np.ndarray | None,
    ) -> QPixmap | None:
        """Convert RadarRift's RGB crop to detached, pre-scaled Qt storage."""

        if icon is None:
            return None
        if not isinstance(icon, np.ndarray) or icon.size == 0:
            return None
        if icon.dtype != np.uint8 or icon.ndim not in (2, 3):
            return None

        array = np.ascontiguousarray(icon)
        if array.ndim == 2:
            height, width = array.shape
            image_format = QImage.Format.Format_Grayscale8
            bytes_per_line = array.strides[0]
        else:
            height, width, channels = array.shape
            if channels == 3:
                # Capture.grab() returns PIL RGB; App converts that directly to
                # NumPy, so live minimap crops are RGB rather than OpenCV BGR.
                image_format = QImage.Format.Format_RGB888
            elif channels == 4:
                image_format = QImage.Format.Format_RGBA8888
            else:
                return None
            bytes_per_line = array.strides[0]

        # QImage initially references NumPy memory. copy() immediately detaches
        # it before the temporary contiguous array can leave scope.
        detached = QImage(
            array.data,
            width,
            height,
            bytes_per_line,
            image_format,
        ).copy()
        if detached.isNull():
            return None
        return QPixmap.fromImage(detached).scaled(
            self._config.icon_width,
            self._config.icon_height,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._apply_native_click_through()

    def _apply_native_click_through(self) -> None:
        """Add Windows styles that supplement Qt's input transparency."""

        if sys.platform != "win32" or self._shutting_down:
            return
        try:
            hwnd = int(self.winId())
            user32 = ctypes.windll.user32
            get_style = user32.GetWindowLongPtrW
            set_style = user32.SetWindowLongPtrW
            get_style.argtypes = [ctypes.c_void_p, ctypes.c_int]
            get_style.restype = ctypes.c_ssize_t
            set_style.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_ssize_t,
            ]
            set_style.restype = ctypes.c_ssize_t

            GWL_EXSTYLE = -20
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_TOOLWINDOW = 0x00000080
            WS_EX_NOACTIVATE = 0x08000000
            WS_EX_LAYERED = 0x00080000

            style = get_style(hwnd, GWL_EXSTYLE)
            style |= (
                WS_EX_LAYERED
                | WS_EX_TRANSPARENT
                | WS_EX_NOACTIVATE
                | WS_EX_TOOLWINDOW
            )
            set_style(hwnd, GWL_EXSTYLE, style)
        except Exception:
            # Qt attributes still provide the portable best-effort behavior.
            pass

    @staticmethod
    def _edge_hug_path(
        cx: float,
        cy: float,
        half_w: float,
        half_h: float,
        centre_deg: float,
        span_deg: float,
        inset: float,
    ) -> QPainterPath:
        """Trace the screen boundary across an angular span around a direction.

        A circular arc only touches a rectangular display at a few points, so
        this walks each ray out to an *inset* rectangle instead. That keeps the
        stroke a constant distance inside every edge and lets it bend around
        corners — required so a thick pen is not clipped off-screen.
        """

        import math as _math

        iw = max(1.0, half_w - max(0.0, inset))
        ih = max(1.0, half_h - max(0.0, inset))
        path = QPainterPath()
        steps = 64
        start = centre_deg - span_deg / 2.0
        for i in range(steps + 1):
            angle = _math.radians(start + span_deg * i / steps)
            ca, sa = _math.cos(angle), _math.sin(angle)
            tx = iw / abs(ca) if abs(ca) > 1e-9 else _math.inf
            ty = ih / abs(sa) if abs(sa) > 1e-9 else _math.inf
            reach = min(tx, ty)
            px, py = cx + ca * reach, cy + sa * reach
            if i == 0:
                path.moveTo(px, py)
            else:
                path.lineTo(px, py)
        return path

    def paintEvent(self, _event) -> None:
        """Paint all active indicator slots independently."""

        import math as _math
        import time as _time

        painter = QPainter(self)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        painter.fillRect(self.rect(), Qt.GlobalColor.transparent)

        if not self._slots or self.width() < 100 or self.height() < 100:
            painter.end()
            return

        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        pulse = 0.55 + 0.45 * _math.sin(_time.monotonic() * _math.tau * 4)
        scr_cx = self.width()  / 2.0
        scr_cy = self.height() / 2.0
        half_w = self.width()  / 2.0
        half_h = self.height() / 2.0
        # Fixed stroke widths — screen_scale only, never the icon size control.
        outer_pen_w = self._config.edge_outer_pen
        inner_pen_w = self._config.edge_inner_pen
        use_edge = self._config.edge_style == "edge"
        # Circular mode: inscribed circle (Valorant/Apex). Edge mode: hug rect.
        edge_r = min(scr_cx, scr_cy) - outer_pen_w * 0.5

        for slot_id, state in self._slots.items():
            direction = state.geometry.direction
            center    = state.geometry.position
            # Clamp using the drawn icon footprint so oversized icons stay on-screen
            # without moving the edge strokes (which ignore size_scale).
            safe = self._config.indicator_radius
            cx = min(max(center.x, safe), max(safe, self.width()  - safe))
            cy = min(max(center.y, safe), max(safe, self.height() - safe))

            icon_half_w = self._config.icon_width  / 2.0
            icon_half_h = self._config.icon_height / 2.0
            icon_rect   = QRectF(cx - icon_half_w, cy - icon_half_h,
                                 self._config.icon_width, self._config.icon_height)

            # ── alert strokes ─────────────────────────────────────────────────
            dir_deg  = state.geometry.direction_angle_degrees
            arc_span = 90.0

            if use_edge:
                # Trace the rectangular screen boundary (including corners).
                outer_path = self._edge_hug_path(
                    scr_cx, scr_cy, half_w, half_h,
                    dir_deg, arc_span, inset=outer_pen_w * 0.5,
                )
                inner_path = self._edge_hug_path(
                    scr_cx, scr_cy, half_w, half_h,
                    dir_deg, arc_span,
                    inset=outer_pen_w + inner_pen_w * 0.5,
                )
                painter.save()
                painter.setOpacity(state.opacity * pulse * 0.45)
                painter.setPen(QPen(
                    QColor(255, 50, 0, 200), outer_pen_w,
                    Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                ))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(outer_path)
                painter.restore()

                painter.save()
                painter.setOpacity(state.opacity * pulse * 0.65)
                painter.setPen(QPen(
                    QColor(255, 140, 0, 230), inner_pen_w,
                    Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                ))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(inner_path)
                painter.restore()
            else:
                # Qt: 0° = 3 o'clock, positive = CCW, units = 1/16°.
                # atan2 is east=0/south=+90, so negate for Qt.
                arc_start = -dir_deg - 45.0
                outer_rect = QRectF(scr_cx - edge_r, scr_cy - edge_r,
                                    edge_r * 2, edge_r * 2)
                painter.save()
                painter.setOpacity(state.opacity * pulse * 0.45)
                painter.setPen(QPen(
                    QColor(255, 50, 0, 200), outer_pen_w,
                    Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                ))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawArc(
                    outer_rect, round(arc_start * 16), round(arc_span * 16))
                painter.restore()

                inner_r = edge_r - outer_pen_w * 0.9
                inner_rect = QRectF(scr_cx - inner_r, scr_cy - inner_r,
                                    inner_r * 2, inner_r * 2)
                painter.save()
                painter.setOpacity(state.opacity * pulse * 0.65)
                painter.setPen(QPen(
                    QColor(255, 140, 0, 230), inner_pen_w,
                    Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                ))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawArc(
                    inner_rect, round(arc_start * 16), round(arc_span * 16))
                painter.restore()

            painter.setOpacity(state.opacity)

            # ── champion icon ─────────────────────────────────────────────────
            pixmap = self._icon_pixmaps.get(slot_id)
            if pixmap is not None:
                painter.save()
                if self._config.circular_icon:
                    clip = QPainterPath()
                    clip.addEllipse(icon_rect)
                    painter.setClipPath(clip)
                painter.drawPixmap(icon_rect, pixmap, QRectF(pixmap.rect()))
                painter.restore()

            # ── icon border ───────────────────────────────────────────────────
            if self._config.icon_border_width > 0:
                border_g = int(60 + pulse * 80)
                painter.save()
                painter.setPen(QPen(QColor(255, border_g, 0, 245),
                                    self._config.icon_border_width + 1))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                if self._config.circular_icon:
                    painter.drawEllipse(icon_rect)
                else:
                    painter.drawRect(icon_rect)
                painter.restore()

            # ── direction arrow ───────────────────────────────────────────────
            ax = cx + direction.x * self._config.arrow_center_offset
            ay = cy + direction.y * self._config.arrow_center_offset
            half_l = self._config.arrow_length / 2.0
            half_w = self._config.arrow_width  / 2.0
            arrow  = QPolygonF([
                QPointF( half_l,  0.0   ),
                QPointF(-half_l, -half_w),
                QPointF(-half_l,  half_w),
            ])
            painter.save()
            painter.translate(ax, ay)
            painter.rotate(dir_deg)
            painter.setPen(QPen(QColor(10, 10, 10, 220), 4))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolygon(arrow)
            fill_g = int(60 + pulse * 80)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, fill_g, 0, 250))
            painter.drawPolygon(arrow)
            painter.restore()

        painter.end()


def create_directional_indicator_overlay(
    config: DirectionalIndicatorConfig = DIRECTIONAL_INDICATOR_CONFIG,
) -> DirectionalIndicatorOverlay | None:
    """Create the persistent overlay only when the feature is enabled."""

    if not config.enabled:
        return None
    return DirectionalIndicatorOverlay(config)


def _screen_bounds(screen) -> ScreenBounds:
    geometry = screen.geometry()
    return ScreenBounds(
        geometry.x(),
        geometry.y(),
        geometry.width(),
        geometry.height(),
    )


def _run_overlay_preview(screen_index: int, seconds: float) -> int:
    existing = QApplication.instance()
    owns_application = existing is None
    application = existing or QApplication(sys.argv)
    screens = application.screens()
    if not screens:
        raise RuntimeError("Qt reported no displays")
    if screen_index < 0 or screen_index >= len(screens):
        raise ValueError(
            f"screen index {screen_index} is unavailable; "
            f"choose 0 through {len(screens) - 1}"
        )

    target = screens[screen_index]
    bounds = _screen_bounds(target)
    config = DIRECTIONAL_INDICATOR_CONFIG
    if not config.enabled:
        print("Directional indicator is disabled; no overlay was created.")
        return 0
    overlay = create_directional_indicator_overlay(config)
    if overlay is None:
        return 0
    controller = DirectionalIndicatorController(
        overlay,
        indicator_radius=config.placement_radius,
        edge_margin=config.edge_margin,
        position_fraction=config.position_fraction,
        hold_seconds=config.hold_seconds,
        fade_seconds=config.fade_seconds,
        max_opacity=config.max_opacity,
        enabled=config.enabled,
    )

    # Asymmetric upright test icons: the bright band is always the top, while
    # alternating colors make icon replacement obvious.
    icon_a = np.zeros((96, 96, 3), dtype=np.uint8)
    icon_a[:, :] = (40, 90, 190)
    icon_a[:18, :] = (250, 220, 60)
    icon_a[22:82, 12:24] = (255, 255, 255)
    icon_a[38:52, 42:56] = (255, 90, 90)

    icon_b = np.zeros((96, 96, 3), dtype=np.uint8)
    icon_b[:, :] = (145, 55, 165)
    icon_b[:18, :] = (90, 245, 150)
    icon_b[22:82, 12:24] = (255, 255, 255)
    icon_b[38:52, 42:56] = (80, 220, 255)

    player = MinimapPoint(200, 200)
    cases = (
        ("North", MinimapPoint(200, 100)),
        ("Northeast", MinimapPoint(300, 100)),
        ("East", MinimapPoint(300, 200)),
        ("Southeast", MinimapPoint(300, 300)),
        ("South", MinimapPoint(200, 300)),
        ("Southwest", MinimapPoint(100, 300)),
        ("West", MinimapPoint(100, 200)),
        ("Northwest", MinimapPoint(100, 100)),
        ("Mostly east, slightly north", MinimapPoint(400, 170)),
        ("Mostly north, slightly west", MinimapPoint(170, 0)),
    )

    animation_timer = QTimer()
    animation_timer.setInterval(config.update_interval_ms)
    animation_timer.timeout.connect(controller.update)
    animation_timer.start()

    # Replacement occurs during the preceding event's fade, demonstrating that
    # a new direction/icon immediately restores configured maximum opacity and
    # restarts time.
    interval_ms = max(
        config.update_interval_ms,
        round((config.hold_seconds + config.fade_seconds * 0.35) * 1000),
    )
    for index, (label, enemy) in enumerate(cases):
        icon = icon_a if index % 2 == 0 else icon_b

        def show_case(
            case_label=label,
            case_enemy=enemy,
            case_icon=icon,
        ) -> None:
            print(f"Preview direction: {case_label}")
            controller.show_enemy_direction(
                player_position=player,
                enemy_position=case_enemy,
                screen_bounds=bounds,
                champion_icon=case_icon,
            )

        QTimer.singleShot(index * interval_ms, show_case)

    print(f"PyQt runtime: {PYQT_VERSION_STR}")
    print(f"Target screen: {screen_index} ({target.name()})")
    print(
        "Desktop bounds: "
        f"origin=({bounds.x:.0f}, {bounds.y:.0f}), "
        f"size={bounds.width:.0f}x{bounds.height:.0f}"
    )
    print(
        "Ten directions will appear in sequence. The bright band must remain "
        "at the top of each icon while only the arrow rotates."
    )
    print(
        "Click through the placeholder and switch back to the console to verify "
        "that the overlay does not take focus."
    )

    def finish() -> None:
        animation_timer.stop()
        controller.shutdown()
        if owns_application:
            application.quit()

    minimum_preview_seconds = (len(cases) * interval_ms + 1000) / 1000.0
    actual_seconds = max(seconds, minimum_preview_seconds)
    QTimer.singleShot(max(1, round(actual_seconds * 1000)), finish)
    return application.exec() if owns_application else 0


def _parse_preview_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview the persistent directional indicator overlay."
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="enable development preview without changing centralized config",
    )
    parser.add_argument(
        "--screen",
        type=int,
        default=0,
        help="zero-based Qt screen index (default: 0)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=12.0,
        help="seconds before clean automatic shutdown (default: 12)",
    )
    args = parser.parse_args()
    if not 0.1 <= args.seconds <= 300.0:
        parser.error("--seconds must be between 0.1 and 300")
    return args


if __name__ == "__main__":
    _args = _parse_preview_args()
    if _args.preview or DIRECTIONAL_INDICATOR_CONFIG.preview:
        raise SystemExit(_run_overlay_preview(_args.screen, _args.seconds))
    print(
        "Directional indicator preview is disabled. Use --preview or set "
        "preview=True in directional_indicator/config.py."
    )
