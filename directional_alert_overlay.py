"""Directional enemy alert drawn over the full game screen.
"""

from __future__ import annotations

import ctypes
import math
import time

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QPointF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import QApplication, QWidget


class _DirectionCanvas(QWidget):
    """Full-screen Qt canvas that positions, paints, and fades one alert."""
    def __init__(self) -> None:
        super().__init__(None)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setGeometry(QApplication.primaryScreen().geometry())
        self._direction = (0.0, -1.0)
        self._icon: QPixmap | None = None
        self._shown_at = 0.0
        self._duration = 2.4
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self._timer.start(16)

    def display(self, icon_rgb: np.ndarray, direction: tuple[float, float]) -> None:
        """Start or replace the alert using an RGB icon and minimap direction."""
        dx, dy = direction
        magnitude = math.hypot(dx, dy)
        if magnitude < 0.001 or icon_rgb.size == 0:
            return
        self._direction = (dx / magnitude, dy / magnitude)
        icon = cv2.resize(icon_rgb, (72, 72), interpolation=cv2.INTER_AREA)
        icon = np.ascontiguousarray(icon)
        image = QImage(
            icon.data, icon.shape[1], icon.shape[0], icon.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()
        self._icon = QPixmap.fromImage(image)
        self._shown_at = time.perf_counter()
        self.setGeometry(QApplication.primaryScreen().geometry())
        self.show()
        self.raise_()
        self._make_click_through()
        self.update()

    def _make_click_through(self) -> None:
        try:
            hwnd = int(self.winId())
            style = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
            ctypes.windll.user32.SetWindowLongW(
                hwnd, -20, style | 0x00000020 | 0x00080000,
            )
        except Exception:
            pass

    def _advance(self) -> None:
        if self.isVisible():
            if time.perf_counter() - self._shown_at >= self._duration:
                self.hide()
            else:
                self.update()

    def paintEvent(self, _event) -> None:
        if self._icon is None:
            return
        age = time.perf_counter() - self._shown_at
        remaining = min(1.0, max(0.0, 1.0 - age / self._duration))
        opacity = remaining ** 0.65
        dx, dy = self._direction

        # Extend the minimap direction from screen center until it reaches an
        # inset screen edge. This keeps the arrow visible at every angle.
        cx, cy = self.width() / 2.0, self.height() / 2.0
        tx = (cx - 88.0) / abs(dx) if abs(dx) > 0.001 else float("inf")
        ty = (cy - 88.0) / abs(dy) if abs(dy) > 0.001 else float("inf")
        ray = min(tx, ty)
        tip = QPointF(cx + dx * ray, cy + dy * ray)
        inward = QPointF(-dx, -dy)
        side = QPointF(-dy, dx)
        base = tip + inward * 42.0
        icon_center = tip + inward * 91.0
        arrow = QPolygonF([
            tip,
            base + side * 23.0,
            base + inward * 15.0,
            base - side * 23.0,
        ])

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setOpacity(opacity)
        painter.setPen(QPen(
            QColor(20, 8, 8, 220), 6, Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin,
        ))
        painter.setBrush(QColor(255, 70, 70, 245))
        painter.drawPolygon(arrow)

        size = 72.0
        x, y = icon_center.x() - size / 2, icon_center.y() - size / 2
        clip = QPainterPath()
        clip.addEllipse(x, y, size, size)
        painter.save()
        painter.setClipPath(clip)
        painter.drawPixmap(int(x), int(y), self._icon)
        painter.restore()
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(255, 70, 70, 245), 4))
        painter.drawEllipse(icon_center, size / 2, size / 2)
        painter.end()

    def stop(self) -> None:
        self._timer.stop()
        self.hide()
        self.deleteLater()


class DirectionalAlertOverlay(QObject):
    """Thread-safe facade between minimap inference and the Qt alert canvas."""

    _display_requested = pyqtSignal(object, object)

    def __init__(self) -> None:
        super().__init__()
        self._canvas = _DirectionCanvas()
        self._display_requested.connect(self._canvas.display)

    def show_enemy(
        self, icon_rgb: np.ndarray, direction: tuple[float, float],
    ) -> None:
        self._display_requested.emit(np.array(icon_rgb, copy=True), direction)

    def stop(self) -> None:
        self._canvas.stop()
