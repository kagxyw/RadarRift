"""Application-facing lifecycle and thread bridge for the indicator feature."""

from __future__ import annotations

from typing import Any

import numpy as np

from PyQt6.QtCore import QObject, QPoint, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QApplication

from .config import DirectionalIndicatorConfig, get_directional_indicator_config
from .controller import DirectionalIndicatorController
from .event import DirectionalPingEvent
from .geometry import MinimapPoint, ScreenBounds
from .icon import BoundingBox, extract_champion_icon
from .overlay_qt import create_directional_indicator_overlay


class DirectionalIndicatorFeature(QObject):
    """Owns one controller, one overlay, one timer, and one queued signal."""

    _ping_received = pyqtSignal(object)

    @classmethod
    def create(cls, parent: QObject) -> "DirectionalIndicatorFeature | None":
        """Create the feature only when enabled; failures disable it safely."""

        config = get_directional_indicator_config()
        if not config.enabled:
            return None
        try:
            return cls(parent, config)
        except Exception:
            return None

    def __init__(
        self,
        parent: QObject,
        config: DirectionalIndicatorConfig,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._shutting_down = False
        renderer = create_directional_indicator_overlay(config)
        if renderer is None:
            raise RuntimeError("directional indicator renderer is disabled")
        self._controller = DirectionalIndicatorController(
            renderer,
            indicator_radius=config.indicator_radius,
            edge_margin=config.edge_margin,
            hold_seconds=config.hold_seconds,
            fade_seconds=config.fade_seconds,
            max_opacity=config.max_opacity,
            enabled=config.enabled,
        )
        self._timer = QTimer(self)
        self._timer.setInterval(config.update_interval_ms)
        self._timer.timeout.connect(self._controller.update)
        self._ping_received.connect(self._on_ping)
        self._timer.start()
        if config.preview:
            self._show_preview()

    def notify_ping(
        self,
        *,
        player_position: tuple[float, float],
        enemy_position: tuple[float, float],
        minimap_frame: np.ndarray,
        bounding_box: BoundingBox | None,
        minimap_capture_region: tuple[int, int, int, int],
    ) -> None:
        """Copy worker-owned data and queue one non-blocking GUI update."""

        if self._shutting_down:
            return
        try:
            player = MinimapPoint(*player_position)
            enemy = MinimapPoint(*enemy_position)
            icon = None
            try:
                icon = extract_champion_icon(
                    minimap_frame,
                    bounding_box,
                    padding=self._config.crop_padding,
                    center_position=enemy,
                )
            except Exception:
                # Missing icon is valid: the GUI still renders the arrow.
                pass
            event = DirectionalPingEvent(
                player_position=player,
                enemy_position=enemy,
                minimap_capture_region=tuple(minimap_capture_region),
                champion_icon=icon,
            )
            self._ping_received.emit(event)
        except Exception:
            # This optional feature never propagates into the original ping path.
            return

    def hide(self) -> None:
        if not self._shutting_down:
            self._controller.hide()

    def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        self._timer.stop()
        self._ping_received.disconnect()
        self._controller.shutdown()

    def _screen_for_capture_region(
        self,
        region: tuple[int, int, int, int],
    ) -> Any:
        x, y, width, height = region
        center_x = x + width / 2.0
        center_y = y + height / 2.0
        for screen in QApplication.screens():
            geometry = screen.geometry()
            dpr = screen.devicePixelRatio()
            left = geometry.x() * dpr
            top = geometry.y() * dpr
            right = left + geometry.width() * dpr
            bottom = top + geometry.height() * dpr
            if left <= center_x < right and top <= center_y < bottom:
                return screen
        screen = QApplication.screenAt(QPoint(round(center_x), round(center_y)))
        return screen or QApplication.primaryScreen()

    @pyqtSlot(object)
    def _on_ping(self, event: object) -> None:
        """Validate queued data and touch renderer resources on the GUI thread."""

        if self._shutting_down or not isinstance(event, DirectionalPingEvent):
            return
        try:
            screen = self._screen_for_capture_region(
                event.minimap_capture_region
            )
            if screen is None:
                return
            geometry = screen.geometry()
            bounds = ScreenBounds(
                geometry.x(),
                geometry.y(),
                geometry.width(),
                geometry.height(),
            )
            self._controller.show_enemy_direction(
                player_position=event.player_position,
                enemy_position=event.enemy_position,
                screen_bounds=bounds,
                champion_icon=event.champion_icon,
            )
        except Exception:
            return

    def _show_preview(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        try:
            geometry = screen.geometry()
            bounds = ScreenBounds(
                geometry.x(),
                geometry.y(),
                geometry.width(),
                geometry.height(),
            )
            icon = np.zeros(
                (self._config.icon_height, self._config.icon_width, 3),
                dtype=np.uint8,
            )
            icon[:, :] = (48, 92, 190)
            top = max(1, self._config.icon_height // 5)
            icon[:top, :] = (250, 220, 65)
            stripe_x = max(1, self._config.icon_width // 8)
            icon[top : max(top + 1, self._config.icon_height - top), :stripe_x] = (
                255,
                255,
                255,
            )
            self._controller.show_enemy_direction(
                player_position=MinimapPoint(100, 100),
                enemy_position=MinimapPoint(200, 45),
                screen_bounds=bounds,
                champion_icon=icon,
            )
        except Exception:
            self._controller.hide()
