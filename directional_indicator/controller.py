"""Framework-neutral controller for directional indicator timing and state.

The controller owns no timer and starts no thread. A future GUI integration
should call ``update()`` from one shared GUI timer. Rendering is delegated
through the small ``IndicatorRenderer`` protocol.

Run this module directly for a deterministic logging preview:

    python -m directional_indicator.controller
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from .geometry import (
    IndicatorGeometry,
    MinimapPoint,
    ScreenBounds,
    calculate_indicator_geometry,
)


class IndicatorPhase(str, Enum):
    HOLD = "hold"
    FADE = "fade"


@dataclass(frozen=True, slots=True)
class IndicatorDisplayState:
    """Complete renderer-facing state for one visible indicator."""

    geometry: IndicatorGeometry
    screen_bounds: ScreenBounds
    champion_icon: np.ndarray | None
    opacity: float
    phase: IndicatorPhase

    def __post_init__(self) -> None:
        opacity = float(self.opacity)
        if not math.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
            raise ValueError("opacity must be finite and between zero and one")
        object.__setattr__(self, "opacity", opacity)


@runtime_checkable
class IndicatorRenderer(Protocol):
    """Minimal interface implemented later by a GUI renderer."""

    def show_indicator(self, slot_id: str, state: IndicatorDisplayState) -> None:
        ...

    def update_indicator(self, states: dict[str, IndicatorDisplayState]) -> None:
        ...

    def hide_indicator(self) -> None:
        ...

    def update_bounds(self, bounds: ScreenBounds) -> None:
        ...

    def shutdown(self) -> None:
        ...


class DirectionalIndicatorController:
    """Controls replacement, hold, linear fade, hide, and shutdown behavior."""

    def __init__(
        self,
        renderer: IndicatorRenderer,
        *,
        indicator_radius: float,
        edge_margin: float,
        position_fraction: float = 1.0,
        hold_seconds: float = 0.75,
        fade_seconds: float = 0.65,
        max_opacity: float = 1.0,
        enabled: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._renderer = renderer
        self._indicator_radius = self._nonnegative_finite(
            indicator_radius, "indicator_radius"
        )
        self._edge_margin = self._nonnegative_finite(edge_margin, "edge_margin")
        self._position_fraction = float(position_fraction)
        self._hold_seconds = self._nonnegative_finite(
            hold_seconds, "hold_seconds"
        )
        self._fade_seconds = self._nonnegative_finite(
            fade_seconds, "fade_seconds"
        )
        self._max_opacity = min(
            1.0,
            self._nonnegative_finite(max_opacity, "max_opacity"),
        )
        self._clock = clock
        self._enabled = bool(enabled)
        self._shutdown = False
        # Per-indicator slots: id → (started_at, base_state)
        self._slots: dict[str, tuple[float, IndicatorDisplayState]] = {}
        # Back-compat properties kept for set_size_scale callers
        self._visible = False

    @staticmethod
    def _nonnegative_finite(value: object, name: str) -> float:
        if value is None or isinstance(value, bool):
            raise ValueError(f"{name} must be a non-negative finite number")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must be a non-negative finite number"
            ) from exc
        if not math.isfinite(result) or result < 0.0:
            raise ValueError(f"{name} must be a non-negative finite number")
        return result

    @staticmethod
    def _owned_icon(champion_icon: np.ndarray | None) -> np.ndarray | None:
        if champion_icon is None:
            return None
        if not isinstance(champion_icon, np.ndarray):
            raise ValueError("champion_icon must be a NumPy array or None")
        if champion_icon.ndim not in (2, 3) or champion_icon.size == 0:
            raise ValueError("champion_icon must be a non-empty image array")
        return np.array(champion_icon, copy=True, order="C")

    def _now(self) -> float:
        now = float(self._clock())
        if not math.isfinite(now):
            raise ValueError("monotonic clock returned a non-finite value")
        return now

    def _safe_renderer_call(self, method_name: str, *args: object) -> bool:
        try:
            getattr(self._renderer, method_name)(*args)
            return True
        except Exception:
            return False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def visible(self) -> bool:
        return bool(self._slots)

    def set_enabled(self, enabled: bool) -> None:
        if self._shutdown:
            return
        self._enabled = bool(enabled)
        if not self._enabled:
            self.hide()

    def show_enemy_direction(
        self,
        *,
        player_position: MinimapPoint,
        enemy_position: MinimapPoint,
        screen_bounds: ScreenBounds,
        champion_icon: np.ndarray | None,
        indicator_id: str = "",
    ) -> bool:
        """Show or refresh one indicator slot, identified by indicator_id.

        Each unique id gets its own independent hold/fade lifecycle so multiple
        enemies can be shown simultaneously without cutting each other short.
        An empty id (default) replaces a single shared slot, preserving the
        original single-indicator behaviour.
        """
        if not self._enabled or self._shutdown:
            return False
        try:
            geometry = calculate_indicator_geometry(
                player_position,
                enemy_position,
                screen_bounds,
                self._indicator_radius,
                self._edge_margin,
                position_fraction=self._position_fraction,
            )
            icon = self._owned_icon(champion_icon)
            state = IndicatorDisplayState(
                geometry=geometry,
                screen_bounds=screen_bounds,
                champion_icon=icon,
                opacity=self._max_opacity,
                phase=IndicatorPhase.HOLD,
            )
        except (TypeError, ValueError, OverflowError):
            return False

        slot_id = indicator_id or ""
        self._slots[slot_id] = (self._now(), state)
        self._safe_renderer_call("update_bounds", screen_bounds)
        self._safe_renderer_call("show_indicator", slot_id, state)
        return True

    def update(self) -> None:
        """Advance every active slot; remove expired ones and hide if all gone."""
        if not self._enabled or self._shutdown or not self._slots:
            return

        try:
            now = self._now()
        except ValueError:
            self.hide()
            return

        updated: dict[str, tuple[float, IndicatorDisplayState]] = {}
        for slot_id, (started_at, base_state) in self._slots.items():
            elapsed = max(0.0, now - started_at)
            if elapsed < self._hold_seconds:
                opacity = self._max_opacity
                phase   = IndicatorPhase.HOLD
            elif self._fade_seconds > 0.0:
                fade_elapsed = elapsed - self._hold_seconds
                if fade_elapsed >= self._fade_seconds:
                    continue          # expired — drop this slot
                opacity = self._max_opacity * (1.0 - fade_elapsed / self._fade_seconds)
                phase   = IndicatorPhase.FADE
            else:
                continue              # expired

            new_state = IndicatorDisplayState(
                geometry=base_state.geometry,
                screen_bounds=base_state.screen_bounds,
                champion_icon=base_state.champion_icon,
                opacity=opacity,
                phase=phase,
            )
            updated[slot_id] = (started_at, new_state)

        self._slots = updated
        if not self._slots:
            self._safe_renderer_call("hide_indicator")
        else:
            all_states = {sid: s for sid, (_, s) in self._slots.items()}
            self._safe_renderer_call("update_indicator", all_states)

    def hide(self) -> None:
        """Clear all slots and hide the renderer."""
        had = bool(self._slots)
        self._slots.clear()
        if had:
            self._safe_renderer_call("hide_indicator")

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self.hide()
        self._shutdown = True
        self._safe_renderer_call("shutdown")


class LoggingIndicatorRenderer:
    """Development renderer that prints controller transitions to the console."""

    @staticmethod
    def _icon_label(icon: np.ndarray | None) -> str:
        if icon is None:
            return "arrow-only"
        marker = int(icon.flat[0]) if icon.size else -1
        return f"icon(shape={icon.shape}, marker={marker})"

    @classmethod
    def _state_label(cls, state: IndicatorDisplayState) -> str:
        g = state.geometry
        return (
            f"phase={state.phase.value:<4} opacity={state.opacity:.3f} "
            f"position=({g.position.x:.1f}, {g.position.y:.1f}) "
            f"angle={g.direction_angle_degrees:+.1f}° "
            f"{cls._icon_label(state.champion_icon)}"
        )

    def show_indicator(self, state: IndicatorDisplayState) -> None:
        print(f"SHOW   {self._state_label(state)}")

    def update_indicator(self, state: IndicatorDisplayState) -> None:
        print(f"UPDATE {self._state_label(state)}")

    def hide_indicator(self) -> None:
        print("HIDE")

    def update_bounds(self, bounds: ScreenBounds) -> None:
        print(
            "BOUNDS "
            f"origin=({bounds.x:.0f}, {bounds.y:.0f}) "
            f"size={bounds.width:.0f}x{bounds.height:.0f}"
        )

    def shutdown(self) -> None:
        print("SHUTDOWN")


class _PreviewClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _run_logging_preview() -> None:
    from .config import DIRECTIONAL_INDICATOR_CONFIG

    config = DIRECTIONAL_INDICATOR_CONFIG
    clock = _PreviewClock()
    renderer = LoggingIndicatorRenderer()
    controller = DirectionalIndicatorController(
        renderer,
        indicator_radius=config.placement_radius,
        edge_margin=config.edge_margin,
        position_fraction=config.position_fraction,
        hold_seconds=config.hold_seconds,
        fade_seconds=config.fade_seconds,
        max_opacity=config.max_opacity,
        enabled=config.enabled,
        clock=clock,
    )
    player = MinimapPoint(200, 200)
    bounds = ScreenBounds(1920, -120, 2560, 1440)
    first_icon = np.full((16, 16, 3), 11, dtype=np.uint8)
    replacement_icon = np.full((20, 20, 3), 22, dtype=np.uint8)

    print("\n1. Initial east event; demonstrate hold")
    controller.show_enemy_direction(
        player_position=player,
        enemy_position=MinimapPoint(300, 200),
        screen_bounds=bounds,
        champion_icon=first_icon,
    )
    clock.advance(config.hold_seconds * 0.4)
    controller.update()

    print("\n2. Refresh during hold; replace direction and icon")
    controller.show_enemy_direction(
        player_position=player,
        enemy_position=MinimapPoint(200, 100),
        screen_bounds=bounds,
        champion_icon=replacement_icon,
    )
    clock.advance(config.hold_seconds + config.fade_seconds * 0.25)
    controller.update()

    print("\n3. Refresh during fade; replace with southwest arrow-only event")
    controller.show_enemy_direction(
        player_position=player,
        enemy_position=MinimapPoint(100, 300),
        screen_bounds=bounds,
        champion_icon=None,
    )
    clock.advance(config.hold_seconds + config.fade_seconds * 0.5)
    controller.update()

    print("\n4. Complete fade and hide")
    clock.advance(config.fade_seconds * 0.5)
    controller.update()

    print("\n5. Shutdown")
    controller.shutdown()


if __name__ == "__main__":
    _run_logging_preview()
