"""Central configuration for the directional enemy indicator feature."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os

# Base sizes are designed for a 1080p screen (1920×1080).
#
# Two independent scales apply:
#   screen_scale — automatic, display height / 1080, so the indicator keeps the
#                  same physical size on 1440p/4K. Applies to everything.
#   size_scale   — the user-facing control. Applies to the champion icon ONLY,
#                  so enlarging the icon never moves or thickens the edge arc.
_BASE_SCREEN_HEIGHT = 1080
_BASE_ICON_SIZE     = 48   # px at 1080p
_BASE_ARROW_LENGTH  = 24
_BASE_ARROW_WIDTH   = 18
_BASE_EDGE_MARGIN   = 24
# Orange/red edge stroke — screen_scale only, never size_scale.
_BASE_EDGE_OUTER_PEN = 36
_BASE_EDGE_INNER_PEN = 12
EDGE_STYLES = ("circular", "edge")


def _screen_scale() -> float:
    """Return height of primary screen / 1080, so assets scale on 4K etc."""
    try:
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            screen = app.primaryScreen()
            if screen is not None:
                return screen.geometry().height() / _BASE_SCREEN_HEIGHT
    except Exception:
        pass
    return 1.0


@dataclass(frozen=True, slots=True)
class DirectionalIndicatorConfig:
    enabled: bool = True
    hold_seconds: float = 1.5
    fade_seconds: float = 2.0
    update_interval_ms: int = 16

    # User control, champion icon only. 1.0 = default, 2.0 = double.
    size_scale: float = 1.0

    # Automatic display scaling, applied to every dimension. Set from the
    # primary screen height by get_directional_indicator_config().
    screen_scale: float = 1.0

    crop_padding: int = 2
    circular_icon: bool = True
    icon_border_width: int = 2
    icon_arrow_spacing: int = 6
    safe_area_padding: int = 0
    max_opacity: float = 1.0

    # 0.0 = screen centre, 1.0 = screen edge minus edge_margin.
    # 0.55 sits just inside the outer third — visible peripherally without
    # blocking the centre of the screen.
    position_fraction: float = 0.55

    # Orange/red alert stroke style:
    #   "circular" — inscribed circle (Valorant/Apex style)
    #   "edge"     — hugs the rectangular screen boundary
    edge_style: str = "circular"

    debug: bool = False
    preview: bool = False

    def __post_init__(self) -> None:
        for name in ("hold_seconds", "fade_seconds", "size_scale",
                     "screen_scale", "position_fraction"):
            value = self._finite_float(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must not be negative")
            object.__setattr__(self, name, value)
        if self.size_scale <= 0.0:
            raise ValueError("size_scale must be greater than zero")
        if self.screen_scale <= 0.0:
            raise ValueError("screen_scale must be greater than zero")
        if not 0.0 <= self.position_fraction <= 1.0:
            raise ValueError("position_fraction must be between 0.0 and 1.0")
        style = str(self.edge_style).strip().lower()
        if style not in EDGE_STYLES:
            raise ValueError(
                f"edge_style must be one of {EDGE_STYLES}, got {self.edge_style!r}"
            )
        object.__setattr__(self, "edge_style", style)

        for name in (
            "update_interval_ms",
        ):
            value = self._integer(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
            object.__setattr__(self, name, value)

        for name in (
            "crop_padding",
            "icon_border_width",
            "icon_arrow_spacing",
            "safe_area_padding",
        ):
            value = self._integer(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} must not be negative")
            object.__setattr__(self, name, value)

        opacity = self._finite_float(self.max_opacity, "max_opacity")
        object.__setattr__(self, "max_opacity", min(1.0, max(0.0, opacity)))

        for name in ("enabled", "circular_icon", "debug", "preview"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")

    @staticmethod
    def _finite_float(value: object, name: str) -> float:
        if value is None or isinstance(value, bool):
            raise ValueError(f"{name} must be a finite number")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite number") from exc
        if not math.isfinite(result):
            raise ValueError(f"{name} must be a finite number")
        return result

    @staticmethod
    def _integer(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        return value

    # ── scaled dimensions ─────────────────────────────────────────────────────
    # Only the icon responds to size_scale; everything else tracks the display
    # so the user's size control cannot disturb the edge arc or the arrow.

    @property
    def icon_width(self) -> int:
        return max(8, round(_BASE_ICON_SIZE * self.screen_scale * self.size_scale))

    @property
    def icon_height(self) -> int:
        return max(8, round(_BASE_ICON_SIZE * self.screen_scale * self.size_scale))

    @property
    def arrow_length(self) -> int:
        return max(4, round(_BASE_ARROW_LENGTH * self.screen_scale))

    @property
    def arrow_width(self) -> int:
        return max(3, round(_BASE_ARROW_WIDTH * self.screen_scale))

    @property
    def edge_margin(self) -> int:
        return max(4, round(_BASE_EDGE_MARGIN * self.screen_scale))

    @property
    def edge_outer_pen(self) -> int:
        return max(8, round(_BASE_EDGE_OUTER_PEN * self.screen_scale))

    @property
    def edge_inner_pen(self) -> int:
        return max(3, round(_BASE_EDGE_INNER_PEN * self.screen_scale))

    # ── derived geometry ───────────────────────────────────────────────────────

    @property
    def _base_icon_size(self) -> int:
        """Icon size at size_scale=1.0 — used for placement so the control
        only grows/shrinks the drawn icon, never moves it."""
        return max(8, round(_BASE_ICON_SIZE * self.screen_scale))

    @property
    def arrow_center_offset(self) -> float:
        return (
            max(self.icon_width, self.icon_height) / 2.0
            + self.icon_arrow_spacing
            + self.arrow_length / 2.0
        )

    @property
    def placement_radius(self) -> float:
        """Inset used to place the icon. Ignores size_scale so enlarging the
        champion icon never pulls the indicator (or edge arc) inward."""
        base = self._base_icon_size
        icon_radius = math.hypot(base / 2.0, base / 2.0) + self.icon_border_width
        arrow_offset = (
            base / 2.0 + self.icon_arrow_spacing + self.arrow_length / 2.0
        )
        arrow_radius = (
            arrow_offset
            + math.hypot(self.arrow_length / 2.0, self.arrow_width / 2.0)
            + self.icon_border_width
        )
        return max(icon_radius, arrow_radius) + self.safe_area_padding

    @property
    def indicator_radius(self) -> float:
        """Conservative on-screen clamp for the currently drawn icon size."""
        icon_radius = (
            math.hypot(self.icon_width / 2.0, self.icon_height / 2.0)
            + self.icon_border_width
        )
        arrow_radius = (
            self.arrow_center_offset
            + math.hypot(self.arrow_length / 2.0, self.arrow_width / 2.0)
            + self.icon_border_width
        )
        return max(icon_radius, arrow_radius) + self.safe_area_padding

    def with_updates(self, **changes) -> "DirectionalIndicatorConfig":
        """Return a copy with the given fields replaced."""
        values = {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }
        values.update(changes)
        return DirectionalIndicatorConfig(**values)

    def scaled(self, size_scale: float) -> "DirectionalIndicatorConfig":
        """Return a copy with a different icon size_scale, same screen_scale."""
        return self.with_updates(size_scale=float(size_scale))


DIRECTIONAL_INDICATOR_CONFIG = DirectionalIndicatorConfig()


def get_directional_indicator_config(
    size_scale: float | None = None,
    edge_style: str | None = None,
) -> DirectionalIndicatorConfig:
    """Return config with display scaling applied and the user's icon scale.

    Call after QApplication is created so _screen_scale() can query the display.
    """
    preview_raw = os.environ.get("RADARRIFT_DIRECTIONAL_PREVIEW", "").strip().lower()
    preview = DIRECTIONAL_INDICATOR_CONFIG.preview or preview_raw in {"1", "true", "yes", "on"}

    values = {
        field: getattr(DIRECTIONAL_INDICATOR_CONFIG, field)
        for field in DIRECTIONAL_INDICATOR_CONFIG.__dataclass_fields__
    }
    values["preview"] = preview
    values["screen_scale"] = _screen_scale()
    values["size_scale"] = 1.0 if size_scale is None else float(size_scale)
    if edge_style is not None:
        values["edge_style"] = edge_style
    return DirectionalIndicatorConfig(**values)
