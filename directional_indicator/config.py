"""Central configuration for the directional enemy indicator feature."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os


@dataclass(frozen=True, slots=True)
class DirectionalIndicatorConfig:
    enabled: bool = True
    hold_seconds: float = 1.5
    fade_seconds: float = 2.0
    update_interval_ms: int = 16

    icon_width: int = 48
    icon_height: int = 48
    crop_padding: int = 2
    circular_icon: bool = True
    icon_border_width: int = 2

    arrow_length: int = 24
    arrow_width: int = 18
    icon_arrow_spacing: int = 6

    edge_margin: int = 24
    safe_area_padding: int = 0
    max_opacity: float = 1.0

    debug: bool = False
    preview: bool = False

    def __post_init__(self) -> None:
        for name in ("hold_seconds", "fade_seconds"):
            value = self._finite_float(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must not be negative")
            object.__setattr__(self, name, value)

        for name in (
            "update_interval_ms",
            "icon_width",
            "icon_height",
            "arrow_length",
            "arrow_width",
        ):
            value = self._integer(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
            object.__setattr__(self, name, value)

        for name in (
            "crop_padding",
            "icon_border_width",
            "icon_arrow_spacing",
            "edge_margin",
            "safe_area_padding",
        ):
            value = self._integer(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} must not be negative")
            object.__setattr__(self, name, value)

        opacity = self._finite_float(self.max_opacity, "max_opacity")
        object.__setattr__(self, "max_opacity", min(1.0, max(0.0, opacity)))

        for name in (
            "enabled",
            "circular_icon",
            "debug",
            "preview",
        ):
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

    @property
    def arrow_center_offset(self) -> float:
        """Distance from icon center to arrowhead center."""

        return (
            max(self.icon_width, self.icon_height) / 2.0
            + self.icon_arrow_spacing
            + self.arrow_length / 2.0
        )

    @property
    def indicator_radius(self) -> float:
        """Conservative radius enclosing the rotated icon-and-arrow assembly."""

        icon_radius = math.hypot(
            self.icon_width / 2.0,
            self.icon_height / 2.0,
        ) + self.icon_border_width
        arrow_radius = (
            self.arrow_center_offset
            + math.hypot(self.arrow_length / 2.0, self.arrow_width / 2.0)
            + self.icon_border_width
        )
        return max(icon_radius, arrow_radius) + self.safe_area_padding


DIRECTIONAL_INDICATOR_CONFIG = DirectionalIndicatorConfig()


def get_directional_indicator_config() -> DirectionalIndicatorConfig:
    """Return defaults with optional development-only environment overrides."""

    preview_raw = os.environ.get("RADARRIFT_DIRECTIONAL_PREVIEW", "").strip().lower()
    preview = DIRECTIONAL_INDICATOR_CONFIG.preview or preview_raw in {
        "1",
        "true",
        "yes",
        "on",
    }
    if preview == DIRECTIONAL_INDICATOR_CONFIG.preview:
        return DIRECTIONAL_INDICATOR_CONFIG
    values = {
        field: getattr(DIRECTIONAL_INDICATOR_CONFIG, field)
        for field in DIRECTIONAL_INDICATOR_CONFIG.__dataclass_fields__
    }
    values["preview"] = preview
    return DirectionalIndicatorConfig(**values)
