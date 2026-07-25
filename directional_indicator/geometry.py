"""Typed data models and pure geometry for the directional ping indicator.

RadarRift's minimap detections use image coordinates: the origin is at the
minimap's top-left, X increases to the right, and Y increases downward. Screen
and Qt widget coordinates use the same axis orientation, so direction vectors
must not invert Y.

This module deliberately has no Qt, detector, image, or application dependency.
Run it directly for a development-only console preview:

    python -m directional_indicator.geometry
"""

from __future__ import annotations

from dataclasses import dataclass
import math


_EPSILON = 1e-12


def _finite_number(value: object, name: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True, slots=True)
class MinimapPoint:
    """A point in minimap-local pixels, with positive Y pointing downward."""

    x: float
    y: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _finite_number(self.x, "x"))
        object.__setattr__(self, "y", _finite_number(self.y, "y"))


@dataclass(frozen=True, slots=True)
class ScreenBounds:
    """Overlay bounds in desktop coordinates."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _finite_number(self.x, "x"))
        object.__setattr__(self, "y", _finite_number(self.y, "y"))
        object.__setattr__(self, "width", _finite_number(self.width, "width"))
        object.__setattr__(self, "height", _finite_number(self.height, "height"))
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("overlay width and height must be greater than zero")


@dataclass(frozen=True, slots=True)
class NormalizedDirection:
    """A unit vector in screen-axis coordinates."""

    x: float
    y: float

    def __post_init__(self) -> None:
        x = _finite_number(self.x, "direction.x")
        y = _finite_number(self.y, "direction.y")
        magnitude = math.hypot(x, y)
        if magnitude <= _EPSILON:
            raise ValueError("direction must not be a zero-length vector")
        if not math.isclose(magnitude, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("direction must be normalized")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)


@dataclass(frozen=True, slots=True)
class IndicatorDimensions:
    """Complete icon-and-arrow footprint centered on the indicator position."""

    width: float
    height: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "width", _finite_number(self.width, "width"))
        object.__setattr__(self, "height", _finite_number(self.height, "height"))
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("indicator width and height must be greater than zero")

    @property
    def bounding_radius(self) -> float:
        """Conservative radius enclosing the complete rectangular footprint."""

        return math.hypot(self.width / 2.0, self.height / 2.0)


@dataclass(frozen=True, slots=True)
class IndicatorGeometry:
    """Resolved placement, expressed locally within the overlay."""

    position: MinimapPoint
    direction: NormalizedDirection
    direction_angle_degrees: float


def normalize_relative_direction(
    player_position: MinimapPoint,
    enemy_position: MinimapPoint,
) -> NormalizedDirection:
    """Return the player-to-enemy unit vector in RadarRift minimap coordinates."""

    if not isinstance(player_position, MinimapPoint):
        raise ValueError("player_position must be a MinimapPoint")
    if not isinstance(enemy_position, MinimapPoint):
        raise ValueError("enemy_position must be a MinimapPoint")

    dx = enemy_position.x - player_position.x
    dy = enemy_position.y - player_position.y
    magnitude = math.hypot(dx, dy)
    if not math.isfinite(magnitude) or magnitude <= _EPSILON:
        raise ValueError("player and enemy positions must be distinct")
    return NormalizedDirection(dx / magnitude, dy / magnitude)


def calculate_indicator_geometry(
    player_position: MinimapPoint,
    enemy_position: MinimapPoint,
    overlay_bounds: ScreenBounds,
    indicator_radius: float,
    edge_margin: float,
) -> IndicatorGeometry:
    """Place an indicator where the direction ray meets a safe inset rectangle.

    ``indicator_radius`` must enclose the complete combined icon-and-arrow
    footprint. ``IndicatorDimensions.bounding_radius`` provides a conservative
    value for a rectangular footprint.

    The returned position is local to the overlay even when ``overlay_bounds``
    has a non-zero or negative desktop origin.
    """

    if not isinstance(overlay_bounds, ScreenBounds):
        raise ValueError("overlay_bounds must be ScreenBounds")
    radius = _finite_number(indicator_radius, "indicator_radius")
    margin = _finite_number(edge_margin, "edge_margin")
    if radius < 0.0:
        raise ValueError("indicator_radius must not be negative")
    if margin < 0.0:
        raise ValueError("edge_margin must not be negative")

    direction = normalize_relative_direction(player_position, enemy_position)
    inset = radius + margin
    half_width = overlay_bounds.width / 2.0 - inset
    half_height = overlay_bounds.height / 2.0 - inset
    if half_width < 0.0 or half_height < 0.0:
        raise ValueError(
            "indicator radius and edge margin do not fit inside overlay bounds"
        )

    # Work in desktop coordinates to account explicitly for non-zero origins.
    origin_x = overlay_bounds.x + overlay_bounds.width / 2.0
    origin_y = overlay_bounds.y + overlay_bounds.height / 2.0

    x_scale = (
        half_width / abs(direction.x)
        if abs(direction.x) > _EPSILON
        else math.inf
    )
    y_scale = (
        half_height / abs(direction.y)
        if abs(direction.y) > _EPSILON
        else math.inf
    )
    ray_scale = min(x_scale, y_scale)

    intersection_x = origin_x + direction.x * ray_scale
    intersection_y = origin_y + direction.y * ray_scale

    local_x = intersection_x - overlay_bounds.x
    local_y = intersection_y - overlay_bounds.y
    angle = math.degrees(math.atan2(direction.y, direction.x))

    return IndicatorGeometry(
        position=MinimapPoint(local_x, local_y),
        direction=direction,
        direction_angle_degrees=angle,
    )


def _run_console_preview() -> None:
    from .config import DIRECTIONAL_INDICATOR_CONFIG

    config = DIRECTIONAL_INDICATOR_CONFIG
    overlay = ScreenBounds(x=2560, y=-120, width=2560, height=1440)
    player = MinimapPoint(200, 200)
    dimensions = IndicatorDimensions(
        width=config.icon_width,
        height=config.icon_height,
    )
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

    print(
        "Overlay desktop bounds: "
        f"origin=({overlay.x:.0f}, {overlay.y:.0f}), "
        f"size={overlay.width:.0f}x{overlay.height:.0f}"
    )
    print(
        "Complete indicator footprint: "
        f"{dimensions.width:.0f}x{dimensions.height:.0f}, "
        f"bounding radius={dimensions.bounding_radius:.2f}"
    )
    print("Angles use screen axes: east=0°, south=90°, north=-90°.\n")

    for label, enemy in cases:
        geometry = calculate_indicator_geometry(
            player,
            enemy,
            overlay,
            config.indicator_radius,
            edge_margin=config.edge_margin,
        )
        print(
            f"{label:30} "
            f"direction=({geometry.direction.x:+.4f}, "
            f"{geometry.direction.y:+.4f})  "
            f"local=({geometry.position.x:8.2f}, "
            f"{geometry.position.y:8.2f})  "
            f"angle={geometry.direction_angle_degrees:+8.2f}°"
        )


if __name__ == "__main__":
    _run_console_preview()
