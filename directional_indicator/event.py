"""Plain cross-thread payload for an accepted directional ping."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .geometry import MinimapPoint


@dataclass(frozen=True, slots=True)
class DirectionalPingEvent:
    player_position: MinimapPoint
    enemy_position: MinimapPoint
    minimap_capture_region: tuple[int, int, int, int]
    champion_icon: np.ndarray | None

    def __post_init__(self) -> None:
        if not isinstance(self.player_position, MinimapPoint):
            raise ValueError("player_position must be a MinimapPoint")
        if not isinstance(self.enemy_position, MinimapPoint):
            raise ValueError("enemy_position must be a MinimapPoint")
        region = self.minimap_capture_region
        if not isinstance(region, tuple) or len(region) != 4:
            raise ValueError(
                "minimap_capture_region must be (x, y, width, height)"
            )
        normalized_region: list[int] = []
        for value in region:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError("capture-region values must be finite numbers")
            normalized_region.append(round(value))
        if normalized_region[2] <= 0 or normalized_region[3] <= 0:
            raise ValueError("capture-region width and height must be positive")
        object.__setattr__(
            self,
            "minimap_capture_region",
            tuple(normalized_region),
        )

        icon = self.champion_icon
        if icon is None:
            return
        if (
            not isinstance(icon, np.ndarray)
            or icon.ndim not in (2, 3)
            or icon.size == 0
        ):
            raise ValueError("champion_icon must be a non-empty image or None")
        # The extractor already returns owned memory. Mark it read-only so the
        # queued cross-thread payload cannot be mutated before GUI consumption.
        if not icon.flags.owndata:
            icon = np.array(icon, copy=True, order="C")
            object.__setattr__(self, "champion_icon", icon)
        icon.setflags(write=False)
