"""Pure minimap champion-icon extraction for the directional indicator.

The extractor consumes an already-captured minimap frame and an already-known
detection box (preferred) or center point (fallback). It performs no capture,
detection, champion lookup, Qt conversion, or mutation of the source frame.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TypeAlias

import numpy as np

from .geometry import MinimapPoint


BoundingBox: TypeAlias = tuple[float, float, float, float]
CropSize: TypeAlias = int | tuple[int, int]


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


def _validate_frame(minimap_frame: np.ndarray) -> tuple[int, int]:
    if not isinstance(minimap_frame, np.ndarray):
        raise ValueError("minimap_frame must be a NumPy array")
    if minimap_frame.ndim not in (2, 3):
        raise ValueError("minimap_frame must have shape (H, W) or (H, W, C)")
    height, width = minimap_frame.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("minimap_frame must not be empty")
    if minimap_frame.ndim == 3 and minimap_frame.shape[2] <= 0:
        raise ValueError("minimap_frame must contain at least one channel")
    return height, width


def _fixed_crop_dimensions(size: CropSize) -> tuple[int, int]:
    if isinstance(size, bool):
        raise ValueError("fixed_crop_size must be a positive integer or pair")
    if isinstance(size, int):
        width = height = size
    elif isinstance(size, tuple) and len(size) == 2:
        width, height = size
        if (
            isinstance(width, bool)
            or isinstance(height, bool)
            or not isinstance(width, int)
            or not isinstance(height, int)
        ):
            raise ValueError("fixed crop width and height must be integers")
    else:
        raise ValueError("fixed_crop_size must be a positive integer or pair")
    if width <= 0 or height <= 0:
        raise ValueError("fixed crop width and height must be greater than zero")
    return width, height


def _box_from_detection(
    bounding_box: BoundingBox,
    padding: float,
) -> tuple[int, int, int, int]:
    if not isinstance(bounding_box, (tuple, list)) or len(bounding_box) != 4:
        raise ValueError("bounding_box must contain (x1, y1, x2, y2)")
    x1 = _finite_number(bounding_box[0], "bounding_box.x1")
    y1 = _finite_number(bounding_box[1], "bounding_box.y1")
    x2 = _finite_number(bounding_box[2], "bounding_box.x2")
    y2 = _finite_number(bounding_box[3], "bounding_box.y2")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bounding_box must have positive width and height")
    return (
        math.floor(x1 - padding),
        math.floor(y1 - padding),
        math.ceil(x2 + padding),
        math.ceil(y2 + padding),
    )


def _box_from_center(
    center_position: MinimapPoint,
    fixed_crop_size: CropSize,
    padding: float,
) -> tuple[int, int, int, int]:
    if not isinstance(center_position, MinimapPoint):
        raise ValueError("center_position must be a MinimapPoint")
    width, height = _fixed_crop_dimensions(fixed_crop_size)
    half_width = width / 2.0
    half_height = height / 2.0
    return (
        math.floor(center_position.x - half_width - padding),
        math.floor(center_position.y - half_height - padding),
        math.ceil(center_position.x + half_width + padding),
        math.ceil(center_position.y + half_height + padding),
    )


def extract_champion_icon(
    minimap_frame: np.ndarray,
    bounding_box: BoundingBox | None,
    padding: float = 0,
    *,
    center_position: MinimapPoint | None = None,
    fixed_crop_size: CropSize = 32,
) -> np.ndarray:
    """Return an owned crop from an existing minimap frame.

    ``bounding_box`` is preferred and uses detector coordinates
    ``(x1, y1, x2, y2)``. When it is ``None``, ``center_position`` and
    ``fixed_crop_size`` define a fallback crop. Coordinates are clamped to the
    frame before slicing. The returned C-contiguous array owns its data.
    """

    frame_height, frame_width = _validate_frame(minimap_frame)
    crop_padding = _finite_number(padding, "padding")
    if crop_padding < 0.0:
        raise ValueError("padding must not be negative")

    if bounding_box is not None:
        x1, y1, x2, y2 = _box_from_detection(bounding_box, crop_padding)
    else:
        if center_position is None:
            raise ValueError(
                "center_position is required when bounding_box is unavailable"
            )
        x1, y1, x2, y2 = _box_from_center(
            center_position,
            fixed_crop_size,
            crop_padding,
        )

    x1 = max(0, min(frame_width, x1))
    y1 = max(0, min(frame_height, y1))
    x2 = max(0, min(frame_width, x2))
    y2 = max(0, min(frame_height, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("crop is empty after clamping to the minimap frame")

    crop_view = minimap_frame[y1:y2, x1:x2]
    if crop_view.size == 0:
        raise ValueError("extracted champion crop is empty")

    # Force immediate independent ownership; never expose a view into a capture
    # or frame buffer that another inference iteration may reuse.
    return np.array(crop_view, copy=True, order="C")


def save_crop_for_inspection(
    crop: np.ndarray,
    output_path: str | Path,
    *,
    source_color_format: str = "RGB",
) -> Path:
    """Development-only helper that writes one extracted crop for inspection.

    Normal execution never calls this function. ``RGB`` is RadarRift's current
    minimap-frame format. ``BGR`` is accepted for isolated backend experiments
    and is converted only in the temporary image being saved.
    """

    _validate_frame(crop)
    output = Path(output_path)
    if not output.parent.exists():
        raise ValueError(f"output directory does not exist: {output.parent}")

    color_format = source_color_format.strip().upper()
    inspection_copy = np.array(crop, copy=True, order="C")
    if inspection_copy.ndim == 3:
        channels = inspection_copy.shape[2]
        if color_format == "BGR":
            if channels != 3:
                raise ValueError("BGR inspection requires exactly three channels")
            inspection_copy = inspection_copy[:, :, ::-1].copy()
            color_format = "RGB"
        if color_format not in ("RGB", "RGBA"):
            raise ValueError("source_color_format must be RGB, RGBA, or BGR")
        expected_channels = 3 if color_format == "RGB" else 4
        if channels != expected_channels:
            raise ValueError(
                f"{color_format} inspection requires {expected_channels} channels"
            )
    elif color_format not in ("L", "GRAY", "GREY"):
        raise ValueError("two-dimensional crops require L/GRAY source format")

    if inspection_copy.dtype != np.uint8:
        raise ValueError("manual inspection currently requires a uint8 crop")

    from PIL import Image

    mode = "L" if inspection_copy.ndim == 2 else color_format
    Image.fromarray(inspection_copy, mode=mode).save(output)
    return output
