"""
select_minimap.py — Read LoL MinimapScale and FlipMiniMap from PersistedSettings.json.

FlipMiniMap: 0 = minimap on the right, non-zero = flipped to the left.
Used by app.py (auto region on game start and PersistedSettings watcher).
"""

from __future__ import annotations

import json
import os
import string
import sys
from pathlib import Path

from constants import minimap_size

_RIOT_PERSISTED_TAIL = Path("Riot Games") / "League of Legends" / "Config" / "PersistedSettings.json"


def _find_persisted_settings_windows() -> Path | None:
    hits: list[Path] = []
    for letter in string.ascii_uppercase:
        root = Path(f"{letter}:/")
        try:
            if not root.exists():
                continue
        except OSError:
            continue
        candidate = root / _RIOT_PERSISTED_TAIL
        if candidate.is_file():
            hits.append(candidate)
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0]
    sd = os.environ.get("SystemDrive", "C:").strip().rstrip("\\/")
    if not sd.endswith(":"):
        sd += ":"
    pref = sd.upper()
    for p in hits:
        if p.drive.upper() == pref:
            return p
    return sorted(hits, key=lambda x: x.as_posix())[0]


def default_persisted_settings_path() -> Path | None:
    if sys.platform == "darwin":
        return (
            Path.home()
            / "Library/Application Support/Riot Games/League of Legends/Config/PersistedSettings.json"
        )
    if sys.platform == "win32":
        return _find_persisted_settings_windows()
    return None


def _find_minimap_scale(obj: object) -> float | None:
    if isinstance(obj, dict):
        if obj.get("name") == "MinimapScale" and "value" in obj:
            try:
                return float(str(obj["value"]).strip())
            except ValueError:
                return None
        for v in obj.values():
            found = _find_minimap_scale(v)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_minimap_scale(item)
            if found is not None:
                return found
    return None


def _find_flip_minimap(obj: object) -> object | None:
    if isinstance(obj, dict):
        if obj.get("name") == "FlipMiniMap" and "value" in obj:
            return obj["value"]
        for v in obj.values():
            found = _find_flip_minimap(v)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_flip_minimap(item)
            if found is not None:
                return found
    return None


def _flip_value_to_corner(raw: object) -> str:
    """
    Persisted FlipMiniMap: 0 = minimap on the right, non-zero = flipped (left).
    """
    if raw is None:
        return "right"
    try:
        v = int(float(str(raw).strip()))
    except ValueError:
        return "right"
    return "left" if v != 0 else "right"


def read_minimap_persisted(path: Path | None = None) -> tuple[float | None, str]:
    """
    Load PersistedSettings.json once.
    Returns (MinimapScale or None, minimap corner 'left'|'right' from FlipMiniMap).
    Missing FlipMiniMap defaults to 'right'.
    """
    p = path if path is not None else default_persisted_settings_path()
    if p is None or not p.is_file():
        return None, "right"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, "right"
    scale = _find_minimap_scale(data)
    corner = _flip_value_to_corner(_find_flip_minimap(data))
    return scale, corner


def read_minimap_scale(path: Path | None = None) -> float | None:
    """Load PersistedSettings.json and return MinimapScale, or None if missing/unreadable."""
    scale, _ = read_minimap_persisted(path)
    return scale


def auto_minimap_region(
    screen_w: int,
    screen_h: int,
    scale: float | None = None,
    *,
    margin: int = 8,
    corner: str | None = None,
) -> tuple[int, int, int, int]:
    """
    (left, top, width, height) for a square minimap capture, bottom-left or bottom-right.
    If scale or corner is None, reads from PersistedSettings (MinimapScale, FlipMiniMap).
    FlipMiniMap 0 = right, non-zero = left. Falls back to scale 1.0 if unreadable.
    """
    if scale is None or corner is None:
        fs, fc = read_minimap_persisted()
        if scale is None:
            scale = fs
        if corner is None:
            corner = fc
    s = scale
    if s is None:
        s = 1.0
    sz = minimap_size(s, screen_w)
    sz = max(80, min(sz, min(screen_w, screen_h) - margin * 2))
    if corner == "left":
        x = margin
    else:
        x = screen_w - sz - margin
    y = screen_h - sz - margin
    return (x, y, sz, sz)
