"""
alert_audio.py — Radius-alert playback (custom ping or champion TTS at altered speed).
"""

from __future__ import annotations

import json
import re
import sys
import threading
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if getattr(sys, "frozen", False):
    _BUNDLE = Path(sys._MEIPASS)
else:
    _BUNDLE = _ROOT
_REG = _BUNDLE / "cache" / "champion_registry.json"
if not _REG.is_file():
    _REG = _ROOT / "cache" / "champion_registry.json"
_TTS_DIRS = (
    _BUNDLE / "tts_out",
    _ROOT / "tts_out",
    _ROOT / "tools" / "tts_out",
)

TTS_PLAYBACK_SPEED = 1.25

_name_for_key_cache: dict[str, str] | None = None
_sound_cache: dict[tuple[str, float], object] = {}
_cache_lock = threading.Lock()
_mixer_lock = threading.Lock()
_pygame_ready = False


def _safe_filename(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().replace(" ", "_")
    return s[:60] or "speech"


def _champion_display_names() -> dict[str, str]:
    global _name_for_key_cache
    if _name_for_key_cache is not None:
        return _name_for_key_cache
    out: dict[str, str] = {}
    if _REG.is_file():
        try:
            raw = json.loads(_REG.read_text(encoding="utf-8"))
            data = raw.get("data", raw)
            if isinstance(data, dict):
                for key, val in data.items():
                    if isinstance(val, dict) and val.get("name"):
                        out[str(key)] = str(val["name"]).strip()
        except Exception:
            pass
    _name_for_key_cache = out
    return out


def resolve_tts_mp3(champion_key: str, display_name: str | None = None) -> Path | None:
    """Find pre-generated edge-tts MP3 for a champion key."""
    stems: list[str] = []
    if display_name:
        stems.append(_safe_filename(display_name))
    reg_name = _champion_display_names().get(champion_key)
    if reg_name:
        stems.append(_safe_filename(reg_name))
    stems.append(_safe_filename(champion_key))

    seen: set[str] = set()
    for stem in stems:
        if not stem or stem in seen:
            continue
        seen.add(stem)
        for base in _TTS_DIRS:
            path = base / f"{stem}.mp3"
            if path.is_file() and path.stat().st_size > 200:
                return path
    return None


def _ensure_mixer() -> None:
    global _pygame_ready
    import pygame

    with _mixer_lock:
        if not _pygame_ready:
            pygame.mixer.init()
            _pygame_ready = True


def _sound_at_speed(path: Path, speed: float):
    import pygame

    key = (str(path.resolve()), speed)
    with _cache_lock:
        cached = _sound_cache.get(key)
    if cached is not None:
        return cached

    base = pygame.mixer.Sound(str(path))
    arr = pygame.sndarray.array(base)
    if speed <= 0 or abs(speed - 1.0) < 0.01:
        fast = arr
    else:
        n = arr.shape[0]
        new_n = max(1, int(n / speed))
        idx = np.linspace(0, n - 1, new_n).astype(np.int32)
        fast = arr[idx]

    out = pygame.sndarray.make_sound(fast.astype(arr.dtype, copy=False))
    with _cache_lock:
        _sound_cache[key] = out
    return out


def play_file(path: str | Path, volume: float, *, speed: float = 1.0) -> bool:
    p = Path(path)
    if not p.is_file():
        return False
    try:
        _ensure_mixer()
        with _mixer_lock:
            snd = _sound_at_speed(p, speed) if speed != 1.0 else None
            if snd is None:
                import pygame
                snd = pygame.mixer.Sound(str(p))
            snd.set_volume(max(0.0, min(1.0, volume)))
            snd.play()
        return True
    except Exception:
        return False


def play_champion_tts(
    champion_key: str,
    volume: float,
    *,
    display_name: str | None = None,
    speed: float = TTS_PLAYBACK_SPEED,
) -> bool:
    path = resolve_tts_mp3(champion_key, display_name)
    if path is None:
        return False
    return play_file(path, volume, speed=speed)
