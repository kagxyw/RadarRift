"""
champions.py — Champion roster data structures + local cache assets.

STATUS_ON_MAP  = 1   champion is visible on the minimap
STATUS_OFF_MAP = 2   champion is not visible

Flow
----
1.  DataDragon reads champion metadata + images only from cache/ (no network).
2.  ChampionIdentifier compares a crop against cached loading portraits.
3.  ChampionRoster holds 1 player + 4 allies + 5 enemies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


# ── status constants ─────────────────────────────────────────────────────────

STATUS_ON_MAP  = 1
STATUS_OFF_MAP = 2

_CACHE_ROOT = Path(__file__).resolve().parent / "cache"
CACHE_DIR = _CACHE_ROOT


# ── data structures ───────────────────────────────────────────────────────────

@dataclass
class Champion:
    name: str           # display name, e.g. "Miss Fortune"
    key: str            # id key, e.g. "MissFortune" (matches cache filenames)
    is_player: bool = False
    status: int = STATUS_OFF_MAP

    def __repr__(self) -> str:
        tag = "PLAYER" if self.is_player else ("ON " if self.status == STATUS_ON_MAP else "off")
        return f"Champion({self.name!r}, {tag})"


class ChampionRoster:
    """
    Holds the full 10-champion roster for one game.

    enemies  — list of 5 Champion objects (opponents)
    allies   — list of 4 Champion objects (teammates, excl. player)
    player   — the local player's Champion
    """

    def __init__(self,
                 player: Champion,
                 allies: list[Champion],
                 enemies: list[Champion],
                 enemy_side: str = "red"):
        self.player     = player
        self.allies     = allies
        self.enemies    = enemies
        self.enemy_side = enemy_side   # "blue" (bot-left base) or "red" (top-right base)

    # ── status helpers ───────────────────────────────────────────────────

    def set_status(self, key: str, status: int) -> None:
        """Set any tracked champion's status by their champion key."""
        for c in (*self.enemies, *self.allies):
            if c.key == key:
                c.status = status
                return

    def on_map(self) -> list[Champion]:
        return [c for c in (*self.enemies, *self.allies) if c.status == STATUS_ON_MAP]

    def off_map(self) -> list[Champion]:
        return [c for c in (*self.enemies, *self.allies) if c.status == STATUS_OFF_MAP]

    # ── serialisable snapshot ────────────────────────────────────────────

    def enemy_statuses(self) -> dict[str, int]:
        """
        Returns {champion_key: status} for all 5 enemies.
        Example: {'Ahri': 2, 'Zed': 1, 'Jinx': 2, 'Thresh': 2, 'Yasuo': 1}
        """
        return {c.key: c.status for c in self.enemies}

    def ally_statuses(self) -> dict[str, int]:
        """
        Returns {champion_key: status} for the 4 allies (excluding player).
        Example: {'Garen': 1, 'Lux': 2, 'Thresh': 1, 'Jinx': 2}
        """
        return {c.key: c.status for c in self.allies}

    def all_tracked(self) -> dict[str, int]:
        """
        Returns statuses for all 9 tracked champions (4 allies + 5 enemies).
        Useful for a single-pass minimap scan.
        """
        return {**self.ally_statuses(), **self.enemy_statuses()}

    def __repr__(self) -> str:
        lines = [f"  Player : {self.player.name}",
                 f"  Allies : {[c.name for c in self.allies]}",
                 "  Enemies:"]
        for c in self.enemies:
            flag = "ON MAP " if c.status == STATUS_ON_MAP else "off map"
            lines.append(f"    [{flag}]  {c.name}")
        return "ChampionRoster(\n" + "\n".join(lines) + "\n)"


# ── Local cache (champion_registry.json + wiki/legacy filenames) ─────────────

class DataDragon:
    """
    Metadata + image paths from cache/ only. No HTTP — populate via python -m tools.rebuild_cache.
    """

    def __init__(self) -> None:
        _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        self._version:   str | None  = None
        self._champ_map: dict | None = None   # key -> {name, ...}

    def version(self) -> str:
        if self._version is None:
            reg = _CACHE_ROOT / "champion_registry.json"
            if reg.exists():
                try:
                    self._version = str(
                        json.loads(reg.read_text(encoding="utf-8")).get("version", "local"),
                    )
                except Exception:
                    self._version = "local"
            else:
                self._version = "local"
        return self._version

    def champion_map(self) -> dict[str, dict]:
        """Returns {key: {name, ...}} from cache/champion_registry.json."""
        if self._champ_map is None:
            reg = _CACHE_ROOT / "champion_registry.json"
            if reg.exists():
                try:
                    raw = json.loads(reg.read_text(encoding="utf-8"))
                    data = raw.get("data", raw)
                    self._champ_map = data if isinstance(data, dict) else {}
                except Exception:
                    self._champ_map = {}
            else:
                self._champ_map = {}
        return self._champ_map

    def name_for_key(self, key: str) -> str:
        """'MissFortune' → 'Miss Fortune'"""
        return self.champion_map().get(key, {}).get("name", key)

    def loading_portrait(self, key: str) -> Image.Image:
        """
        Loading-screen portrait from cache (wiki rebuild: *OriginalLoading.jpg, etc.).
        """
        for name in (
            f"{key}_OriginalLoading.jpg",
            f"{key}_0.jpg",
            f"{key}_loading.jpg",
        ):
            path = _CACHE_ROOT / name
            if path.exists():
                return Image.open(path).convert("RGB")
        matches = sorted(_CACHE_ROOT.glob(f"{key}_*Loading.jpg"))
        if matches:
            return Image.open(matches[0]).convert("RGB")
        raise FileNotFoundError(
            f"No loading portrait in cache for {key!r} — run python -m tools.rebuild_cache.",
        )

    def square_icon(self, key: str) -> Image.Image:
        """Square minimap icon from cache/icons/{key}.png."""
        path = _CACHE_ROOT / "icons" / f"{key}.png"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing icon {path.name} — run python -m tools.rebuild_cache (download_icons).",
            )
        return Image.open(path).convert("RGB")

    def preload_all_portraits(self,
                              on_progress=None) -> None:
        """Open each known champion portrait if present (no download)."""
        champs = self.champion_map()
        total  = len(champs)
        for i, key in enumerate(champs):
            try:
                self.loading_portrait(key)
            except Exception:
                pass
            if on_progress:
                on_progress(i + 1, total, key)


# ── champion identifier — raw-pixel NCC ──────────────────────────────────────

class ChampionIdentifier:
    """
    Identifies a champion from a PIL Image crop using raw-pixel NCC.

    Same pipeline as match_start.SkinDatabase and tracker._icon_vec:
      1. Crop to the top _DD_TOP_H rows of the reference portrait (matches
         what the YOLO loading-screen crop captures)
      2. Trim thin gold border
      3. Resize to NCC_SIZE
      4. Mean-centre + L2-normalise
      dot-product → NCC score in [-1, 1]  (higher = better match)
    """

    # Keep in sync with match_start constants
    _NCC_SIZE    = (128, 173)
    _BORDER_FRAC = 0.04
    _DD_TOP_H    = 416

    def __init__(self, dragon: DataDragon) -> None:
        self.dragon = dragon
        self._vec_cache: dict[str, np.ndarray] = {}   # key → NCC vector

    # ── NCC helpers ──────────────────────────────────────────────────────

    def _ncc_vec(self, img: Image.Image) -> np.ndarray:
        img = img.convert("RGB")
        w, h = img.size
        # Take top portion matching loading-screen crop content
        if h > self._DD_TOP_H:
            img = img.crop((0, 0, w, self._DD_TOP_H))
        # Trim border
        w2, h2 = img.size
        px = max(1, int(w2 * self._BORDER_FRAC))
        py = max(1, int(h2 * self._BORDER_FRAC))
        img = img.crop((px, py, w2 - px, h2 - py))
        arr = np.array(img.resize(self._NCC_SIZE, Image.LANCZOS), dtype=np.float32)
        arr -= arr.mean()
        return (arr / (np.linalg.norm(arr) + 1e-8)).flatten()

    def _ref_vec(self, key: str) -> np.ndarray:
        if key not in self._vec_cache:
            self._vec_cache[key] = self._ncc_vec(self.dragon.loading_portrait(key))
        return self._vec_cache[key]

    # ── public API ───────────────────────────────────────────────────────

    def precompute(self, on_progress=None) -> None:
        """Pre-compute NCC vectors for every cached portrait."""
        champs = self.dragon.champion_map()
        total  = len(champs)
        for i, key in enumerate(champs):
            try:
                self._ref_vec(key)
            except Exception:
                pass
            if on_progress:
                on_progress(i + 1, total, key)

    def identify(self,
                 crop: Image.Image,
                 top_n: int = 3) -> list[tuple[str, str, float]]:
        """
        Compare crop against every champion portrait using NCC.

        Returns [(key, display_name, score), ...] sorted best-first.
        """
        query     = self._ncc_vec(crop)
        champ_map = self.dragon.champion_map()
        scores: list[tuple[str, str, float]] = []

        for key in champ_map:
            try:
                score = float(np.dot(self._ref_vec(key), query))
                scores.append((key, champ_map[key]["name"], score))
            except Exception:
                pass

        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[:top_n]
