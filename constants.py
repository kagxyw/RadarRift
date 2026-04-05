"""constants.py — Shared theme colours, paths, and small game-layout helpers."""

import sys
from pathlib import Path

# Runtime: repo root. PyInstaller onedir: _MEIPASS (bundled datas land next to extracted libs).
_ROOT = Path(__file__).resolve().parent
if getattr(sys, "frozen", False):
    ASSETS_DIR = Path(sys._MEIPASS)
else:
    ASSETS_DIR = _ROOT / "assets"


def minimap_size(scale: float, screen_width: int) -> int:
    """Approximate LoL minimap edge length (px) from MinimapScale and screen width."""
    return int(screen_width * (0.0703 + 0.0243 * scale) * 1.5)


# ── Catppuccin-Mocha colour palette ───────────────────────────────────────────

BG   = "#1e1e2e"   # base
FG   = "#cdd6f4"   # text
DIM  = "#585b70"   # overlay 0
ALLY = "#89b4fa"   # blue
ENE  = "#f38ba8"   # red
ACT  = "#a6e3a1"   # green

# ── Persistent state files ────────────────────────────────────────────────────

_POS_FILE    = _ROOT / ".radarrift_pos.json"
_ROSTER_FILE = _ROOT / ".radarrift_roster.json"
