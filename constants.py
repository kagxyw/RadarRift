"""constants.py — Shared theme colours, paths, and small game-layout helpers."""

from pathlib import Path


def minimap_size(scale: float, screen_width: int) -> int:
    """Approximate LoL minimap edge length (px) from MinimapScale and screen width."""
    return int(screen_width * (0.0703 + 0.0243 * scale))


# ── Catppuccin-Mocha colour palette ───────────────────────────────────────────

BG   = "#1e1e2e"   # base
FG   = "#cdd6f4"   # text
DIM  = "#585b70"   # overlay 0
ALLY = "#89b4fa"   # blue
ENE  = "#f38ba8"   # red
ACT  = "#a6e3a1"   # green

# ── Persistent state files ────────────────────────────────────────────────────

_POS_FILE    = Path(__file__).parent / ".radarrift_pos.json"
_ROSTER_FILE = Path(__file__).parent / ".radarrift_roster.json"
