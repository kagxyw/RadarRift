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

_POS_FILE = _ROOT / ".radarrift_pos.json"

# LoL lane roles (for radius-alert filtering + manual roster column order)
LANE_ROLES = ("top", "jungle", "mid", "adc", "support")
LANE_ROLE_LABELS = ("Top", "Jungle", "Mid", "ADC", "Support")
BOT_LANE_ROLES = frozenset({"adc", "support"})

# Manual / shorthand → canonical lane id (for alert role filter)
ROLE_ALIASES: dict[str, str] = {
    "jg": "jungle",
    "jun": "jungle",
    "jgl": "jungle",
    "sup": "support",
    "bot": "adc",
}


def normalize_lane_role(role: str) -> str:
    r = (role or "").strip().lower()
    return ROLE_ALIASES.get(r, r)


# After mute-on-map triggers, enemy must stay off minimap this long before alerts return
ALERT_UNMUTE_OFF_MAP_SEC = 10.0

# Radius alerts: suppress same-role / bot-lane pair only this long after tracking starts.
# ~14 min ≈ typical end of laning; 12 min = stricter, 15–18 min = looser.
ROLE_ALERT_FILTER_SEC = 14 * 60
