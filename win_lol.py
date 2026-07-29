"""win_lol.py — Windows helpers for detecting and launching the League client."""

from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path

from select_minimap import (
    _LEAGUE_CLIENT_EXE,
    default_league_install_path,
    default_riot_client_services_path,
)

_league_launch_attempted = False

# Riot Client often stays in the background; do not treat it as "League is open".
_ACTIVE_LOL_IMAGES = (
    "League of Legends.exe",
    "LeagueClient.exe",
    "LeagueClientUx.exe",
)


def lol_client_is_foreground() -> bool:
    """True if the focused window title looks like the League of Legends client."""
    try:
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        if not hwnd:
            return False
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd) + 1
        buf = ctypes.create_unicode_buffer(length)
        ctypes.windll.user32.GetWindowTextW(hwnd, buf, length)
        return "league of legends" in buf.value.lower()
    except Exception:
        return False


def _process_running(image_name: str) -> bool:
    if sys.platform != "win32":
        return False
    try:
        out = subprocess.check_output(
            ["tasklist", "/FI", f"IMAGENAME eq {image_name}", "/NH"],
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        ).decode(errors="ignore")
        return image_name.lower() in out.lower()
    except Exception:
        return False


def league_client_is_active() -> bool:
    """True if the League client UI or in-game process is running (not just Riot tray)."""
    return any(_process_running(name) for name in _ACTIVE_LOL_IMAGES)


def launch_league_on_startup() -> bool:
    """
    Open League once when RadarRift starts, if the client/game is not already up.
    Uses Riot Client (--launch-product=league_of_legends) when available.
    """
    global _league_launch_attempted
    if _league_launch_attempted or sys.platform != "win32":
        return False
    _league_launch_attempted = True
    if league_client_is_active():
        return False

    riot = default_riot_client_services_path()
    if riot is not None:
        try:
            subprocess.Popen(
                [
                    str(riot),
                    "--launch-product=league_of_legends",
                    "--launch-patchline=live",
                ],
            )
            return True
        except OSError:
            pass

    install = default_league_install_path()
    if install is None:
        return False
    client = install / _LEAGUE_CLIENT_EXE
    if not client.is_file():
        return False
    try:
        subprocess.Popen([str(client)], cwd=str(install))
        return True
    except OSError:
        return False
