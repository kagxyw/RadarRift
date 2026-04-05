"""win_lol.py — Windows helpers for detecting the League client window."""

from __future__ import annotations

import ctypes


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
