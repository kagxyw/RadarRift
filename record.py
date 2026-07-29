"""
record.py — Record the minimap, synced to RadarRift's tracking state.

Automatically starts a new recording when RadarRift begins tracking and saves
it when RadarRift stops.  Multiple sessions are saved as separate files.

Controls (any time the script is running)
-----------------------------------------
  F11   mark a checkpoint in the current recording
  F12   quit record.py entirely

Output  (recordings/ folder next to this file)
------
  minimap_YYYYMMDD_HHMMSS.avi    video
  minimap_YYYYMMDD_HHMMSS.json   checkpoint list

Usage
-----
  python record.py              # 10 fps, region from .radarrift_pos.json
  python record.py --fps 15     # override frame rate
"""

from __future__ import annotations

import argparse
import ctypes
import json
import signal
import time
from datetime import datetime
from pathlib import Path

import cv2
import mss
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────

_ROOT       = Path(__file__).resolve().parent
_POS_FILE   = _ROOT / ".radarrift_pos.json"
_SENTINEL   = _ROOT / ".rr_active"          # written by app.py on start
_OUT_DIR    = _ROOT / "recordings"

_VK_F11 = 0x7A
_VK_F12 = 0x7B
_POLL   = 0.25    # seconds between sentinel polls while idle

# ── helpers ───────────────────────────────────────────────────────────────────

def _vk_pressed(vk: int) -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


def _load_region() -> tuple[int, int, int, int]:
    if not _POS_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {_POS_FILE}\n"
            "Run the main RadarRift app first to calibrate the minimap region."
        )
    with _POS_FILE.open() as f:
        data = json.load(f)
    region = data.get("region")
    if not region or len(region) != 4:
        raise ValueError(f"'region' key missing or invalid in {_POS_FILE}")
    return tuple(region)   # type: ignore[return-value]


def _record_session(sct: mss.base.MSSBase,
                    region: tuple[int, int, int, int],
                    fps: float) -> None:
    """Record frames until the sentinel disappears or F12 is pressed."""
    x, y, w, h = region
    interval = 1.0 / fps

    stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _OUT_DIR / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    cp_path = out_dir / "checkpoints.json"

    monitor     = {"left": x, "top": y, "width": w, "height": h}
    checkpoints: list[dict] = []
    frame_idx   = 0
    f11_prev    = _vk_pressed(_VK_F11)
    f12_prev    = _vk_pressed(_VK_F12)

    print(f"  ● Recording  →  {out_dir}/")

    t_next = time.perf_counter()
    try:
        while _SENTINEL.exists():
            now = time.perf_counter()

            # ── keyboard ──────────────────────────────────────────────────────
            f11 = _vk_pressed(_VK_F11)
            f12 = _vk_pressed(_VK_F12)

            if f11 and not f11_prev:
                cp = {"frame": frame_idx,
                      "time":  time.time(),
                      "elapsed_s": round(frame_idx / fps, 3)}
                checkpoints.append(cp)
                print(f"  [checkpoint {len(checkpoints)}]"
                      f"  frame {frame_idx}  ({cp['elapsed_s']:.1f} s)")

            if f12 and not f12_prev:
                raise KeyboardInterrupt

            f11_prev = f11
            f12_prev = f12

            # ── capture ───────────────────────────────────────────────────────
            if now >= t_next:
                shot  = sct.grab(monitor)
                frame = np.frombuffer(shot.bgra, dtype=np.uint8)
                frame = frame.reshape((h, w, 4))[:, :, :3]
                fname = out_dir / f"frame_{frame_idx:06d}.png"
                cv2.imwrite(str(fname), frame)
                frame_idx += 1
                t_next += interval

                if frame_idx % max(1, int(fps * 10)) == 0:
                    print(f"  {frame_idx} frames  ({frame_idx / fps:.0f} s)",
                          end="\r")
            else:
                time.sleep(max(0.0, t_next - now - 0.001))

    finally:
        cp_data = {
            "folder":        out_dir.name,
            "region":        list(region),
            "fps":           fps,
            "total_frames":  frame_idx,
            "total_seconds": round(frame_idx / fps, 2),
            "checkpoints":   checkpoints,
        }
        with cp_path.open("w") as f:
            json.dump(cp_data, f, indent=2)
        print(f"\n  ■ Saved {frame_idx} frames  →  {out_dir.name}/"
              f"  ({len(checkpoints)} checkpoints)")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record the minimap, synced to RadarRift tracking.")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Frames per second (default: 1.0; use 0.2 for 1 frame/5 s)")
    parser.add_argument("--region", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                        help="Override minimap region instead of reading from JSON")
    args = parser.parse_args()

    fps      = max(1.0 / 60.0, min(60.0, args.fps))

    if args.region:
        region = tuple(args.region)
    else:
        region = _load_region()

    x, y, w, h = region
    _OUT_DIR.mkdir(exist_ok=True)

    print("RadarRift minimap recorder")
    print(f"  Region  x={x} y={y} w={w} h={h}")
    print(f"  FPS     {fps:.0f}")
    print(f"  Output  {_OUT_DIR}/")
    print()
    print("  Waiting for RadarRift to start tracking…")
    print("  F11 = checkpoint    F12 = quit")
    print()

    f12_prev = _vk_pressed(_VK_F12)

    with mss.mss() as sct:
        try:
            while True:
                f12 = _vk_pressed(_VK_F12)
                if f12 and not f12_prev:
                    break
                f12_prev = f12

                if _SENTINEL.exists():
                    try:
                        _record_session(sct, region, fps)
                    except KeyboardInterrupt:
                        break
                    # sentinel gone → RR stopped → go back to waiting
                    if not _SENTINEL.exists():
                        print("\n  Waiting for RadarRift to start tracking…")

                time.sleep(_POLL)

        except KeyboardInterrupt:
            pass

    print("\nDone.")


if __name__ == "__main__":
    main()
