#!/usr/bin/env python3
"""Remake ONNX files from .pt weights. Run from project root.

Requires:
  - cache/champion_yolo11n.pt   (minimap champion detector → imgsz 320)
  - cache/splash_detection.pt  (loading-screen splash detector → imgsz 1920)
  - ultralytics (pip install ultralytics)

Usage:
  python remake_onnx.py        # export only if .onnx missing
  python remake_onnx.py --force # overwrite existing .onnx
"""
from pathlib import Path

def main():
    cache = Path(__file__).parent / "cache"
    cache.mkdir(exist_ok=True)

    required = [
        ("champion_yolo11n.pt", "champion_yolo11n.onnx", "minimap (imgsz 320)"),
        ("splash_detection.pt", "splash_detection.onnx", "splash (imgsz 1920)"),
    ]
    missing = [name for name, _, _ in required if not (cache / name).exists()]
    if missing:
        print("Missing .pt files in cache/:")
        for name in missing:
            print(f"  {cache / name}")
        print("\nAdd the .pt weights there, then run:  python remake_onnx.py [--force]")
        return 1

    from onnx_model import export_all
    import sys
    force = "--force" in sys.argv or "-f" in sys.argv
    print("Remaking ONNX from .pt…" + (" (force overwrite)" if force else ""))
    results = export_all(force=force)
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'}  {name}")
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    raise SystemExit(main())
