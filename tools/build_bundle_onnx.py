#!/usr/bin/env python3
"""Export bundled ONNX files into cache/ from local YOLO .pt weights.

Writes:
  cache/champion_yolo11n.onnx   (imgsz 320)  — minimap champion icons
  cache/splash_detection.onnx   (imgsz 1920) — loading-screen detector

Weight search order:
  Minimap: cache/champion_yolo11n.pt → runs/detect/radarrift_champion/weights/best.pt
           → cache/minimap_yolo11n.pt → yolo11n.pt (repo root)
  Splash:  cache/splash_detection.pt → runs/detect/splash/weights/best.pt

Requires: pip install ultralytics  (see requirements-build.txt)

Usage (repo root):
  python -m tools.build_bundle_onnx
  python -m tools.build_bundle_onnx --force

PyInstaller runs this automatically before copying cache/ into the bundle.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Repo root on sys.path so `import onnx_model` works when run as
#   python tools/build_bundle_onnx.py
# (not only: python -m tools.build_bundle_onnx from repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _root() -> Path:
    return _REPO_ROOT


def _pick_minimap_pt(root: Path, cache: Path) -> Path | None:
    # Prefer whatever yolo_champion.py declares as WEIGHTS (single source of truth)
    try:
        import importlib.util, sys as _sys
        spec = importlib.util.spec_from_file_location(
            "yolo_champion", root / "yolo_champion.py"
        )
        mod = importlib.util.module_from_spec(spec)
        # Don't execute the module fully — just grab the WEIGHTS constant
        src = (root / "yolo_champion.py").read_text(encoding="utf-8")
        for line in src.splitlines():
            line = line.strip()
            if line.startswith("WEIGHTS") and "=" in line and "runs" in line:
                # e.g. WEIGHTS = ROOT / "runs" / "detect" / "continue_Teleport" / "weights" / "best.pt"
                parts = [p.strip().strip('"').strip("'") for p in line.split("/")]
                # Reconstruct relative to root
                rel_parts = []
                capture = False
                for p in parts:
                    if "runs" in p:
                        capture = True
                    if capture:
                        rel_parts.append(p)
                if rel_parts:
                    candidate = root / Path(*rel_parts)
                    if candidate.is_file():
                        print(f"Using WEIGHTS from yolo_champion.py: {candidate}")
                        return candidate
    except Exception as e:
        print(f"Could not read WEIGHTS from yolo_champion.py: {e}")

    for p in (
        cache / "champion_yolo11n.pt",
        cache / "minimap_yolo11n.pt",
        root / "yolo11n.pt",
    ):
        if p.is_file():
            return p
    return None


def _pick_splash_pt(root: Path, cache: Path) -> Path | None:
    for p in (
        cache / "splash_detection.pt",
        root / "runs" / "detect" / "splash" / "weights" / "best.pt",
    ):
        if p.is_file():
            return p
    return None


def main() -> int:
    force = "--force" in sys.argv or "-f" in sys.argv
    root = _root()
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    try:
        from onnx_model import export_to_onnx
    except ImportError as e:
        print(f"Cannot import onnx_model: {e}")
        return 1

    out_min = cache / "champion_yolo11n.onnx"
    out_spl = cache / "splash_detection.onnx"

    ok_min = False
    ok_spl = False

    if out_min.is_file() and not force:
        print(f"Keep existing {out_min.name} (use --force to rebuild)")
        ok_min = True
    else:
        mp = _pick_minimap_pt(root, cache)
        if mp:
            print(f"Minimap export: {mp} -> {out_min.name} (imgsz 320)")
            ok_min = export_to_onnx(mp, out_min, imgsz=320, force=force)
            if not ok_min:
                print("  ✗  minimap ONNX export failed")
        else:
            if out_min.is_file():
                print(f"Using existing {out_min.name} (no .pt found)")
                ok_min = True
            else:
                print(
                    "✗  No minimap weights found. Add one of:\n"
                    "     cache/champion_yolo11n.pt, cache/minimap_yolo11n.pt,\n"
                    "     runs/detect/radarrift_champion/weights/best.pt, yolo11n.pt"
                )

    if out_spl.is_file() and not force:
        print(f"Keep existing {out_spl.name} (use --force to rebuild)")
        ok_spl = True
    else:
        sp = _pick_splash_pt(root, cache)
        if sp:
            print(f"Splash export: {sp} -> {out_spl.name} (imgsz 1920)")
            ok_spl = export_to_onnx(sp, out_spl, imgsz=1920, force=force)
            if not ok_spl:
                print("  ✗  splash ONNX export failed")
        else:
            if out_spl.is_file():
                print(f"Using existing {out_spl.name} (no .pt found)")
                ok_spl = True
            else:
                print(
                    "✗  No splash weights found. Add one of:\n"
                    "     cache/splash_detection.pt,\n"
                    "     runs/detect/splash/weights/best.pt"
                )

    if ok_min and ok_spl:
        print("ONNX bundle ready:", out_min.name, ",", out_spl.name)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
