#!/usr/bin/env python3
"""Delete superseded model weights, keeping everything the code still loads.

Only .pt/.onnx/.engine files are touched. Training records (results.csv,
args.yaml, plots) are always preserved, so the history of every run stays
readable after its weights are gone.

Nothing under runs/ or cache/ is tracked by git, so deletions are permanent.
Dry-run by default:

    python -m tools.clean_weights           # show what would go
    python -m tools.clean_weights --apply   # actually delete
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# Files the running app, the build, or a documented reproduction step needs.
# Each entry records why, so a future cleanup does not have to re-derive it.
KEEP: dict[str, str] = {
    # --- live inference path ---
    "runs/detect/continue_Teleport/weights/best.pt":
        "LIVE detector (yolo_champion.WEIGHTS)",
    "runs/classify/tp_exp_aug_faint/weights/best.pt":
        "LIVE tp CNN, dev/PyTorch path (tp_confirm._PT_CANDIDATES)",
    "runs/classify/tp_confirm_cls/weights/best.onnx":
        "LIVE tp CNN, ONNX path — bundled into the exe",
    "cache/champion_yolo11n.onnx":
        "bundled minimap detector (onnx_model.MINIMAP_ONNX)",
    "cache/splash_detection.onnx":
        "bundled splash detector (onnx_model.SPLASH_ONNX)",

    # --- build-time fallbacks ---
    "runs/detect/splash/weights/best.pt":
        "splash source (splash_model.py, tools/build_bundle_onnx.py)",
    "runs/detect/radarrift_champion/weights/best.pt":
        "minimap fallback in tools/build_bundle_onnx.py",

    # --- documented reproduction ---
    "runs/detect/radarrift_final4/weights/best.pt":
        "CP2 baseline — cp3/README.md reproduction step requires this path",
    "runs/detect/radarrift_final4/weights/best.onnx":
        "CP2 ONNX baseline (source of cp2/best.onnx)",
    "runs/classify/tp_confirm_cls/weights/best.pt":
        "CP3 baseline CNN — tools/compare_tp_models.py, eval_tp_cls*.py",

    # --- pretrained bases used to start training runs ---
    "yolo11n.pt":     "pretrained base",
    "yolo11n-cls.pt": "pretrained base (tools/train_tp_cls.py)",
    "yolo11s-cls.pt": "pretrained base (tools/experiment_tp_cls.py)",
    "yolo26n.pt":     "pretrained base",
}

# Whole trees left alone: shipped deliverables and the built executable.
SKIP_TREES = ("cp2/", "cp3/", "dist/", ".git/")

# Unreferenced, but a distinct model rather than a duplicate, so deleting it
# loses something real. Held back pending an explicit call.
HOLD = {
    "runs/detect/runs/detect/continue_Teleport/weights/best.pt":
        "stray 200-epoch run (2026-07-05), superseded by the live 51-epoch "
        "continuation (2026-07-08); created by training from the wrong cwd",
    "runs/detect/runs/detect/continue_Teleport/weights/last.pt":
        "same stray run",
}


def classify():
    keep_abs = {(_ROOT / k).resolve(): v for k, v in KEEP.items()}
    hold_abs = {(_ROOT / k).resolve(): v for k, v in HOLD.items()}
    delete: list[tuple[Path, int]] = []
    kept: list[tuple[Path, int, str]] = []
    held: list[tuple[Path, int, str]] = []

    for ext in ("*.pt", "*.onnx", "*.engine"):
        for f in _ROOT.rglob(ext):
            rel = f.relative_to(_ROOT).as_posix()
            if any(rel.startswith(t) for t in SKIP_TREES):
                continue
            r = f.resolve()
            if r in keep_abs:
                kept.append((f, f.stat().st_size, keep_abs[r]))
            elif r in hold_abs:
                held.append((f, f.stat().st_size, hold_abs[r]))
            else:
                delete.append((f, f.stat().st_size))

    delete.sort(key=lambda x: -x[1])
    kept.sort(key=lambda x: x[0].as_posix())
    return kept, held, delete


def main() -> int:
    apply = "--apply" in sys.argv
    kept, held, delete = classify()

    print(f"KEEPING {len(kept)} files "
          f"({sum(s for _, s, _ in kept)/1e6:.0f} MB)")
    for f, s, why in kept:
        print(f"  {s/1e6:>7.1f} MB  {f.relative_to(_ROOT).as_posix():<62} {why}")

    if held:
        print(f"\nHELD BACK {len(held)} files "
              f"({sum(s for _, s, _ in held)/1e6:.0f} MB)")
        for f, s, why in held:
            print(f"  {s/1e6:>7.1f} MB  {f.relative_to(_ROOT).as_posix()}\n"
                  f"{'':14}{why}")

    print(f"\n{'DELETING' if apply else 'WOULD DELETE'} {len(delete)} files "
          f"({sum(s for _, s in delete)/1e6:.0f} MB)")
    for f, s in delete:
        print(f"  {s/1e6:>7.1f} MB  {f.relative_to(_ROOT).as_posix()}")

    if not apply:
        print("\nDry run. Re-run with --apply to delete.")
        return 0

    freed = 0
    for f, s in delete:
        try:
            f.unlink()
            freed += s
        except OSError as e:
            print(f"  !! {f}: {e}")

    # Drop weight dirs and run dirs that are now empty of anything useful.
    for d in sorted(_ROOT.rglob("weights"), key=lambda p: -len(p.parts)):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    print(f"\nFreed {freed/1e6:.0f} MB. Training records (results.csv, "
          f"args.yaml, plots) left intact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
