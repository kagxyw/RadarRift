#!/usr/bin/env python3
"""Verify the bundled ONNX models match the PyTorch weights the app trains on.

Checks:
  1. File timestamps  — is the .onnx older than the .pt it should come from?
  2. Graph I/O shapes — input imgsz and output class count
  3. Class-name table — ONNX hardcoded dict vs the .pt model's names
  4. Live parity      — same frame through both backends, compare detections
  5. TP CNN           — whether the exported classifier already softmaxes

Run from repo root:
    python -m tools.verify_onnx_parity
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _stamp(p: Path) -> str:
    if not p.is_file():
        return "MISSING"
    return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def check_freshness() -> None:
    print("=" * 68)
    print("1. FILE FRESHNESS")
    print("=" * 68)
    pt = _ROOT / "runs" / "detect" / "continue_Teleport" / "weights" / "best.pt"
    onnx = _ROOT / "cache" / "champion_yolo11n.onnx"
    print(f"  detector .pt   {_stamp(pt):>18}   {pt.relative_to(_ROOT)}")
    print(f"  detector .onnx {_stamp(onnx):>18}   {onnx.relative_to(_ROOT)}")
    if pt.is_file() and onnx.is_file():
        if onnx.stat().st_mtime < pt.stat().st_mtime:
            age = (pt.stat().st_mtime - onnx.stat().st_mtime) / 86400
            print(f"  >> STALE: onnx is {age:.1f} days OLDER than the .pt")
        else:
            print("  >> onnx is newer than .pt (ok)")

    cpt = _ROOT / "runs" / "classify" / "tp_confirm_cls" / "weights" / "best.pt"
    conx = _ROOT / "runs" / "classify" / "tp_confirm_cls" / "weights" / "best.onnx"
    print(f"  tp-cnn   .pt   {_stamp(cpt):>18}")
    print(f"  tp-cnn   .onnx {_stamp(conx):>18}")


def check_graphs() -> None:
    print()
    print("=" * 68)
    print("2. ONNX GRAPH I/O")
    print("=" * 68)
    import onnxruntime as ort

    for label, rel in (
        ("detector", "cache/champion_yolo11n.onnx"),
        ("tp-cnn", "runs/classify/tp_confirm_cls/weights/best.onnx"),
    ):
        p = _ROOT / rel
        if not p.is_file():
            print(f"  {label}: MISSING {rel}")
            continue
        s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        i = s.get_inputs()[0]
        o = s.get_outputs()[0]
        print(f"  {label}: in {i.name}{i.shape}  out {o.name}{o.shape}")
        if label == "detector" and len(o.shape) == 3:
            nc = int(o.shape[1]) - 4
            print(f"      -> implies {nc} classes")


def check_class_names() -> None:
    print()
    print("=" * 68)
    print("3. CLASS NAME TABLE")
    print("=" * 68)
    from onnx_model import _MINIMAP_CLASSES

    print(f"  onnx_model hardcoded : {_MINIMAP_CLASSES}")
    try:
        import torch
        _o = torch.load

        def _p(*a, **k):
            k["weights_only"] = False
            return _o(*a, **k)

        torch.load = _p
        from ultralytics import YOLO

        pt = _ROOT / "runs" / "detect" / "continue_Teleport" / "weights" / "best.pt"
        names = YOLO(str(pt)).names
        print(f"  actual .pt names     : {names}")
        mism = [k for k in names if _MINIMAP_CLASSES.get(k) != names[k]]
        print("  >> MISMATCH at ids " + str(mism) if mism else "  >> names match")
    except Exception as e:
        print(f"  (could not load .pt names: {e})")

    # tp classifier class order
    try:
        from ultralytics import YOLO

        cpt = _ROOT / "runs" / "classify" / "tp_confirm_cls" / "weights" / "best.pt"
        cnames = YOLO(str(cpt)).names
        from tp_confirm import _TP_CLASS_IDX

        print(f"  tp-cnn names         : {cnames}")
        print(f"  tp_confirm uses idx  : {_TP_CLASS_IDX} "
              f"-> '{cnames.get(_TP_CLASS_IDX)}'")
        if cnames.get(_TP_CLASS_IDX) != "teleport":
            print("  >> WRONG INDEX: does not point at 'teleport'")
        else:
            print("  >> tp index correct")
    except Exception as e:
        print(f"  (tp-cnn names unavailable: {e})")


def check_cnn_softmax() -> None:
    print()
    print("=" * 68)
    print("4. TP-CNN OUTPUT SEMANTICS (softmax already applied?)")
    print("=" * 68)
    import onnxruntime as ort

    p = _ROOT / "runs" / "classify" / "tp_confirm_cls" / "weights" / "best.onnx"
    if not p.is_file():
        print("  MISSING onnx")
        return
    s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    iname = s.get_inputs()[0].name
    rng = np.random.default_rng(0)
    x = rng.random((1, 3, 64, 64), dtype=np.float32)
    out = s.run(None, {iname: x})[0][0]
    print(f"  raw output      : {out}")
    print(f"  sum             : {out.sum():.6f}")
    print(f"  min/max         : {out.min():.6f} / {out.max():.6f}")
    already = abs(out.sum() - 1.0) < 1e-3 and out.min() >= 0
    if already:
        print("  >> ALREADY SOFTMAXED — applying softmax again is a BUG")
    else:
        print("  >> raw logits — manual softmax is correct")


def check_live_parity() -> None:
    print()
    print("=" * 68)
    print("5. LIVE PARITY on a real frame")
    print("=" * 68)
    import cv2

    frames = sorted((_ROOT / "tp_detections").glob("*.png"))
    if not frames:
        print("  no tp_detections/*.png to test with")
        return
    f = frames[-1]
    bgr = cv2.imread(str(f))
    print(f"  frame: {f.name}  {bgr.shape}")

    conf = 0.1

    import onnx_model

    om = onnx_model.load_model()
    o_det = onnx_model.infer(om, bgr, imgsz=320, conf=conf)
    print(f"\n  ONNX  ({len(o_det)} dets @ conf>={conf}):")
    for d in sorted(o_det, key=lambda d: -d["conf"])[:12]:
        print(f"     {d['class_name']:<14} {d['conf']:.3f}  {d['box']}")

    try:
        import yolo_champion

        pm = yolo_champion.load_model()
        p_det = yolo_champion.infer(pm, bgr, imgsz=320, conf=conf)
        print(f"\n  PyTorch ({len(p_det)} dets @ conf>={conf}):")
        for d in sorted(p_det, key=lambda d: -d["conf"])[:12]:
            print(f"     {d['class_name']:<14} {d['conf']:.3f}  {d['box']}")

        def tp(ds):
            return [d for d in ds if d["class_name"] in ("teleport", "recall")]

        print(f"\n  teleport/recall — ONNX {len(tp(o_det))}  PT {len(tp(p_det))}")
    except Exception as e:
        print(f"  (PyTorch path unavailable: {e})")


if __name__ == "__main__":
    check_freshness()
    check_graphs()
    check_class_names()
    check_cnn_softmax()
    check_live_parity()
