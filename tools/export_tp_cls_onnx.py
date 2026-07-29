#!/usr/bin/env python3
"""Export the active TP-confirmation classifier to ONNX for the executable.

The exe cannot use torch/ultralytics (both excluded from the PyInstaller
bundle), so the classifier must ship as ONNX.

Run from repo root:
    python -m tools.export_tp_cls_onnx
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

_orig = torch.load


def _patched(*a, **k):
    k["weights_only"] = False
    return _orig(*a, **k)


torch.load = _patched

# Source weights and the input size they were trained at — these MUST agree
# with CROP_SZ in tp_confirm.py.
SRC = _ROOT / "runs/classify/tp_exp_aug_faint/weights/best.pt"
IMGSZ = 128
DEST = _ROOT / "runs/classify/tp_confirm_cls/weights/best.onnx"


def main() -> int:
    from ultralytics import YOLO

    if not SRC.is_file():
        print(f"missing weights: {SRC}")
        return 1

    m = YOLO(str(SRC))
    print(f"class names: {m.names}")
    tp_idx = next((k for k, v in m.names.items() if v == "teleport"), None)
    print(f"teleport index: {tp_idx}")
    if tp_idx is None:
        print("no 'teleport' class — aborting")
        return 1

    out = Path(m.export(format="onnx", imgsz=IMGSZ, simplify=True, opset=17))
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if out.resolve() != DEST.resolve():
        shutil.copy2(out, DEST)
    print(f"exported -> {DEST}")

    # Verify the graph matches what tp_confirm.py expects.
    import numpy as np
    import onnxruntime as ort

    s = ort.InferenceSession(str(DEST), providers=["CPUExecutionProvider"])
    i, o = s.get_inputs()[0], s.get_outputs()[0]
    print(f"graph: in {i.shape}  out {o.shape}")
    probe = s.run(None, {i.name: np.zeros((1, 3, IMGSZ, IMGSZ),
                                          dtype=np.float32)})[0][0]
    print(f"probe output {probe}  sum={probe.sum():.6f} "
          f"(softmax baked in: {abs(probe.sum()-1.0) < 1e-3})")

    from tp_confirm import CROP_SZ, _TP_CLASS_IDX

    ok = True
    if CROP_SZ != IMGSZ:
        print(f"MISMATCH: tp_confirm.CROP_SZ={CROP_SZ} but exported at {IMGSZ}")
        ok = False
    if _TP_CLASS_IDX != tp_idx:
        print(f"MISMATCH: tp_confirm._TP_CLASS_IDX={_TP_CLASS_IDX} "
              f"but teleport is {tp_idx}")
        ok = False
    print("consistent with tp_confirm.py" if ok else "FIX tp_confirm.py")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
