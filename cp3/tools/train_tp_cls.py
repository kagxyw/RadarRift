"""
Train a binary teleport/not_teleport YOLO-cls classifier.

Run from repo root:
  python tools/train_tp_cls.py
"""
import torch
_orig = torch.load
def _p(*a, **k):
    k["weights_only"] = False
    return _orig(*a, **k)
torch.load = _p

if __name__ == "__main__":
    from ultralytics import YOLO
    from pathlib import Path

    OUT_ROOT = Path(__file__).resolve().parent.parent / "dataset_tp_cls"

    model = YOLO("yolo11n-cls.pt")
    model.train(
        data    = str(OUT_ROOT),
        epochs  = 100,
        imgsz   = 64,
        batch   = 32,
        patience= 25,
        lr0     = 5e-4,
        lrf     = 0.01,
        name    = "tp_confirm_cls",
        exist_ok= True,
    )
    out_weights = OUT_ROOT.parent / "runs" / "classify" / "tp_confirm_cls" / "weights" / "best.pt"
    print(f"Done. Copy best weights to cp3/tp_confirm_cls.pt:\n  {out_weights}")
