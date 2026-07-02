"""
Evaluate radarrift_final4 on the validation set in cp2/vallabels
and plot per-class Precision, Recall, F1, and mAP50.

Run from repo root:
    python cp2/eval.py
"""

from pathlib import Path
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO


def main():
    ROOT      = Path(__file__).resolve().parent.parent
    MODEL_PT  = ROOT / "cp2/best.pt"
    OUT_DIR  = ROOT / "cp2"
    OUT_PNG  = OUT_DIR / "eval.png"

    YAML_PATH = OUT_DIR / "_eval_dataset.yaml"
    CLASSES   = ["ally", "enemy", "teleport", "recall", "champion_icon"]

    yaml_content = {
        "path":  str(ROOT / "cp2/vallabels"),
        "train": "images",
        "val":   "images",
        "nc":    len(CLASSES),
        "names": CLASSES,
    }
    with open(YAML_PATH, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    model   = YOLO(str(MODEL_PT))
    metrics = model.val(data=str(YAML_PATH), imgsz=320, conf=0.25, iou=0.5, verbose=True)

    precision = np.array(metrics.box.p)
    recall    = np.array(metrics.box.r)
    ap50      = np.array(metrics.box.ap50)

    f1 = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0.0,
    )

    names = [metrics.names[i] for i in range(len(precision))]

    print(f"\n{'Class':<16} {'Precision':>10} {'Recall':>10} {'F1':>10} {'mAP50':>10}")
    print("-" * 58)
    for n, p, r, f, a in zip(names, precision, recall, f1, ap50):
        print(f"{n:<16} {p:>10.4f} {r:>10.4f} {f:>10.4f} {a:>10.4f}")
    print("-" * 58)
    print(f"{'mean':<16} {precision.mean():>10.4f} {recall.mean():>10.4f} "
          f"{f1.mean():>10.4f} {ap50.mean():>10.4f}")

    x      = np.arange(len(names))
    width  = 0.2
    colors = ["#4C9BE8", "#57C47A", "#E88B4C", "#C45C9B"]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - 1.5*width, precision, width, label="Precision", color=colors[0])
    ax.bar(x - 0.5*width, recall,    width, label="Recall",    color=colors[1])
    ax.bar(x + 0.5*width, f1,        width, label="F1",        color=colors[2])
    ax.bar(x + 1.5*width, ap50,      width, label="mAP50",     color=colors[3])

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("radarrift_final4 — per-class metrics (val set)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for xi, fi in zip(x, f1):
        ax.text(xi + 0.5*width, fi + 0.01, f"{fi:.2f}", ha="center", va="bottom",
                fontsize=9, color=colors[2])

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nChart saved -> {OUT_PNG}")


if __name__ == "__main__":
    main()
