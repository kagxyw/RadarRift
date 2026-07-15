"""
Compare CP2 (radarrift_final4) vs CP3 (continue_Teleport) YOLO per-class metrics
using the session_teleport validation split, then save eval_compare.png.

Run from repo root:
  python cp3/tools/eval_yolo_compare.py
"""
import torch, yaml
_orig = torch.load
def _p(*a, **k):
    k["weights_only"] = False
    return _orig(*a, **k)
torch.load = _p

from pathlib import Path
import numpy as np

_CP3    = Path(__file__).resolve().parent.parent
_ROOT   = _CP3.parent   # repo root (for CP2 model which lives outside cp3)

CLASSES = ["ally", "enemy", "teleport", "recall", "champion_icon"]
VAL_DIR = _CP3 / "session_teleport"

YAML_PATH = _CP3 / "_tmp_eval.yaml"
yaml.dump({
    "path":  str(VAL_DIR),
    "train": "images",
    "val":   "images",
    "nc":    len(CLASSES),
    "names": CLASSES,
}, open(YAML_PATH, "w"))

MODELS = {
    "CP2 radarrift_final4":  str(_ROOT / "runs/detect/radarrift_final4/weights/best.pt"),
    "CP3 continue_Teleport": str(_CP3 / "best.pt"),
}


def _plot(collected: dict[str, dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(collected.keys())
    metrics = ["P", "R", "F1", "mAP50"]
    colors = [
        ["#6baed6", "#2171b5"],
        ["#74c476", "#238b45"],
        ["#fd8d3c", "#d94801"],
        ["#9e9ac8", "#54278f"],
    ]

    x     = np.arange(len(CLASSES))
    width = 0.18
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    fig.suptitle(
        "CP2 (radarrift_final4) vs CP3 (continue_Teleport)\n"
        "Per-class YOLO metrics on session_teleport dataset",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for ax, metric, (c0, c1) in zip(axes, metrics, colors):
        for i, (label, data) in enumerate(collected.items()):
            col  = c0 if i == 0 else c1
            vals = data[metric]
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=label, color=col,
                          alpha=0.85 if i == 0 else 0.95)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom",
                        fontsize=7 + i * 0.5, color=col,
                        fontweight="bold" if i == 1 else "normal",
                        alpha=0.8 if i == 0 else 1.0)

        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score" if ax == axes[0] else "")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        tp_idx = CLASSES.index("teleport")
        ax.axvspan(tp_idx - 0.45, tp_idx + 0.45, color="yellow", alpha=0.08, zorder=0)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved chart → {out}")


if __name__ == "__main__":
    from ultralytics import YOLO

    eval_results = {}
    for label, pt in MODELS.items():
        if not Path(pt).exists():
            print(f"SKIP {label} — {pt} not found")
            continue
        print(f"\n{'='*60}")
        print(f"Evaluating: {label}")
        model = YOLO(pt)
        m = model.val(data=str(YAML_PATH), imgsz=320, conf=0.25, iou=0.5, verbose=False)
        p  = np.array(m.box.p)
        r  = np.array(m.box.r)
        a  = np.array(m.box.ap50)
        f1 = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
        eval_results[label] = {"P": p, "R": r, "F1": f1, "mAP50": a}

        names = [m.names[i] for i in range(len(p))]
        print(f"\n  {'Class':<16} {'P':>7} {'R':>7} {'F1':>7} {'mAP50':>7}")
        print(f"  {'-'*46}")
        for n, pi, ri, fi, ai in zip(names, p, r, f1, a):
            print(f"  {n:<16} {pi:>7.3f} {ri:>7.3f} {fi:>7.3f} {ai:>7.3f}")
        print(f"  {'-'*46}")
        print(f"  {'mean':<16} {p.mean():>7.3f} {r.mean():>7.3f} {f1.mean():>7.3f} {a.mean():>7.3f}")

    Path(YAML_PATH).unlink(missing_ok=True)

    if eval_results:
        _plot(eval_results, _CP3 / "eval_compare.png")
