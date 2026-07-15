"""
Generate CP2 vs CP3 per-class metric comparison bar chart.
Saves to cp3/eval_compare.png.

Run from repo root:
  python tools/plot_yolo_compare.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

CLASSES = ["ally", "enemy", "teleport", "recall", "champion_icon"]

# Results from eval_yolo_compare.py run on session_teleport dataset
CP2 = {
    "P":    [0.355, 0.374, 0.886, 0.530, 0.329],
    "R":    [0.814, 0.769, 0.348, 0.966, 0.971],
    "F1":   [0.495, 0.503, 0.500, 0.685, 0.491],
    "mAP50":[0.440, 0.418, 0.623, 0.791, 0.439],
}
CP3 = {
    "P":    [0.977, 0.944, 0.969, 0.964, 0.961],
    "R":    [0.800, 0.833, 0.832, 0.910, 0.964],
    "F1":   [0.880, 0.885, 0.895, 0.936, 0.963],
    "mAP50":[0.895, 0.903, 0.912, 0.954, 0.977],
}

x      = np.arange(len(CLASSES))
width  = 0.18
metrics = ["P", "R", "F1", "mAP50"]
colors_cp2 = ["#6baed6", "#74c476", "#fd8d3c", "#9e9ac8"]
colors_cp3 = ["#2171b5", "#238b45", "#d94801", "#54278f"]

fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
fig.suptitle("CP2 (radarrift_final4) vs CP3 (continue_Teleport)\nPer-class YOLO metrics on session_teleport dataset",
             fontsize=13, fontweight="bold", y=1.01)

for ax, metric, c2, c3 in zip(axes, metrics, colors_cp2, colors_cp3):
    bars2 = ax.bar(x - width/2, CP2[metric], width, label="CP2", color=c2, alpha=0.85)
    bars3 = ax.bar(x + width/2, CP3[metric], width, label="CP3", color=c3, alpha=0.95)

    for bar in bars3:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=7.5, color=c3, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=7, color=c2, alpha=0.8)

    ax.set_title(metric, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score" if ax == axes[0] else "")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # highlight teleport row
    tp_idx = CLASSES.index("teleport")
    ax.axvspan(tp_idx - 0.45, tp_idx + 0.45, color="yellow", alpha=0.08, zorder=0)

fig.tight_layout()
out = Path("cp3/eval_compare.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
