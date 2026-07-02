# cp2 — RadarRift Validation Package

This folder contains the trained model weights, validation dataset, evaluation script, and annotation tool for the **radarrift_final4** minimap tracker model.

---

## Contents

```
cp2/
├── best.pt              # Trained YOLO model (radarrift_final4)
├── eval.py              # Evaluation script — runs model on val set, outputs chart
├── annotation_tool.py   # Label viewer / editor for the validation images
├── eval.png             # Last evaluation output chart
└── vallabels/           # NOT committed — provide your own val images/labels
    ├── images/          # 73 validation frames (PNG)
    └── labels/          # 73 YOLO label files (.txt)
```

---

## Classes

| ID | Name          |
|----|---------------|
| 0  | ally          |
| 1  | enemy         |
| 2  | teleport      |
| 3  | recall        |
| 4  | champion_icon |

---

## Requirements

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics==8.4.21 opencv-python pillow numpy matplotlib pyyaml
```

---

## 1. Evaluate the model

Runs `best.pt` against the 73-frame validation set and prints a per-class table plus saves a bar chart (`eval.png`).

**Run from the repo root (the `RadarRift/` folder):**

```bash
python cp2/eval.py
```

**Output:**

```
Class             Precision     Recall         F1      mAP50
----------------------------------------------------------
ally                 0.9697     0.7887     0.8699     0.8903
enemy                0.9924     0.7238     0.8371     0.8606
teleport             1.0000     0.8889     0.9412     0.9444
recall               0.9333     1.0000     0.9655     0.9950
champion_icon        0.9697     0.9634     0.9666     0.9803
----------------------------------------------------------
mean                 0.9730     0.8730     0.9160     0.9341
```

Chart is saved to `cp2/eval.png`.

---

## 2. View / edit validation labels

Opens a GUI showing each validation frame with its bounding box annotations overlaid. Use it to verify labels or make corrections.

**Run from the repo root (the `RadarRift/` folder):**

```bash
python cp2/annotation_tool.py cp2/vallabels/images/val
```

**Keyboard shortcuts:**

| Key | Action |
|-----|--------|
| `←` / `→` | Previous / next image |
| `A` or `1` | Select class: ally |
| `E` or `2` | Select class: enemy |
| `T` or `3` | Select class: teleport |
| `R` or `4` | Select class: recall |
| `C` or `5` | Select class: champion_icon |
| `Z` | Undo last box |
| `Del` | Delete selected box |
| `S` / `Ctrl+S` | Save labels |
| `Ctrl+Z` | Undo |

Labels are saved back to `vallabels/labels/val/`.

---

## 3. Run inference on a single image (quick test)

```python
from ultralytics import YOLO
model = YOLO("cp2/best.pt")
results = model.predict("path/to/minimap.png", imgsz=320, conf=0.25)
results[0].show()
```
