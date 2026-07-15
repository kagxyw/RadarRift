# RadarRift — CP3

## Download (pre-built Windows executable)

> **[RadarRift-v1.0.0.zip](https://github.com/kagxyw/RadarRift/releases/download/v1.0.0/RadarRift-v1.0.0.zip)**
> (~380 MB, no Python required)

1. Download and extract the zip
2. Run `RadarRift.exe`

## What changed since CP2

### 1. Continued model training (`best.pt` → `continue_Teleport`)

The CP2 model (`radarrift_final4`) had weak teleport detection because the training set only contained a narrow range of TP animation sizes — the model learned a fixed-size expectation for the TP circle and confidently missed detections where the circle was larger or smaller than what it had seen. The core fix was **scale diversity**:

- Annotated **135 real frames** from recordings containing teleport/recall events (`session_teleport/`)
- Synthesised **228 additional images** by copy-pasting real teleport swirl crops onto clean map backgrounds at **randomised scales** (0.4× to 1.1× of the original), teaching the model that the TP circle can appear at many sizes depending on how far into the channel the animation is
- Total training set: **363 images**, **339 teleport annotations**, **89 recall annotations**
- Fine-tuned from CP2 weights (`continue_Teleport`)

#### CP2 vs CP3 YOLO per-class metrics (evaluated on `session_teleport` dataset)

| Class | CP2 P | CP2 R | CP2 F1 | CP2 mAP50 | CP3 P | CP3 R | CP3 F1 | CP3 mAP50 |
|---|---|---|---|---|---|---|---|---|
| ally | 0.355 | 0.814 | 0.495 | 0.440 | 0.977 | 0.800 | 0.880 | 0.895 |
| enemy | 0.374 | 0.769 | 0.503 | 0.418 | 0.944 | 0.833 | 0.885 | 0.903 |
| **teleport** | 0.886 | 0.348 | **0.500** | 0.623 | 0.969 | 0.832 | **0.895** | 0.912 |
| recall | 0.530 | 0.966 | 0.685 | 0.791 | 0.964 | 0.910 | 0.936 | 0.954 |
| champion_icon | 0.329 | 0.971 | 0.491 | 0.439 | 0.961 | 0.964 | 0.963 | 0.977 |
| **mean** | 0.495 | 0.774 | **0.535** | 0.542 | 0.963 | 0.868 | **0.912** | 0.928 |

The CP2 teleport recall of **0.348** is the core problem — the model only detected 1 in 3 teleports it was shown. This is the direct consequence of the fixed-size bias: when the TP circle is at a scale the model hadn't seen, it simply doesn't fire. CP3 brings teleport recall to **0.832** and mAP50 to **0.912**.

Note: these metrics are measured on the same `session_teleport` dataset used for fine-tuning, so they reflect in-distribution performance. In live recordings the improvement is qualitatively even more pronounced because the model now handles the full range of TP animation phases rather than only the mid-channel size.

See **`eval_compare.png`** for the visual chart.

#### Reproducing the comparison

```
# 1. Run YOLO evaluation (requires both model weights in runs/detect/)
python cp3/tools/eval_yolo_compare.py

# 2. Regenerate the chart from hardcoded results (no models needed)
python cp3/tools/plot_yolo_compare.py
```

#### Reproducing the training from scratch

```
# Step 1 — synthesise scale-diverse TP training images
python cp3/tools/synth_teleport_v2.py

# Step 2 — fine-tune from CP2 weights
#   (edit train_continue_teleport.py to set base model and epochs, then:)
python tools/train_continue_teleport.py

# Step 3 — extract crops and train the CNN confirmer
python cp3/tools/extract_tp_crops.py
python cp3/tools/train_tp_cls.py

# Step 4 — evaluate
python cp3/tools/eval_yolo_compare.py
python cp3/tools/eval_tp_cls.py
```

---

### 2. Teleport voice alert

Added a dedicated **"Teleport" TTS audio clip** (`tts_out/Teleport.mp3`) that fires when the tracker confirms a teleport. This uses the same TTS pipeline as the existing champion proximity alerts (`alert_audio.play_tp_alert()`).

---

### 3. Teleport alert routing (the full detection pipeline)

The entire TP detection path was restructured:

**Before (CP2):** Standard YOLO pass at `conf=0.25`, radius-based alert with a per-location bucket cooldown.

**After (CP3):**

1. **Separate low-confidence YOLO pass at `tp_conf=0.08`** — runs in addition to the normal champion detection pass. The lower threshold catches faint early-phase TP animations that `0.25` would discard.

2. **CNN secondary confirmation gate** — every YOLO `teleport` box is cropped and passed to a binary CNN (`tp_confirm_cls.pt`). The CNN outputs a probability; if it's below 0.55 the detection is dropped silently before it can trigger an alert. `recall` boxes skip the CNN (they pass through directly).

3. **Alert covers the entire map** — the earlier proximity radius check was removed. Any confirmed teleport anywhere on the minimap triggers the warning, not just ones within a certain distance of the player.

4. **Global 3-second cooldown** — one alert fires per 3 seconds maximum, regardless of where on the map the TP is.

   **Why 3 seconds:** A teleport animation lasts roughly 3–4 seconds total. The intent is to alert once per teleport event, not once per detected frame.

   **Known limitation:** If the teleport animation lasts longer than 3 seconds (which it can — the channel + travel phase combined can exceed this), the cooldown may expire while the TP is still ongoing, causing a **second alert to fire for the same teleport**. This is a trade-off: a shorter cooldown risks double-alerts on long TPs; a longer cooldown risks missing rapid back-to-back TPs from different champions.

---

### 4. CNN teleport confirmer (`tp_confirm_cls.pt` + `tp_confirm.py`)

A new lightweight binary classifier trained on 64×64 crops of labeled YOLO boxes from `session_teleport`:

- **Positive:** boxes labeled `teleport`
- **Negative:** boxes labeled `ally`, `enemy`, `recall`, `champion_icon`

Training used oversampling (with flip + brightness jitter) to balance the 1:6 class imbalance.

Val-split metrics (threshold = 0.55): **Precision 72.7% / Recall 83.6% / F1 0.778**

---

### 5. TP frame logging

Every time a teleport is confirmed, the minimap frame is saved to `tp_detections/` next to the executable. These frames are the primary data source for reviewing false positives and feeding new negatives back into CNN retraining.

---

## Files

| File | Purpose |
|---|---|
| `best.pt` | YOLO detector (`continue_Teleport`) — 5 classes: ally / enemy / teleport / recall / champion_icon |
| `tp_confirm_cls.pt` | CNN binary classifier weights (teleport vs not_teleport) |
| `tp_confirm.py` | Module used by `tracker.py` to load and run the CNN |
| `eval_compare.png` | CP2 vs CP3 per-class metric chart (pre-generated) |
| `annotation_tool.py` | Label tool for creating/editing YOLO annotations |
| `session_teleport/` | Full training dataset (135 real + 228 synth images, YOLO labels) |
| `dataset_tp_cls/` | CNN binary classifier crops (train/val split) |
| `tools/eval_yolo_compare.py` | Run CP2 vs CP3 YOLO evaluation, prints table |
| `tools/plot_yolo_compare.py` | Regenerate `eval_compare.png` from hardcoded results |
| `tools/extract_tp_crops.py` | Extract labeled 64×64 crops for CNN training |
| `tools/train_tp_cls.py` | Train/retrain the binary CNN confirmer |
| `tools/eval_tp_cls.py` | Evaluate CNN on the held-out val split |
| `tools/eval_tp_cls_full.py` | Evaluate CNN across all labeled data |
| `tools/synth_teleport_v2.py` | Synthesise new TP training images (requires `recordings/` from repo) |

---

## Setup

```
pip install ultralytics torch torchvision opencv-python matplotlib pyyaml
```

All scripts in `cp3/tools/` resolve paths relative to their own location — no hardcoded paths, no need to run from a specific working directory.

---

## Running the evaluations

All commands can be run from anywhere:

```
# CP2 vs CP3 YOLO comparison — prints table and saves eval_compare.png
# Requires CP2 model at runs/detect/radarrift_final4/weights/best.pt
python cp3/tools/eval_yolo_compare.py

# CNN confirmer — val split only
python cp3/tools/eval_tp_cls.py

# CNN confirmer — all labeled data
python cp3/tools/eval_tp_cls_full.py
```

---

## Retraining the CNN confirmer

```
# 1. Rebuild the crop dataset from session_teleport labels
python cp3/tools/extract_tp_crops.py

# 2. Train
python cp3/tools/train_tp_cls.py

# 3. Evaluate
python cp3/tools/eval_tp_cls.py
```

After training, copy the output weights into cp3:
```
copy runs\classify\tp_confirm_cls\weights\best.pt cp3\tp_confirm_cls.pt
```

---

## Synthesising new training images

The model was trained on a limited number of real teleport frames. To improve robustness to different TP circle sizes, `synth_teleport_v2.py` copy-pastes real teleport swirl crops from `session_teleport/` onto clean minimap backgrounds at randomised scales, generating additional labeled images without any manual annotation.

**Requirements:** a folder of raw minimap recording frames to use as backgrounds. By default the script expects `recordings/20260705_004828/` relative to the repo root. To use a different folder, edit the `BG_DIR` variable at the top of the script.

```
python cp3/tools/synth_teleport_v2.py
```

Synthetic images and their label files are written directly into `cp3/session_teleport/images/` and `cp3/session_teleport/labels/`. Re-run training afterwards to incorporate them.
