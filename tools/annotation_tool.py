"""
Minimap Annotation Tool
-----------------------
Keyboard:
  A / 1        - set class: ally  (green)
  E / 2        - set class: enemy (red)
  T / 3        - set class: teleport
  R / 4        - set class: recall
  C / 5        - set class: champion_icon
  Left / Right - previous / next image
  Z            - undo last box
  Del          - delete selected box (click to select first)
  S / Ctrl+S   - save labels
  Ctrl+Z       - undo

NOTE: drawing an ally or enemy box automatically also adds a champion_icon
      box at the same position (same-domain co-label for the champion model).

Run (from repo root):
  1. Put PNG/JPG screenshots in:  session/images/
  2. python -m tools.annotation_tool
  3. Labels are saved to:           session/labels/

Run with a dataset split folder:
  python -m tools.annotation_tool path/to/images/train

Requires: pip install pillow opencv-python numpy
"""

import shutil
import sys as _sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext

import cv2
import numpy as np
from PIL import Image, ImageTk

_ROOT = Path(__file__).resolve().parent.parent

# Parse --labels PATH before standard arg handling
_labels_override: Path | None = None
_args = _sys.argv[1:]
if "--labels" in _args:
    _li = _args.index("--labels")
    _labels_override = Path(_args[_li + 1])
    _args = _args[:_li] + _args[_li + 2:]
    _sys.argv = [_sys.argv[0]] + _args

if len(_sys.argv) > 1:
    _split = Path(_sys.argv[1])
    IMG_DIR = _split
    LABEL_DIR = _labels_override if _labels_override else (_split.parent.parent / "labels" / _split.name)
else:
    SESSION_DIR = _ROOT / "session"
    IMG_DIR = SESSION_DIR / "images"
    LABEL_DIR = _labels_override if _labels_override else SESSION_DIR / "labels"

LABEL_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

_DEFAULT_CLASSES = ["ally", "enemy", "teleport", "recall", "champion_icon"]

# Classes that automatically also emit a champion_icon co-label when drawn
_CHAMPION_CO_LABEL_CLASSES = {"ally", "enemy"}

# --no-co-label flag disables auto champion_icon emission
_AUTO_CO_LABEL = "--no-co-label" not in _sys.argv
if not _AUTO_CO_LABEL:
    _sys.argv.remove("--no-co-label")


def _load_classes_from_yaml(split_path: Path):
    for yaml_name in ("data.yaml", "dataset.yaml"):
        yaml_path = split_path.parent.parent / yaml_name
        if yaml_path.exists():
            import re

            text = yaml_path.read_text()
            names = re.findall(r"^\s*-\s*(\S+)", text, re.MULTILINE)
            if names:
                for extra in ("teleport", "recall", "champion_icon"):
                    if extra not in names:
                        names.append(extra)
                # drop legacy map class if present
                names = [n for n in names if n != "map"]
                return names
    return list(_DEFAULT_CLASSES)


CLASSES = (
    _load_classes_from_yaml(IMG_DIR)
    if len(_sys.argv) > 1
    else list(_DEFAULT_CLASSES)
)

_PALETTE = [
    "#22dd44",
    "#dd2244",
    "#ffcc00",
    "#aa66ff",
    "#2299ff",
    "#ffaa22",
    "#cc44ff",
    "#ff66aa",
    "#44ffee",
    "#ffff44",
]
COLORS = {i: _PALETTE[i % len(_PALETTE)] for i in range(len(CLASSES))}

# Canvas grows to fit each image (aspect ratio kept); capped by screen minus UI chrome.
_CANVAS_MARGIN_X = 80
_CANVAS_MARGIN_Y = 460   # team row + effect row + nav + hint + status bar + padding
_CANVAS_MIN_W, _CANVAS_MIN_H = 320, 240

HELP_BODY = """QUICK START
1. Put your minimap screenshots in the images folder (see path in the title bar).
2. Pick a label type with the colored buttons below (or press 1–5 / A E T R M).
3. Click and drag on the image to draw a box around the thing you labeled.
4. Press S or click Save — your work is saved for the current image.
5. Use Next / Prev to move between images (each image has its own .txt file).

WHAT EACH LABEL MEANS
• Ally          — friendly champion dot on the minimap
• Enemy         — enemy champion dot on the minimap
• Teleport      — teleport animation / icon / beam on minimap
• Recall        — recall channel / animation on minimap
• Champion Icon — champion portrait icon (auto-added for every ally/enemy box)

NOTE: Every time you draw an Ally or Enemy box, a Champion Icon box is
automatically added at the same position so both models get trained at once.

MOUSE
• Left drag     — draw a new box (uses the selected label). You can overlap boxes or draw inside another.
• Left click    — release without dragging: select the topmost box under the cursor (thick dashed outline)
• Right click   — delete the box under the cursor

KEYBOARD
• 1–9           — select label class 1–9 (defaults: 1–5 = ally, enemy, teleport, recall, champion_icon)
• A — Ally   E — Enemy   T — Teleport   R — Recall   C — Champion Icon
• Left / Right  — previous / next image (auto-saves current first)
• S, Ctrl+S     — save labels for this image
• Z, Ctrl+Z     — undo last box
• Del           — delete selected box (select with click first)

TIPS
• Save often (S). Next/Prev also saves automatically.
• Boxes are stored in YOLO format (normalized center x,y and width,height).
"""


# ── helpers ──────────────────────────────────────────────────────────────────
def load_labels(path: Path):
    boxes = []
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
                boxes.append([cls, cx, cy, w, h])
    return boxes


def save_labels(path: Path, boxes):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in boxes]
    path.write_text("\n".join(lines))


def yolo_to_px(cx, cy, bw, bh, W, H):
    x1 = (cx - bw / 2) * W
    y1 = (cy - bh / 2) * H
    x2 = (cx + bw / 2) * W
    y2 = (cy + bh / 2) * H
    return x1, y1, x2, y2


def px_to_yolo(x1, y1, x2, y2, W, H):
    cx = ((x1 + x2) / 2) / W
    cy = ((y1 + y2) / 2) / H
    bw = abs(x2 - x1) / W
    bh = abs(y2 - y1) / H
    return cx, cy, bw, bh


def _class_index(name: str) -> int | None:
    try:
        return CLASSES.index(name.lower())
    except ValueError:
        return None


class AnnotationTool:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"RadarRift labeler — images: {IMG_DIR.resolve()}")
        self.root.configure(bg="#1a1a2e")

        self.images = sorted(IMG_DIR.glob("*.png")) + sorted(IMG_DIR.glob("*.jpg"))
        self.idx = 0
        self.boxes = []        # [cls, cx, cy, bw, bh]  — YOLO fields only
        self._groups: list[int] = []   # parallel list: group id per box (-1 = loaded)
        self._group_ctr = 0    # incremented each draw; all boxes in one draw share same id
        self.sel = -1
        # cur_team: 0=ally, 1=enemy  (always required)
        # cur_effect: None | 2=teleport | 3=recall  (optional)
        self.cur_team = 0
        self.cur_effect = None
        self._last_draw_count = 0   # boxes emitted by last draw, for undo grouping
        self.drag_start = None
        self.drag_rect = None
        self._press_hit = -1
        self.tk_img = None
        self.img_w = self.img_h = 1
        self.disp_w = 640
        self.disp_h = 480

        self._build_ui()
        self._bind_keys()

        if not self.images:
            self._show_empty_folder_help()
        else:
            self._load_image()

    def _show_empty_folder_help(self):
        self.lbl_file.config(text="(no images yet)")
        self.lbl_count.config(text="0 / 0")
        messagebox.showinfo(
            "No images found",
            f"No PNG or JPG files in:\n\n{IMG_DIR.resolve()}\n\n"
            "Copy some minimap screenshots into that folder, then restart this tool.",
            parent=self.root,
        )

    def _show_help(self):
        win = tk.Toplevel(self.root)
        win.title("How to use — RadarRift labeler")
        win.configure(bg="#1a1a2e")
        win.geometry("520x420")
        txt = scrolledtext.ScrolledText(
            win,
            wrap="word",
            bg="#252540",
            fg="#e8e8ff",
            insertbackground="white",
            font=("Segoe UI", 10),
            padx=12,
            pady=12,
        )
        txt.pack(fill="both", expand=True, padx=8, pady=8)
        txt.insert("1.0", HELP_BODY)
        txt.config(state="disabled")
        tk.Button(
            win,
            text="Close",
            command=win.destroy,
            bg="#4444aa",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padx=16,
            pady=6,
        ).pack(pady=(0, 8))

    def _build_ui(self):
        BG = "#1a1a2e"
        FG = "#e8e8ff"
        ACC = "#4444aa"

        menubar = tk.Menu(self.root, tearoff=0, bg=BG, fg=FG)
        self.root.config(menu=menubar)
        help_m = tk.Menu(menubar, tearoff=0, bg="#252540", fg=FG)
        menubar.add_cascade(label="Help", menu=help_m)
        help_m.add_command(label="Instructions…", command=self._show_help)
        help_m.add_command(label="Open images folder", command=self._open_images_folder)

        top = tk.Frame(self.root, bg=BG)
        top.pack(fill="x", padx=10, pady=(8, 4))

        tk.Label(
            top,
            text="Draw boxes on the minimap — pick a label, then drag.",
            bg=BG,
            fg="#aaaacc",
            font=("Segoe UI", 10),
        ).pack(anchor="w")

        row2 = tk.Frame(self.root, bg=BG)
        row2.pack(fill="x", padx=10, pady=(0, 4))

        self.lbl_file = tk.Label(
            row2, text="", bg=BG, fg=FG, font=("Consolas", 10, "bold")
        )
        self.lbl_file.pack(side="left")

        self.lbl_count = tk.Label(
            row2, text="", bg=BG, fg="#8888cc", font=("Consolas", 10)
        )
        self.lbl_count.pack(side="right")

        self.canvas = tk.Canvas(
            self.root,
            width=self.disp_w,
            height=self.disp_h,
            bg="#000010",
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.pack(padx=10, pady=6)

        bot = tk.Frame(self.root, bg=BG)
        bot.pack(fill="x", padx=10, pady=(0, 6))

        # ── Row 1: Team (required, mutually exclusive) ────────────────────────
        team_frame = tk.Frame(self.root, bg=BG)
        team_frame.pack(fill="x", padx=10, pady=(2, 0))
        tk.Label(team_frame, text="Team:", bg=BG, fg="#8888aa",
                 font=("Segoe UI", 9, "bold"), width=7, anchor="w").pack(side="left")

        self._team_btns = []
        team_defs = [
            (0, "ally",  "#22dd44", "A"),
            (1, "enemy", "#dd2244", "E"),
        ]
        for tidx, tname, tcolor, tkey in team_defs:
            btn = tk.Button(
                team_frame,
                text=f"{'✔  ' if tidx == 0 else '    '}{tname.title()}  [{tkey}]",
                command=lambda t=tidx: self._set_team(t),
                bg="#1a1a2e", fg=tcolor,
                activebackground="#333355",
                font=("Segoe UI", 10, "bold"),
                relief="flat", bd=0, padx=10, pady=5,
            )
            btn.pack(side="left", padx=(0, 8))
            self._team_btns.append(btn)

        # champion_icon co-label toggle
        self._co_label_var = tk.BooleanVar(value=_AUTO_CO_LABEL)
        co_cb = tk.Checkbutton(
            team_frame, text="+ champion_icon  (auto)",
            variable=self._co_label_var,
            bg=BG, fg="#44ffee", selectcolor="#1a1a2e",
            activebackground=BG, activeforeground="#44ffee",
            font=("Segoe UI", 9, "italic"),
        )
        co_cb.pack(side="left", padx=(16, 0))

        # ── Row 2: Effect (optional, mutually exclusive) ──────────────────────
        eff_frame = tk.Frame(self.root, bg=BG)
        eff_frame.pack(fill="x", padx=10, pady=(4, 2))
        tk.Label(eff_frame, text="Effect:", bg=BG, fg="#8888aa",
                 font=("Segoe UI", 9, "bold"), width=7, anchor="w").pack(side="left")

        self._effect_btns = []
        effect_defs = [
            (None,  "none",      "#666688", "N"),
            (2,     "teleport",  "#ffcc00", "T"),
            (3,     "recall",    "#aa66ff", "R"),
        ]
        for eidx, ename, ecolor, ekey in effect_defs:
            btn = tk.Button(
                eff_frame,
                text=f"{'✔  ' if eidx is None else '    '}{ename.title()}  [{ekey}]",
                command=lambda ef=eidx: self._set_effect(ef),
                bg="#1a1a2e", fg=ecolor,
                activebackground="#333355",
                font=("Segoe UI", 10, "bold"),
                relief="flat", bd=0, padx=10, pady=5,
            )
            btn.pack(side="left", padx=(0, 8))
            self._effect_btns.append(btn)

        nav = tk.Frame(self.root, bg=BG)
        nav.pack(fill="x", padx=10, pady=(4, 6))

        tk.Button(
            nav,
            text="◀ Previous image",
            command=self._prev,
            bg=ACC,
            fg=FG,
            font=("Segoe UI", 10),
            relief="flat",
            width=16,
        ).pack(side="left", padx=(0, 6))
        tk.Button(
            nav,
            text="Next image ▶",
            command=self._next,
            bg=ACC,
            fg=FG,
            font=("Segoe UI", 10),
            relief="flat",
            width=16,
        ).pack(side="left", padx=0)

        tk.Button(
            nav,
            text="Save",
            command=self._save,
            bg="#2a5a2a",
            fg="#aaffaa",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            width=10,
        ).pack(side="right", padx=(6, 0))
        tk.Button(
            nav,
            text="Undo last box",
            command=self._undo,
            bg="#4a3a2a",
            fg="#ffcc88",
            font=("Segoe UI", 10),
            relief="flat",
            width=14,
        ).pack(side="right")

        self.lbl_class = tk.Label(
            nav, text="", bg=BG, font=("Segoe UI", 10, "bold")
        )
        self.lbl_class.pack(side="left", padx=(16, 0))

        self._hint_lbl = tk.Label(
            self.root,
            text=(
                "Drag = new box  •  Click = select  •  Right-click = delete  •  "
                "S = save  •  ← → = prev/next  •  Z = undo"
            ),
            bg=BG,
            fg="#555577",
            font=("Segoe UI", 9),
            wraplength=self.disp_w + 40,
            justify="center",
        )
        self._hint_lbl.pack(pady=(0, 2))

        # ── Status bar: all options shown with live ✔/○ per selection ──────
        status_frame = tk.Frame(self.root, bg="#0d0d1a", pady=6)
        status_frame.pack(fill="x", padx=0, pady=(0, 0))

        tk.Label(status_frame, text="Will draw:", bg="#0d0d1a",
                 fg="#555577", font=("Segoe UI", 9, "bold")).pack(side="left", padx=(10, 8))

        # separator helper
        def _sep():
            tk.Label(status_frame, text="|", bg="#0d0d1a",
                     fg="#333355", font=("Segoe UI", 11)).pack(side="left", padx=(4, 8))

        # champion_icon — always active, static
        self._chip_champion = tk.Label(
            status_frame, text="✔  champion_icon",
            bg="#0d2a2a", fg="#44ffee", font=("Segoe UI", 10, "bold"), padx=8, pady=3)
        self._chip_champion.pack(side="left", padx=(0, 4))

        _sep()

        # team chips
        self._chip_ally = tk.Label(
            status_frame, text="", bg="#0d0d1a", fg="#22dd44",
            font=("Segoe UI", 10, "bold"), padx=8, pady=3)
        self._chip_ally.pack(side="left", padx=(0, 4))

        self._chip_enemy = tk.Label(
            status_frame, text="", bg="#0d0d1a", fg="#dd2244",
            font=("Segoe UI", 10, "bold"), padx=8, pady=3)
        self._chip_enemy.pack(side="left", padx=(0, 4))

        _sep()

        # effect chips
        self._chip_none = tk.Label(
            status_frame, text="", bg="#0d0d1a", fg="#555577",
            font=("Segoe UI", 10, "bold"), padx=8, pady=3)
        self._chip_none.pack(side="left", padx=(0, 4))

        self._chip_teleport = tk.Label(
            status_frame, text="", bg="#0d0d1a", fg="#ffcc00",
            font=("Segoe UI", 10, "bold"), padx=8, pady=3)
        self._chip_teleport.pack(side="left", padx=(0, 4))

        self._chip_recall = tk.Label(
            status_frame, text="", bg="#0d0d1a", fg="#aa66ff",
            font=("Segoe UI", 10, "bold"), padx=8, pady=3)
        self._chip_recall.pack(side="left", padx=(0, 4))

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Button-3>", self._on_right_click)

        self._update_ui()

    def _open_images_folder(self):
        import os
        import subprocess

        p = str(IMG_DIR.resolve())
        try:
            os.startfile(p)
        except Exception:
            try:
                subprocess.run(["explorer", p], check=False)
            except Exception:
                messagebox.showinfo("Folder path", p, parent=self.root)

    def _bind_keys(self):
        self.root.bind("<Left>", lambda e: self._prev())
        self.root.bind("<Right>", lambda e: self._next())
        for i in range(min(9, len(CLASSES))):
            pass  # number keys replaced by A/E/T/R/N hotkeys

        # Team
        self.root.bind("a", lambda e: self._set_team(0))
        self.root.bind("A", lambda e: self._set_team(0))
        self.root.bind("e", lambda e: self._set_team(1))
        self.root.bind("E", lambda e: self._set_team(1))
        # Effect
        it = _class_index("teleport")
        ir = _class_index("recall")
        if it is not None:
            self.root.bind("t", lambda e: self._set_effect(it))
            self.root.bind("T", lambda e: self._set_effect(it))
        if ir is not None:
            self.root.bind("r", lambda e: self._set_effect(ir))
            self.root.bind("R", lambda e: self._set_effect(ir))
        self.root.bind("n", lambda e: self._set_effect(None))
        self.root.bind("N", lambda e: self._set_effect(None))

        self.root.bind("s", lambda e: self._save())
        self.root.bind("<Control-s>", lambda e: self._save())
        self.root.bind("z", lambda e: self._undo())
        self.root.bind("<Control-z>", lambda e: self._undo())
        self.root.bind("<Delete>", lambda e: self._delete_selected())

    def _compute_display_size(self, iw: int, ih: int) -> tuple[int, int]:
        """Scale image to fit the screen while preserving aspect ratio (no letterboxing)."""
        try:
            sw = max(_CANVAS_MIN_W, self.root.winfo_screenwidth() - _CANVAS_MARGIN_X)
            sh = max(_CANVAS_MIN_H, self.root.winfo_screenheight() - _CANVAS_MARGIN_Y)
        except tk.TclError:
            sw, sh = 1280, 720
        scale = min(sw / max(iw, 1), sh / max(ih, 1))
        dw = max(1, int(round(iw * scale)))
        dh = max(1, int(round(ih * scale)))
        return dw, dh

    def _load_image(self):
        if not self.images:
            return
        p = self.images[self.idx]
        pil_raw = Image.open(str(p)).convert("RGB")
        img_rgb = np.array(pil_raw)
        self.img_h, self.img_w = img_rgb.shape[:2]
        self.disp_w, self.disp_h = self._compute_display_size(self.img_w, self.img_h)
        _rs = (
            Image.Resampling.LANCZOS
            if hasattr(Image, "Resampling")
            else Image.LANCZOS
        )
        pil = Image.fromarray(img_rgb).resize((self.disp_w, self.disp_h), _rs)
        self.tk_img = ImageTk.PhotoImage(pil)
        self.canvas.config(width=self.disp_w, height=self.disp_h)
        if hasattr(self, "_hint_lbl"):
            self._hint_lbl.config(wraplength=max(self.disp_w + 40, 400))
        lbl_path = LABEL_DIR / f"{p.stem}.txt"
        if not lbl_path.exists():
            auto = _ROOT / "dataset_minimap" / "labels" / "train" / f"{p.stem}.txt"
            if auto.exists():
                shutil.copy(auto, lbl_path)
        self.boxes   = load_labels(lbl_path)
        self._groups = [-1] * len(self.boxes)   # loaded boxes have no group
        self.sel = -1

        self.lbl_file.config(text=p.name)
        self.lbl_count.config(text=f"Image {self.idx + 1} of {len(self.images)}")
        self._redraw()

    def _redraw(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        sx = self.disp_w / self.img_w
        sy = self.disp_h / self.img_h
        for i, (cls, cx, cy, bw, bh) in enumerate(self.boxes):
            if cls < 0 or cls >= len(CLASSES):
                continue
            x1, y1, x2, y2 = yolo_to_px(cx, cy, bw, bh, self.img_w, self.img_h)
            dx1, dy1 = x1 * sx, y1 * sy
            dx2, dy2 = x2 * sx, y2 * sy
            color = COLORS[cls]
            thick = 3 if i == self.sel else 2
            dash = (4, 2) if i == self.sel else None
            tag = f"box_{i}"
            self.canvas.create_rectangle(
                dx1, dy1, dx2, dy2, outline=color, width=thick, dash=dash, tags=tag
            )
            lname = CLASSES[cls]
            tw = max(60, len(lname) * 7 + 8)
            self.canvas.create_rectangle(
                dx1, dy1 - 18, dx1 + tw, dy1, fill=color, outline=""
            )
            self.canvas.create_text(
                dx1 + 4,
                dy1 - 9,
                anchor="w",
                text=lname,
                fill="white",
                font=("Segoe UI", 8, "bold"),
            )
        self._update_ui()

    def _set_team(self, team: int):
        self.cur_team = team
        self._update_ui()

    def _set_effect(self, effect):
        self.cur_effect = effect
        self._update_ui()

    def _update_ui(self):
        team_names  = ["ally", "enemy"]
        team_colors = ["#22dd44", "#dd2244"]
        effect_map  = {None: (0, "none", "#666688"), 2: (1, "teleport", "#ffcc00"), 3: (2, "recall", "#aa66ff")}

        for i, btn in enumerate(self._team_btns):
            active = i == self.cur_team
            tname  = team_names[i]
            tkey   = "A" if i == 0 else "E"
            tcolor = team_colors[i]
            btn.config(
                text=f"{'✔  ' if active else '    '}{tname.title()}  [{tkey}]",
                relief="sunken" if active else "flat",
                bg="#2a2a4a" if active else "#1a1a2e",
            )

        eff_order = [None, 2, 3]
        eff_keys  = ["N", "T", "R"]
        eff_names = ["none", "teleport", "recall"]
        eff_colors= ["#666688", "#ffcc00", "#aa66ff"]
        for i, btn in enumerate(self._effect_btns):
            ev = eff_order[i]
            active = ev == self.cur_effect
            btn.config(
                text=f"{'✔  ' if active else '    '}{eff_names[i].title()}  [{eff_keys[i]}]",
                relief="sunken" if active else "flat",
                bg="#2a2a4a" if active else "#1a1a2e",
            )

        tcolor = team_colors[self.cur_team]
        tname  = team_names[self.cur_team]
        parts  = ["champion_icon", tname]
        if self.cur_effect is not None:
            _, eff_name, _ = effect_map[self.cur_effect]
            parts.append(eff_name)
        self.lbl_class.config(
            text="Drawing: " + " + ".join(parts), fg=tcolor
        )

        # update bottom status chips
        if hasattr(self, "_chip_ally"):
            # team
            if self.cur_team == 0:
                self._chip_ally.config(text="✔  ally",   bg="#0a2a0a")
                self._chip_enemy.config(text="○  enemy",  bg="#0d0d1a")
            else:
                self._chip_ally.config(text="○  ally",   bg="#0d0d1a")
                self._chip_enemy.config(text="✔  enemy",  bg="#2a0a0a")
            # effect
            if self.cur_effect is None:
                self._chip_none.config(    text="✔  none",     bg="#1a1a2e")
                self._chip_teleport.config(text="○  teleport", bg="#0d0d1a")
                self._chip_recall.config(  text="○  recall",   bg="#0d0d1a")
            elif self.cur_effect == 2:
                self._chip_none.config(    text="○  none",     bg="#0d0d1a")
                self._chip_teleport.config(text="✔  teleport", bg="#1a1a08")
                self._chip_recall.config(  text="○  recall",   bg="#0d0d1a")
            else:
                self._chip_none.config(    text="○  none",     bg="#0d0d1a")
                self._chip_teleport.config(text="○  teleport", bg="#0d0d1a")
                self._chip_recall.config(  text="✔  recall",   bg="#150d20")

    def _canvas_to_img(self, cx, cy):
        sx = self.disp_w / self.img_w
        sy = self.disp_h / self.img_h
        return cx / sx, cy / sy

    def _box_at(self, cx, cy):
        ix, iy = self._canvas_to_img(cx, cy)
        for i in range(len(self.boxes) - 1, -1, -1):
            cls, c_cx, c_cy, bw, bh = self.boxes[i]
            if cls < 0 or cls >= len(CLASSES):
                continue
            x1, y1, x2, y2 = yolo_to_px(c_cx, c_cy, bw, bh, self.img_w, self.img_h)
            if x1 <= ix <= x2 and y1 <= iy <= y2:
                return i
        return -1

    def _on_press(self, ev):
        # Defer selection to release so a drag starting inside a box still creates a new box.
        self._press_hit = self._box_at(ev.x, ev.y)
        self.drag_start = (ev.x, ev.y)
        self.drag_rect = None

    def _on_drag(self, ev):
        if self.drag_start is None:
            return
        if self.drag_rect:
            self.canvas.delete(self.drag_rect)
        x0, y0 = self.drag_start
        color = "#22dd44" if self.cur_team == 0 else "#dd2244"
        self.drag_rect = self.canvas.create_rectangle(
            x0, y0, ev.x, ev.y, outline=color, width=2, dash=(4, 2)
        )

    def _on_release(self, ev):
        if self.drag_start is None:
            return
        x0, y0 = self.drag_start
        x1, y1 = ev.x, ev.y
        if self.drag_rect:
            self.canvas.delete(self.drag_rect)
            self.drag_rect = None
        self.drag_start = None

        # Chebyshev distance: both axes small = click (select); thin drags still create boxes.
        if max(abs(x1 - x0), abs(y1 - y0)) < 5:
            if self._press_hit >= 0:
                self.sel = self._press_hit
            else:
                self.sel = -1
            self._redraw()
            return

        ix0, iy0 = self._canvas_to_img(x0, y0)
        ix1, iy1 = self._canvas_to_img(x1, y1)
        cx, cy, bw, bh = px_to_yolo(ix0, iy0, ix1, iy1, self.img_w, self.img_h)
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.001, min(1.0, bw))
        bh = max(0.001, min(1.0, bh))

        champ_idx = _class_index("champion_icon")
        to_add = []
        if champ_idx is not None and self._co_label_var.get():
            to_add.append([champ_idx, cx, cy, bw, bh])          # champion_icon (optional)
        to_add.append([self.cur_team, cx, cy, bw, bh])           # ally or enemy
        if self.cur_effect is not None:
            to_add.append([self.cur_effect, cx, cy, bw, bh])     # teleport or recall

        gid = self._group_ctr
        self._group_ctr += 1
        self.boxes.extend(to_add)
        self._groups.extend([gid] * len(to_add))
        self._last_draw_count = len(to_add)
        self.sel = len(self.boxes) - 1
        self._redraw()

    def _delete_group(self, idx: int):
        """Remove all boxes that share the same group as box[idx]."""
        if idx < 0 or idx >= len(self.boxes):
            return
        gid = self._groups[idx]
        if gid == -1:
            # loaded box with no group — delete just that one
            self.boxes.pop(idx)
            self._groups.pop(idx)
        else:
            keep_b = [b for b, g in zip(self.boxes, self._groups) if g != gid]
            keep_g = [g for g in self._groups if g != gid]
            self.boxes   = keep_b
            self._groups = keep_g
        self.sel = -1
        self._redraw()

    def _on_right_click(self, ev):
        hit = self._box_at(ev.x, ev.y)
        if hit >= 0:
            self._delete_group(hit)

    def _delete_selected(self):
        if 0 <= self.sel < len(self.boxes):
            self._delete_group(self.sel)

    def _undo(self):
        n = max(1, self._last_draw_count)
        if self.boxes:
            del self.boxes[-n:]
            del self._groups[-n:]
            self._last_draw_count = 0
            self.sel = -1
            self._redraw()

    def _save(self):
        if not self.images:
            return
        p = self.images[self.idx]
        lbl_path = LABEL_DIR / f"{p.stem}.txt"
        save_labels(lbl_path, self.boxes)
        self.lbl_file.config(text=f"Saved ✓  {p.name}")
        self.root.after(1200, lambda: self.lbl_file.config(text=p.name))

    def _prev(self):
        if not self.images:
            return
        self._save()
        self.idx = (self.idx - 1) % len(self.images)
        self._load_image()

    def _next(self):
        if not self.images:
            return
        self._save()
        self.idx = (self.idx + 1) % len(self.images)
        self._load_image()


def main():
    root = tk.Tk()
    root.resizable(True, True)
    AnnotationTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
