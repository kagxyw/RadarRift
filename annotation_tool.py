"""
Minimap Annotation Tool
-----------------------
Keyboard:
  A / 1        - set class: ally  (green)
  E / 2        - set class: enemy (red)
  M / 5        - set class: map (minimap bounds / terrain)
  Left / Right - previous / next image
  Z            - undo last box
  Del          - delete selected box (click to select first)
  S / Ctrl+S   - save labels
  Ctrl+Z       - undo

Run:
  1. Put PNG/JPG screenshots in:  session/images/
  2. Double-click this file or run:  python annotation_tool.py
  3. Labels are saved to:           session/labels/

Run with a dataset split folder:
  python annotation_tool.py path/to/images/train

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

if len(_sys.argv) > 1:
    _split = Path(_sys.argv[1])
    IMG_DIR = _split
    LABEL_DIR = _split.parent.parent / "labels" / _split.name
else:
    SESSION_DIR = Path("session")
    IMG_DIR = SESSION_DIR / "images"
    LABEL_DIR = SESSION_DIR / "labels"

LABEL_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

_DEFAULT_CLASSES = ["ally", "enemy", "teleport", "recall", "map"]


def _load_classes_from_yaml(split_path: Path):
    for yaml_name in ("data.yaml", "dataset.yaml"):
        yaml_path = split_path.parent.parent / yaml_name
        if yaml_path.exists():
            import re

            text = yaml_path.read_text()
            names = re.findall(r"^\s*-\s*(\S+)", text, re.MULTILINE)
            if names:
                for extra in ("teleport", "recall", "map"):
                    if extra not in names:
                        names.append(extra)
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
DISPLAY_SZ = 768

HELP_BODY = """QUICK START
1. Put your minimap screenshots in the images folder (see path in the title bar).
2. Pick a label type with the colored buttons below (or press 1–5 / A E T R M).
3. Click and drag on the image to draw a box around the thing you labeled.
4. Press S or click Save — your work is saved for the current image.
5. Use Next / Prev to move between images (each image has its own .txt file).

WHAT EACH LABEL MEANS
• Ally      — friendly champion icon on the minimap
• Enemy     — enemy champion icon
• Teleport  — teleport animation / icon / beam on minimap (if you are training that)
• Recall    — recall channel / animation on minimap (if you are training that)
• Map       — minimap panel / playable area bounds (full map ROI or mask region)

MOUSE
• Left drag     — draw a new box (uses the selected label)
• Left click    — select a box (thick dashed outline)
• Right click   — delete the box under the cursor

KEYBOARD
• 1–9           — select label class 1–9 (defaults: 1–5 = ally, enemy, teleport, recall, map)
• A — Ally   E — Enemy   T — Teleport   R — Recall   M — Map
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
        self.boxes = []
        self.sel = -1
        self.cur_class = 0
        self.drag_start = None
        self.drag_rect = None
        self.tk_img = None
        self.img_w = self.img_h = 1

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
            width=DISPLAY_SZ,
            height=DISPLAY_SZ,
            bg="#000010",
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.pack(padx=10, pady=6)

        bot = tk.Frame(self.root, bg=BG)
        bot.pack(fill="x", padx=10, pady=(0, 6))

        tk.Label(
            bot,
            text="Label (next box you draw):",
            bg=BG,
            fg="#8888aa",
            font=("Segoe UI", 9),
        ).pack(anchor="w")

        btn_row = tk.Frame(self.root, bg=BG)
        btn_row.pack(fill="x", padx=10, pady=(0, 4))

        self._class_btns = []
        for i, cname in enumerate(CLASSES):
            color = COLORS[i]
            hotkey = str(i + 1) if i < 9 else ""
            label = f"{cname.replace('_', ' ').title()}"
            if hotkey:
                label = f"{label} [{hotkey}]"
            btn = tk.Button(
                btn_row,
                text=label,
                command=lambda c=i: self._set_class(c),
                bg="#1a1a2e",
                fg=color,
                activebackground="#333355",
                font=("Segoe UI", 9, "bold"),
                relief="flat",
                bd=0,
                padx=8,
                pady=4,
            )
            btn.pack(side="left", padx=(0, 6))
            self._class_btns.append(btn)

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
            nav, text="", bg=BG, font=("Segoe UI", 11, "bold")
        )
        self.lbl_class.pack(side="left", padx=(16, 0))

        hint = tk.Label(
            self.root,
            text=(
                "Drag = new box  •  Click box = select  •  Right-click = delete  •  "
                "S = save  •  ← → = prev/next image  •  Help menu for full guide"
            ),
            bg=BG,
            fg="#666688",
            font=("Segoe UI", 9),
            wraplength=DISPLAY_SZ + 40,
            justify="center",
        )
        hint.pack(pady=(0, 8))

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Button-3>", self._on_right_click)

        self._update_class_ui()

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
            self.root.bind(str(i + 1), lambda e, c=i: self._set_class(c))

        ia = _class_index("ally")
        ie = _class_index("enemy")
        it = _class_index("teleport")
        ir = _class_index("recall")
        im = _class_index("map")
        if ia is not None:
            self.root.bind("a", lambda e: self._set_class(ia))
            self.root.bind("A", lambda e: self._set_class(ia))
        if ie is not None:
            self.root.bind("e", lambda e: self._set_class(ie))
            self.root.bind("E", lambda e: self._set_class(ie))
        if it is not None:
            self.root.bind("t", lambda e: self._set_class(it))
            self.root.bind("T", lambda e: self._set_class(it))
        if ir is not None:
            self.root.bind("r", lambda e: self._set_class(ir))
            self.root.bind("R", lambda e: self._set_class(ir))
        if im is not None:
            self.root.bind("m", lambda e: self._set_class(im))
            self.root.bind("M", lambda e: self._set_class(im))

        self.root.bind("s", lambda e: self._save())
        self.root.bind("<Control-s>", lambda e: self._save())
        self.root.bind("z", lambda e: self._undo())
        self.root.bind("<Control-z>", lambda e: self._undo())
        self.root.bind("<Delete>", lambda e: self._delete_selected())

    def _load_image(self):
        if not self.images:
            return
        p = self.images[self.idx]
        pil_raw = Image.open(str(p)).convert("RGB")
        img_rgb = np.array(pil_raw)
        self.img_h, self.img_w = img_rgb.shape[:2]
        pil = Image.fromarray(img_rgb).resize((DISPLAY_SZ, DISPLAY_SZ), Image.NEAREST)
        self.tk_img = ImageTk.PhotoImage(pil)
        lbl_path = LABEL_DIR / f"{p.stem}.txt"
        if not lbl_path.exists():
            auto = Path("dataset_minimap/labels/train") / f"{p.stem}.txt"
            if auto.exists():
                shutil.copy(auto, lbl_path)
        self.boxes = load_labels(lbl_path)
        self.sel = -1

        self.lbl_file.config(text=p.name)
        self.lbl_count.config(text=f"Image {self.idx + 1} of {len(self.images)}")
        self._redraw()

    def _redraw(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        sx = DISPLAY_SZ / self.img_w
        sy = DISPLAY_SZ / self.img_h
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
        self._update_class_ui()

    def _set_class(self, cls):
        self.cur_class = max(0, min(len(CLASSES) - 1, cls))
        self._update_class_ui()

    def _update_class_ui(self):
        for i, btn in enumerate(self._class_btns):
            active = i == self.cur_class
            btn.config(
                relief="sunken" if active else "flat",
                bg="#333355" if active else "#1a1a2e",
            )
        name = CLASSES[self.cur_class]
        color = COLORS[self.cur_class]
        self.lbl_class.config(
            text=f"Drawing: {name.replace('_', ' ').title()}", fg=color
        )

    def _canvas_to_img(self, cx, cy):
        sx = DISPLAY_SZ / self.img_w
        sy = DISPLAY_SZ / self.img_h
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
        hit = self._box_at(ev.x, ev.y)
        if hit >= 0:
            self.sel = hit
            self.drag_start = None
            self._redraw()
        else:
            self.sel = -1
            self.drag_start = (ev.x, ev.y)
            self.drag_rect = None

    def _on_drag(self, ev):
        if self.drag_start is None:
            return
        if self.drag_rect:
            self.canvas.delete(self.drag_rect)
        x0, y0 = self.drag_start
        color = COLORS[self.cur_class]
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

        if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
            return

        ix0, iy0 = self._canvas_to_img(x0, y0)
        ix1, iy1 = self._canvas_to_img(x1, y1)
        cx, cy, bw, bh = px_to_yolo(ix0, iy0, ix1, iy1, self.img_w, self.img_h)
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.001, min(1.0, bw))
        bh = max(0.001, min(1.0, bh))
        self.boxes.append([self.cur_class, cx, cy, bw, bh])
        self.sel = len(self.boxes) - 1
        self._redraw()

    def _on_right_click(self, ev):
        hit = self._box_at(ev.x, ev.y)
        if hit >= 0:
            self.boxes.pop(hit)
            self.sel = -1
            self._redraw()

    def _delete_selected(self):
        if 0 <= self.sel < len(self.boxes):
            self.boxes.pop(self.sel)
            self.sel = -1
            self._redraw()

    def _undo(self):
        if self.boxes:
            self.boxes.pop()
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
    root.resizable(False, False)
    AnnotationTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
