"""
match_start.py — Loading screen champion identification.

Identifies all 10 champions from a LoL loading screen screenshot and returns
a ChampionRoster ready for the minimap tracker.

Blue side = top row.  Red side = bottom row.

Identification uses a 2-stage pipeline:
  Stage 1 — pHash shortlist
      Compute a 64-bit DCT perceptual hash for the YOLO crop.
      Find the top-K references with smallest Hamming distance.
      This is O(n) but very fast; tolerates crop/scale differences well.
  Stage 2 — ORB verification
      For each shortlisted skin, compute ORB features on the YOLO crop and
      the cached reference portrait.  Pick the skin with the most good matches.
      ORB is scale- and rotation-invariant, so it handles zoom mismatches
      that would fool plain NCC/template matching.

Requires the cache built by tools.rebuild_cache:
    python -m tools.rebuild_cache    # run once (downloads skins + builds index)
    python match_start.py <screenshot>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from champions import Champion, ChampionRoster


# ── cache paths ───────────────────────────────────────────────────────────────

CACHE_DIR    = Path(__file__).parent / "cache"
THUMB_MATRIX = CACHE_DIR / "thumb_matrix.npy"
HIST_MATRIX  = CACHE_DIR / "hist_matrix.npy"
THUMB_INDEX  = CACHE_DIR / "thumb_index.json"
THUMB_SIZE   = (128, 128)


# ── Fixed layout mask ─────────────────────────────────────────────────────────
_REF_W, _REF_H = 2048, 1152

_COL_ART = [
    ((244 + 14)  / _REF_W,  (244 + 287 - 14) / _REF_W),
    ((560 + 14)  / _REF_W,  (560 + 287 - 14) / _REF_W),
    ((876 + 14)  / _REF_W,  (876 + 287 - 14) / _REF_W),
    ((1191 + 14) / _REF_W,  (1191 + 287 - 14) / _REF_W),
    ((1507 + 14) / _REF_W,  (1507 + 287 - 14) / _REF_W),
]

_TOP_ART  = ((60 + 16)        / _REF_H,  (60 + 509 - 150)  / _REF_H)
_BOT_ART  = ((613 + 16)       / _REF_H,  (613 + 509 - 150) / _REF_H)
_TOP_NAME = ((60 + 509 - 150) / _REF_H,  (60 + 509)        / _REF_H)
_BOT_NAME = ((613 + 509 - 150)/ _REF_H,  (613 + 509)       / _REF_H)


def _grid(img: Image.Image):
    w, h      = img.size
    top_y0    = int(h * _TOP_ART[0])
    top_y1    = int(h * _TOP_ART[1])
    bot_y0    = int(h * _BOT_ART[0])
    bot_y1    = int(h * _BOT_ART[1])
    col_segs  = [(int(w * x0), int(w * x1)) for x0, x1 in _COL_ART]
    return top_y0, top_y1, bot_y0, bot_y1, col_segs


# ── gold-pixel helpers ────────────────────────────────────────────────────────

_G_R     = (170, 255)
_G_G     = (130, 215)
_G_B     = ( 20, 130)
_G_DIFF  = 40
_MIN_PIX = 20


def _gold_mask(arr: np.ndarray) -> np.ndarray:
    r = arr[:, :, 0].astype(int)
    g = arr[:, :, 1].astype(int)
    b = arr[:, :, 2].astype(int)
    return (
        (r >= _G_R[0]) & (r <= _G_R[1]) &
        (g >= _G_G[0]) & (g <= _G_G[1]) &
        (b >= _G_B[0]) & (b <= _G_B[1]) &
        ((r - b) >= _G_DIFF)
    )


def _name_strip_ys(h: int, row: int) -> tuple[int, int]:
    frac = _TOP_NAME if row == 0 else _BOT_NAME
    return int(h * frac[0]), min(int(h * frac[1]), h)


def find_player_row(screenshot: Image.Image) -> int | None:
    """Determine which row (0=top/blue, 1=bottom/red) the player is in.

    The player's own card has a distinctly bright gold name-strip highlight
    compared to teammates.  We measure the within-row standout:
        standout = max_col_gold - mean_of_other_4_cols
    The row with the larger standout contains the player.
    """
    arr  = np.array(screenshot)
    h, w = arr.shape[:2]
    gold = _gold_mask(arr)
    col_segs = [(int(w * x0), int(w * x1)) for x0, x1 in _COL_ART]

    def _counts(row: int) -> list[int]:
        y0, y1 = _name_strip_ys(h, row)
        strip  = gold[y0:y1, :]
        return [int(strip[:, x0:x1].sum()) for x0, x1 in col_segs]

    def _standout(counts: list[int]) -> float:
        """Max minus mean of the remaining four columns."""
        mx  = max(counts)
        idx = counts.index(mx)
        others = [v for i, v in enumerate(counts) if i != idx]
        return mx - (sum(others) / max(len(others), 1))

    top_counts = _counts(0)
    bot_counts = _counts(1)
    top_standout = _standout(top_counts)
    bot_standout = _standout(bot_counts)

    # print(f"  gold counts  top={top_counts}  bot={bot_counts}")
    # print(f"  standout     top={top_standout:.0f}  bot={bot_standout:.0f}")

    if max(top_standout, bot_standout) < _MIN_PIX:
        return None
    return 0 if top_standout >= bot_standout else 1


def find_player_col(screenshot: Image.Image, row: int) -> int:
    arr  = np.array(screenshot)
    h, w = arr.shape[:2]
    gold = _gold_mask(arr)
    col_segs = [(int(w * x0), int(w * x1)) for x0, x1 in _COL_ART]
    y0, y1   = _name_strip_ys(h, row)
    strip    = gold[y0:y1, :]
    counts   = [int(strip[:, x0:x1].sum()) for x0, x1 in col_segs]
    return int(np.argmax(counts))


# ── identification constants ──────────────────────────────────────────────────
#
# 2-stage pipeline: HSV histogram shortlist → ORB verification.
#
# Stage 1 — HSV histogram (Bhattacharyya distance, top-K candidates)
#   Scale- and crop-invariant: same champion = same dominant colors.
#   Bhattacharyya dist(query, correct) ≈ 0.19 vs wrong champions ≈ 0.7+.
#
# Stage 2 — ORB (scale-invariant feature matching)
#   Handles zoom mismatch between YOLO crop and reference portrait.
#   Both images are resized to _ORB_SIZE before feature detection.
#
_BORDER_FRAC   = 0.04   # trim 4 % each edge (gold card frame)
_DD_TOP_H      = 480    # top rows of loading portrait visible in card (legacy name)
_HIST_BINS     = [16, 8, 8]      # H × S × V bins  (1024-dim vector)
_ORB_FEATURES  = 1000
_ORB_SIZE      = (220, 384)
_HIST_TOP_K    = 25              # histogram candidates forwarded to ORB


def _prep_query(crop: Image.Image) -> Image.Image:
    """Trim the gold card border from a YOLO loading-screen crop."""
    img = crop.convert("RGB")
    w, h = img.size
    px = max(1, int(w * _BORDER_FRAC))
    py = max(1, int(h * _BORDER_FRAC))
    return img.crop((px, py, w - px, h - py))


def _prep_ref(img: Image.Image) -> Image.Image:
    """Take top _DD_TOP_H rows + trim border from a reference loading portrait."""
    img = img.convert("RGB")
    w, h = img.size
    if h > _DD_TOP_H:
        img = img.crop((0, 0, w, _DD_TOP_H))
    w2, h2 = img.size
    px = max(1, int(w2 * _BORDER_FRAC))
    py = max(1, int(h2 * _BORDER_FRAC))
    return img.crop((px, py, w2 - px, h2 - py))


def _hsv_hist(img: Image.Image) -> np.ndarray:
    """L1-normalised 3-D HSV histogram (1024-dim float32, sum = 1)."""
    bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h   = cv2.calcHist([hsv], [0, 1, 2], None, _HIST_BINS, [0, 180, 0, 256, 0, 256])
    cv2.normalize(h, h, norm_type=cv2.NORM_L1)
    return h.flatten().astype(np.float32)


def _orb_score(query_gray: np.ndarray, ref_gray: np.ndarray) -> int:
    """Number of cross-checked ORB matches between two grayscale arrays."""
    orb = cv2.ORB_create(nfeatures=_ORB_FEATURES, scaleFactor=1.2, nlevels=8)
    kp1, des1 = orb.detectAndCompute(query_gray, None)
    kp2, des2 = orb.detectAndCompute(ref_gray,   None)
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    return len(bf.match(des1, des2))


# ── skin database ─────────────────────────────────────────────────────────────

class SkinDatabase:
    """
    2-stage champion identification:
      Stage 1 — HSV histogram shortlist  (Bhattacharyya distance, top-K)
      Stage 2 — ORB verification         (scale-invariant feature matching)
    """

    def __init__(self):
        if not THUMB_INDEX.exists():
            # print("Cache not found — run  python rebuild_cache.py  first.")
            sys.exit(1)
        # print("Loading skin database…", end=" ", flush=True)
        with open(THUMB_INDEX, encoding="utf-8") as f:
            raw = json.load(f)
        self._keys   = [e["key"]  for e in raw]
        self._names  = [e["name"] for e in raw]
        self._files  = [CACHE_DIR / e.get("file", f"{e['key']}_0.jpg") for e in raw]

        # Load precomputed HSV histogram matrix (N × 1024)
        if HIST_MATRIX.exists():
            self._hists: np.ndarray | None = np.load(HIST_MATRIX)
        else:
            self._hists = None
            # print("(hist_matrix.npy missing — run rebuild_cache.py)", end=" ")

        # print(f"{len(self._keys)} skins ready.")

    # ── stage 1: histogram shortlist ───────────────────────────────────────

    def _shortlist(self, crop: Image.Image, k: int = _HIST_TOP_K) -> list[int]:
        """Return indices of the top-k skins by HSV histogram similarity."""
        q_hist = _hsv_hist(crop)       # use full crop (no border trim needed)

        if self._hists is not None:
            # Bhattacharyya distance: lower = more similar
            # Vectorised: sqrt( sum(sqrt(q * r)) ) per row
            dists = 1.0 - np.sum(np.sqrt(self._hists * q_hist[np.newaxis, :]),
                                  axis=1)
            return list(np.argsort(dists)[:k])

        # Fallback: if matrix missing, return first k entries
        return list(range(min(k, len(self._keys))))

    # ── stage 2: ORB verification ───────────────────────────────────────────

    def _query_gray(self, crop: Image.Image) -> np.ndarray:
        prepped = _prep_query(crop).convert("L")
        return cv2.resize(np.array(prepped), _ORB_SIZE, interpolation=cv2.INTER_AREA)

    def _ref_gray(self, path: Path) -> np.ndarray:
        img = Image.open(path)
        prepped = _prep_ref(img).convert("L")
        return cv2.resize(np.array(prepped), _ORB_SIZE, interpolation=cv2.INTER_AREA)

    def identify(self, crop: Image.Image, debug: bool = False) -> tuple[str, str]:
        shortlist = self._shortlist(crop)

        if debug:
            pass  # print(f"    hist top-5: " + "  ".join(self._keys[i] for i in shortlist[:5]))

        q_gray     = self._query_gray(crop)
        best_key   = self._keys[shortlist[0]]
        best_name  = self._names[shortlist[0]]
        best_score = -1

        for idx in shortlist:
            try:
                r_gray = self._ref_gray(self._files[idx])
                score  = _orb_score(q_gray, r_gray)
            except Exception:
                score = 0

            if debug:
                pass  # print(f"      ORB {self._keys[idx]:<16}: {score:3d} matches")

            if score > best_score:
                best_score = score
                best_key   = self._keys[idx]
                best_name  = self._names[idx]

        return best_key, best_name


# ── card cropping ─────────────────────────────────────────────────────────────

def crop_card(screenshot: Image.Image, row: int, col: int) -> Image.Image:
    top_y0, top_y1, bot_y0, bot_y1, col_segs = _grid(screenshot)
    y0 = top_y0 if row == 0 else bot_y0
    y1 = top_y1 if row == 0 else bot_y1
    x0, x1 = col_segs[col]
    return screenshot.crop((x0, y0, x1, y1))


# ── YOLO-based card location ──────────────────────────────────────────────────
# Shared NMS + filter so both match_start._run_splash_yolo and backend/onnx
# detect_cards use the same logic (build uses onnx, so it must go through this).

def process_splash_detections(
    splash_dets: list[tuple[float, tuple[int, int, int, int]]],
    name_dets: list[tuple[float, tuple[int, int, int, int]]],
    img_w: int,
    img_h: int,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """NMS (iou 0.10) + plausible-splash filter. Return (splash_boxes, name_boxes) top-10 each."""
    def _iou(a, b):
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter)

    def _nms_dets(dets, iou_thr=0.10):
        ordered = sorted(dets, key=lambda d: -d[0])
        kept = []
        for det in ordered:
            if all(_iou(det[1], k[1]) < iou_thr for k in kept):
                kept.append(det)
        return kept

    def _is_plausible_splash_box(box, w: int, h: int) -> bool:
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            return False
        if bh / bw < 0.85 or bh / bw > 2.5:
            return False
        if (bw * bh) / (w * h) < 0.012:
            return False
        return True

    def _top10(dets, apply_splash_filter: bool):
        nms = _nms_dets(dets)
        if apply_splash_filter:
            nms = [(c, b) for c, b in nms if _is_plausible_splash_box(b, img_w, img_h)]
        top = nms[:10]
        return sorted([b for _, b in top], key=lambda b: (round(b[1] / 200), b[0]))

    return _top10(splash_dets, True), _top10(name_dets, False)


def _run_splash_yolo(screenshot: Image.Image):
    """Run the splash model and return top-10 per class sorted by confidence.

    Returns (splash_boxes, name_boxes) where each is a list of
    (x1, y1, x2, y2) sorted left→right within each row.
    Returns (None, None) if the model is unavailable.
    """
    try:
        from splash_model import load_model as load_splash
        import numpy as _np
    except ImportError:
        return None, None

    try:
        model  = load_splash()
        arr    = _np.array(screenshot.convert("RGB"))
        results = model(arr, imgsz=1920, conf=0.35, max_det=30,
                        verbose=False, half=False)
    except Exception:
        return None, None

    splash_det, name_det = [], []   # (conf, box)
    for r in results:
        if r.boxes is None:
            continue
        for box, cls, conf in zip(r.boxes.xyxy.cpu().numpy(),
                                   r.boxes.cls.cpu().numpy(),
                                   r.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box[:4])
            b = (x1, y1, x2, y2)
            if int(cls) == 0:
                splash_det.append((float(conf), b))
            elif int(cls) == 3:
                name_det.append((float(conf), b))

    img_h, img_w = arr.shape[:2]
    return process_splash_detections(splash_det, name_det, img_w, img_h)


def _boxes_to_grid(
    boxes: list[tuple[int,int,int,int]],
    img_h: int,
) -> tuple[list[tuple[int,int,int,int]], list[tuple[int,int,int,int]]]:
    mid = img_h // 2
    top = sorted([b for b in boxes if (b[1]+b[3])//2 < mid], key=lambda b: b[0])
    bot = sorted([b for b in boxes if (b[1]+b[3])//2 >= mid], key=lambda b: b[0])
    return top, bot


def _gold_score(arr: np.ndarray) -> float:
    """Count gold/orange pixels in a cropped region (player name highlight)."""
    r = arr[:, :, 0].astype(int)
    g = arr[:, :, 1].astype(int)
    b = arr[:, :, 2].astype(int)
    mask = (
        (r >= 170) & (r <= 255) &
        (g >= 120) & (g <= 215) &
        (b >=   5) & (b <= 130) &
        ((r - b) >= 40)
    )
    return float(mask.sum())


def _find_player_from_name_boxes(
    screenshot: Image.Image,
    name_boxes: list[tuple[int,int,int,int]],
    img_h: int,
) -> tuple[int, int] | tuple[None, None]:
    """Find which name box is the player's (most gold pixels).

    Returns (player_row, player_col) where row 0=top/blue, 1=bottom/red.
    """
    arr  = np.array(screenshot)
    mid  = img_h // 2

    top_names = sorted([b for b in name_boxes if (b[1]+b[3])//2 < mid],
                       key=lambda b: b[0])
    bot_names = sorted([b for b in name_boxes if (b[1]+b[3])//2 >= mid],
                       key=lambda b: b[0])

    best_score = -1
    player_row = player_col = None

    for row_idx, row_names in enumerate([top_names, bot_names]):
        for col_idx, (x1, y1, x2, y2) in enumerate(row_names):
            crop  = arr[y1:y2, x1:x2]
            score = _gold_score(crop)
            # print(f"    name[{row_idx}][{col_idx}] gold={score:.0f}")
            if score > best_score:
                best_score = score
                player_row = row_idx
                player_col = col_idx

    return player_row, player_col


# ── debug composite ───────────────────────────────────────────────────────────

def _save_debug_composite(
    crops_top: list[tuple[Image.Image, str]],
    crops_bot: list[tuple[Image.Image, str]],
    out_path: Path,
) -> None:
    try:
        from PIL import ImageDraw, ImageFont
        CELL_W, CELL_H = 160, 240
        PAD            = 4
        ROWS, COLS     = 2, 5
        W = COLS * (CELL_W + PAD) + PAD
        H = ROWS * (CELL_H + PAD + 20) + PAD

        canvas = Image.new("RGB", (W, H), (30, 30, 46))
        draw   = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype("arial.ttf", 11)
        except Exception:
            font = ImageFont.load_default()

        for row_idx, row_crops in enumerate([crops_top, crops_bot]):
            for col_idx, (crop, label) in enumerate(row_crops):
                x = PAD + col_idx * (CELL_W + PAD)
                y = PAD + row_idx * (CELL_H + PAD + 20)
                thumb = crop.resize((CELL_W, CELL_H), Image.LANCZOS)
                canvas.paste(thumb, (x, y))
                draw.text((x + 2, y + CELL_H + 2), label,
                          fill=(200, 200, 200), font=font)

        canvas.save(str(out_path))
        # print(f"  Debug composite → {out_path}")
    except Exception as e:
        pass  # print(f"  Debug composite failed: {e}")


# ── main analysis ─────────────────────────────────────────────────────────────

def identify_all(
    screenshot:   Image.Image,
    db:           SkinDatabase,
    splash_boxes: list | None = None,
    name_boxes:   list | None = None,
) -> ChampionRoster | None:
    w, h      = screenshot.size
    debug_dir = Path(__file__).parent / "debug_crops"
    debug_dir.mkdir(exist_ok=True)

    def _crop_box(box):
        x1, y1, x2, y2 = box
        return screenshot.crop((x1, y1, x2, y2))

    def _identify_box(box, label) -> tuple[str, str]:
        crop = _crop_box(box)
        crop.save(debug_dir / f"{label}.png")
        # print(f"  [{label}]")
        return db.identify(crop)

    # ── YOLO path ─────────────────────────────────────────────────────────────
    # Use pre-computed boxes from the watcher loop when available (avoids
    # running the splash model a second time and uses the correct class mapping).
    if splash_boxes is None or name_boxes is None:
        splash_boxes, name_boxes = _run_splash_yolo(screenshot)

    if splash_boxes and len(splash_boxes) == 10:
        # print(f"  YOLO: {len(splash_boxes)} splash  {len(name_boxes or [])} names")

        top_splash, bot_splash = _boxes_to_grid(splash_boxes, h)

        # Determine player row/col from gold highlight on summoner name boxes
        player_row = player_col = None
        if name_boxes and len(name_boxes) >= 5:
            player_row, player_col = _find_player_from_name_boxes(
                screenshot, name_boxes, h)

        # Fallback: pixel scanning
        if player_row is None:
            player_row = find_player_row(screenshot)
            player_col = find_player_col(screenshot, player_row) if player_row is not None else 0

        if player_row is None:
            # print("  Could not determine player side.")
            return None

        side = "blue" if player_row == 0 else "red"
        # print(f"  Side: {side.upper()}  |  player col {player_col + 1}")

        ally_boxes  = top_splash if player_row == 0 else bot_splash
        enemy_boxes = bot_splash if player_row == 0 else top_splash

        if len(ally_boxes) < 5 or len(enemy_boxes) < 5:
            pass  # print("  Incomplete rows — falling back to fixed grid.")
        else:
            p_key, p_name = _identify_box(ally_boxes[player_col], "player")
            player = Champion(name=p_name, key=p_key, is_player=True)

            ally_labels, allies, ai = [], [], 0
            for ci, box in enumerate(ally_boxes):
                if ci == player_col:
                    ally_labels.append((_crop_box(box), f"{p_name} ★"))
                    continue
                key, name = _identify_box(box, f"ally_{ai}")
                allies.append(Champion(name=name, key=key))
                ally_labels.append((_crop_box(box), f"{name}"))
                ai += 1

            enemy_labels, enemies = [], []
            for ci, box in enumerate(enemy_boxes):
                key, name = _identify_box(box, f"enemy_{ci}")
                enemies.append(Champion(name=name, key=key))
                enemy_labels.append((_crop_box(box), f"{name}"))

            top_l = ally_labels  if player_row == 0 else enemy_labels
            bot_l = enemy_labels if player_row == 0 else ally_labels
            _save_debug_composite(top_l, bot_l, debug_dir / "debug_composite.jpg")
            enemy_side = "red" if side == "blue" else "blue"
            return ChampionRoster(player=player, allies=allies, enemies=enemies,
                                  enemy_side=enemy_side)

    # ── Fallback: fixed proportional grid ─────────────────────────────────────
    # if splash_boxes:
    #     print(f"  YOLO found {len(splash_boxes)} cards (expected 10) — using fixed grid.")
    # else:
    #     print("  Splash model unavailable — using fixed grid.")

    player_row = find_player_row(screenshot)
    if player_row is None:
        return None
    player_col = find_player_col(screenshot, player_row)
    enemy_row  = 1 - player_row
    side       = "blue" if player_row == 0 else "red"
    # print(f"  Side: {side.upper()}  |  player col {player_col + 1}")

    def _identify_grid(row: int, col: int, label: str) -> tuple[str, str]:
        crop = crop_card(screenshot, row, col)
        crop.save(debug_dir / f"{label}.png")
        return db.identify(crop)

    p_key, p_name = _identify_grid(player_row, player_col, "player")
    player = Champion(name=p_name, key=p_key, is_player=True)

    ally_labels, allies, ai = [], [], 0
    for col in range(5):
        crop = crop_card(screenshot, player_row, col)
        if col == player_col:
            ally_labels.append((crop, f"{p_name} ★"))
            continue
        key, name = _identify_grid(player_row, col, f"ally_{ai}")
        allies.append(Champion(name=name, key=key))
        ally_labels.append((crop, f"{name}"))
        ai += 1

    enemy_labels, enemies = [], []
    for col in range(5):
        crop = crop_card(screenshot, enemy_row, col)
        key, name = _identify_grid(enemy_row, col, f"enemy_{col}")
        enemies.append(Champion(name=name, key=key))
        enemy_labels.append((crop, f"{name}"))

    top_l = ally_labels  if player_row == 0 else enemy_labels
    bot_l = enemy_labels if player_row == 0 else ally_labels
    _save_debug_composite(top_l, bot_l, debug_dir / "debug_composite.jpg")
    return ChampionRoster(player=player, allies=allies, enemies=enemies)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python match_start.py <screenshot>")
        sys.exit(1)

    path = " ".join(sys.argv[1:])
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        print(f"Could not open image: {e}")
        sys.exit(1)

    db     = SkinDatabase()
    roster = identify_all(img, db)

    if roster is None:
        print("Could not detect player side — is this a loading screen?")
        sys.exit(1)

    print()
    print(roster)
