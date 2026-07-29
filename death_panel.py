"""
death_panel.py — Detect dead enemies from the HUD death strip.

The death strip is the horizontal band of grayscale champion portraits
that appears above the minimap when enemies are dead.  It is a separate
screen region from the minimap and must be captured independently.

Pipeline per frame
──────────────────
1. Convert strip to HSV.
2. Scan for icon-shaped blobs with low saturation (dead = grayscale icon).
3. Histogram-correlate each blob against pre-built grayscale reference histograms.
4. Update library._state[key].dead in-place and return a
   {key: respawn_secs | None} dict for the caller to use.

Region auto-detection (relative to minimap region x, y, w, h)
──────────────────────────────────────────────────────────────
  panel_h  = h * _PANEL_H_FRAC   (~22 % of minimap height)
  panel_y  = y - h * _PANEL_Y_FRAC  (~60 % above minimap top)
  panel_x  = x,  panel_w = w
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from tracker import ChampionLibrary

# ── Tuning constants ──────────────────────────────────────────────────────────

_DEAD_CONFIRM_MIN = 2     # consecutive matched frames before committing dead
_DEAD_CLEAR_MIN   = 2     # consecutive absent frames before confirming respawn
_TMPL_MATCH_MIN   = 0.4  # NCC score to accept an identity (spatial match)

# Circle detection geometry
_CIRC_R_FRAC_LO  = 0.28  # expected radius >= panel_h * this fraction
_CIRC_R_FRAC_HI  = 0.50  # expected radius <= panel_h * this fraction
_HOUGH_PARAM2    = 15    # HoughCircles accumulator votes (lower -> more detections)
_MIN_CIRCULARITY = 0.50  # 4*pi*A/P^2 threshold for contour fallback (circle = 1.0)

# Region position relative to minimap (x, y, w, h)
_PANEL_H_FRAC    = 0.21  # panel height as fraction of minimap height
_PANEL_Y_FRAC    = 0.55  # top of strip = minimap_y - minimap_h * this

# Red text bounds for timer digits (enemy death strip)
_TIMER_R_DIFF    = 55    # R - max(G, B) must exceed this to count as red
_TIMER_R_MIN     = 110
_TIMER_B_MAX     = 110
_TIMER_MIN_PIX   = 8     # minimum red pixels needed to even try OCR

# Player portrait respawn timer (~#FF0000 on grayscale icon when dead)
_PLAYER_DEATH_RED_DIFF = 55
_PLAYER_DEATH_RED_R_MIN  = 175
_PLAYER_DEATH_RED_G_MAX  = 95
_PLAYER_DEATH_RED_B_MAX  = 95
_PLAYER_DEATH_RED_MIN_PIX = 10
_PLAYER_DEATH_RED_BAND_FRAC = 0.40   # search lower 40% of portrait crop
_PLAYER_DEAD_CONFIRM_MIN = 3   # consecutive red-timer frames before declaring dead
_PLAYER_DEAD_CLEAR_MIN   = 2   # consecutive non-red frames before declaring alive

# Stack-snap distance limit: snap only if the candidate is within this many
# icon-widths of the player's last known position.  Prevents wild jumps to
# champs on the other side of the map.
_SNAP_MAX_ICON_RADII = 1.0


# ── Region helper ─────────────────────────────────────────────────────────────

def death_panel_region(minimap_region: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """
    Return the (x, y, w, h) capture region for the dead-champion portrait strip
    based on the minimap region.  Clamps to non-negative coordinates.
    """
    x, y, w, h = minimap_region
    panel_h = max(20, int(h * _PANEL_H_FRAC))
    panel_y = max(0,  y - int(h * _PANEL_Y_FRAC))
    return (x, panel_y, w, panel_h)


def death_panel_slot_region(
    panel_region: tuple[int, int, int, int],
    col: int,
    row: int,
) -> tuple[int, int, int, int]:
    """
    One cell of the death-panel grid: panel width/height each split into 4 slots.
    col, row are 0-based (0..3), clamped to the panel bounds.
    """
    px, py, pw, ph = panel_region
    slot_w = max(8, pw // 4)
    slot_h = max(8, ph // 4)
    c = max(0, min(3, col))
    r = max(0, min(3, row))
    return (px + c * slot_w, py + r * slot_h, slot_w, slot_h)


# Local-player death portrait — measured on 532×248 shot vs top death strip row only.
# Strip row ≈ y 10–110 (h=100); player box ≈ (15,125)–(125,235), 15px below strip.
# In-game the player icon sits one portrait-width left of the reference x=15 mark.
_STRIP_REF_W = 532
_STRIP_REF_H = 100
_PLAYER_W_FRAC = 110 / _STRIP_REF_W
_PLAYER_X_FRAC = (15 - 110) / _STRIP_REF_W
_PLAYER_GAP_FRAC = 15 / _STRIP_REF_H
_PLAYER_H_FRAC = 110 / _STRIP_REF_H
_PLAYER_NUDGE_FRAC = 0.10  # fine-tune: right + down vs portrait size


def _player_portrait_box(
    origin_x: int, origin_y: int, pw: int, ph: int,
) -> tuple[int, int, int, int]:
    """(x, y, w, h) from a top-left origin and strip-sized pw×ph."""
    w = int(pw * _PLAYER_W_FRAC)
    h = int(ph * _PLAYER_H_FRAC)
    x0 = origin_x + int(pw * _PLAYER_X_FRAC) + int(w * _PLAYER_NUDGE_FRAC)
    y0 = origin_y + ph + int(ph * _PLAYER_GAP_FRAC) + int(h * _PLAYER_NUDGE_FRAC)
    return (x0, y0, w, h)


def player_death_portrait_region(
    minimap_region: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """
    Screen (x, y, w, h) for the local player's death portrait.

    Anchored to ``death_panel_region`` (top strip only): same left edge and
    width as the strip capture, offset down by the gap measured in the reference.
    """
    px, py, pw, ph = death_panel_region(minimap_region)
    return _player_portrait_box(px, py, pw, ph)


def player_death_portrait_slot_local(
    panel_region: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Player portrait rect in coords where the death-strip crop is (0, 0, pw, ph)."""
    _px, _py, pw, ph = panel_region
    return _player_portrait_box(0, 0, pw, ph)


def save_player_death_portrait_debug(
    strip_frame_rgb: np.ndarray,
    player_crop_rgb: np.ndarray | None = None,
    panel_region: tuple[int, int, int, int] | None = None,
    path: str = "debug_crops/player_death_portrait_slot.png",
) -> bool:
    """
    Write player portrait crop + death-strip panel with slot outline.

    The player portrait sits below the death-strip capture, so pass
    ``player_crop_rgb`` from a dedicated ``player_death_portrait_region`` grab.
    """
    return  # debug_crops disabled
    if strip_frame_rgb is None or strip_frame_rgb.size == 0:
        return False
    if player_crop_rgb is None or player_crop_rgb.size == 0:
        return False
    try:
        sh, sw = strip_frame_rgb.shape[:2]
        if panel_region is None:
            panel_region = (0, 0, sw, sh)
        lx, ly, lw, lh = player_death_portrait_slot_local(panel_region)
        pad_h = max(0, ly + lh - sh)
        canvas = strip_frame_rgb
        if pad_h > 0:
            pad = np.zeros((pad_h, sw, 3), dtype=strip_frame_rgb.dtype)
            canvas = np.vstack([strip_frame_rgb, pad])
        out_dir = os.path.dirname(path) or "."
        os.makedirs(out_dir, exist_ok=True)
        stem = path.rsplit(".", 1)[0] if "." in path else path
        panel_path = f"{stem}_panel.png"
        dbg = cv2.cvtColor(canvas.copy(), cv2.COLOR_RGB2BGR)
        x2, y2 = lx + lw, ly + lh
        cv2.rectangle(dbg, (lx, ly), (x2 - 1, y2 - 1), (0, 255, 255), 2)
        cv2.putText(
            dbg,
            f"player slot ({lx},{ly})-({x2},{y2})",
            (4, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        ok_crop = cv2.imwrite(
            path, cv2.cvtColor(player_crop_rgb, cv2.COLOR_RGB2BGR))
        ok_panel = cv2.imwrite(panel_path, dbg)
        return bool(ok_crop and ok_panel)
    except Exception:
        return False


# ── Template reference cache & combined scorer ────────────────────────────────

_TMPL_SIZE = 64   # px — reference and query resized to this before scoring

# Blending weights  (must sum to 1.0)
_W_GRAY  = 0.55
_W_EDGE  = 0.30
_W_ORB   = 0.15

# ORB tuning
_ORB_FEATURES    = 200
_ORB_GOOD_DIST   = 64   # Hamming distance threshold for a "good" match

# Module-level ORB detector and brute-force matcher (created once)
_ORB_DET = cv2.ORB_create(nfeatures=_ORB_FEATURES)
_BF      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def _ensure_tmpl_refs(library: "ChampionLibrary") -> None:
    """
    Build per-champion reference features (gray, edges, ORB) once per library.
    Stored as library._tmpl_refs = {key: {"gray", "edges", "orb_des", "n_kp"}}
    """
    if hasattr(library, "_tmpl_refs"):
        return
    refs: dict[str, dict | None] = {}
    for ci, c in enumerate(library.all):
        icon = library.icon_imgs.get(c.key)
        if icon is None:
            refs[c.key] = None
            continue
        bgr     = icon[:, :, ::-1]
        resized = cv2.resize(bgr, (_TMPL_SIZE, _TMPL_SIZE))
        gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        edges   = cv2.Canny(gray, 40, 120)
        kp, des = _ORB_DET.detectAndCompute(gray, None)
        refs[c.key] = {
            "gray":    gray,
            "edges":   edges,
            "orb_des": des,
            "n_kp":    len(kp),
        }
    library._tmpl_refs = refs  # type: ignore[attr-defined]


# ── Per-blob feature extraction ───────────────────────────────────────────────

def _extract_features(crop_rgb: np.ndarray) -> dict:
    """Return gray, edges, and ORB descriptors for a single detected orb crop."""
    resized = cv2.resize(crop_rgb, (_TMPL_SIZE, _TMPL_SIZE))
    gray    = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    edges   = cv2.Canny(gray, 40, 120)
    kp, des = _ORB_DET.detectAndCompute(gray, None)
    return {"gray": gray, "edges": edges, "orb_des": des, "n_kp": len(kp)}


# ── Combined scoring ──────────────────────────────────────────────────────────

def _combined_score(query: dict, ref: dict) -> float:
    """
    score = W_GRAY * ncc_gray + W_EDGE * ncc_edges + W_ORB * orb_score

    Each component is clamped to [0, 1] before weighting.
    """
    def _ncc(a: np.ndarray, b: np.ndarray) -> float:
        try:
            res = cv2.matchTemplate(a.astype(np.float32),
                                    b.astype(np.float32),
                                    cv2.TM_CCOEFF_NORMED)
            return max(0.0, float(res[0, 0]))
        except Exception:
            return 0.0

    ncc_gray  = _ncc(query["gray"],  ref["gray"])
    ncc_edges = _ncc(query["edges"], ref["edges"])

    orb_score = 0.0
    q_des = query.get("orb_des")
    r_des = ref.get("orb_des")
    if q_des is not None and r_des is not None and len(q_des) > 0 and len(r_des) > 0:
        try:
            matches  = _BF.match(q_des, r_des)
            good     = [m for m in matches if m.distance < _ORB_GOOD_DIST]
            denom    = min(query["n_kp"], ref["n_kp"])
            orb_score = min(1.0, len(good) / max(denom, 1))
        except Exception:
            pass

    return _W_GRAY * ncc_gray + _W_EDGE * ncc_edges + _W_ORB * orb_score


# ── Circle detection ──────────────────────────────────────────────────────────

def _nms_circles(circles: list[tuple[int, int, int]],
                 overlap_thresh: float = 0.5) -> list[tuple[int, int, int]]:
    """
    Remove duplicate circles whose centres are closer than overlap_thresh × 2r.
    Keep the one with the largest radius when two overlap.
    """
    if not circles:
        return []
    # sort by radius descending so we always keep the "best" one
    cs = sorted(circles, key=lambda c: c[2], reverse=True)
    kept: list[tuple[int, int, int]] = []
    for cx, cy, r in cs:
        for kx, ky, kr in kept:
            dist = ((cx - kx) ** 2 + (cy - ky) ** 2) ** 0.5
            if dist < overlap_thresh * (r + kr):
                break
        else:
            kept.append((cx, cy, r))
    return kept


def _find_icon_boxes(frame_rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Return (x1, y1, x2, y2) bounding boxes derived from circular dead-portrait
    blobs, sorted left -> right.

    Stage 1 — HoughCircles on a Gaussian-blurred grey image.
    Stage 2 — circularity-filtered contours on the desaturated mask (fallback /
              supplement when Hough finds nothing or misses icons).
    Both sets are merged and de-duplicated via NMS.
    """
    h, w = frame_rgb.shape[:2]

    r_min = max(6,  int(h * _CIRC_R_FRAC_LO))
    r_max = max(10, int(h * _CIRC_R_FRAC_HI))
    min_area = np.pi * r_min ** 2 * 0.5  # half of smallest expected circle area

    gray    = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

    raw_circles: list[tuple[int, int, int]] = []

    # ── Stage 1: HoughCircles ────────────────────────────────────────────────
    hough = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=max(r_min * 2, 12),  # portraits don't overlap
        param1=60,                    # Canny high threshold
        param2=_HOUGH_PARAM2,         # accumulator votes
        minRadius=r_min,
        maxRadius=r_max,
    )
    if hough is not None:
        for cx, cy, r in np.round(hough[0]).astype(int):
            raw_circles.append((int(cx), int(cy), int(r)))

    # ── Stage 2: Circularity-filtered contours ───────────────────────────────
    # Build a binary mask of the dark circular rim that frames each portrait.
    # The rim is darker than the portrait face; Canny + morphology exposes it.
    edges = cv2.Canny(blurred, 30, 80)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri < 1:
            continue
        circularity = 4.0 * np.pi * area / (peri * peri)
        if circularity < _MIN_CIRCULARITY:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / max(bh, 1)
        if not (0.55 <= aspect <= 1.8):
            continue
        # size guard: bounding box diameter must be in expected range
        diam = (bw + bh) / 2.0
        if not (r_min * 1.5 <= diam <= r_max * 2.5):
            continue
        cx = bx + bw // 2
        cy = by + bh // 2
        r  = int((bw + bh) / 4)
        raw_circles.append((cx, cy, r))

    # ── Merge & NMS ──────────────────────────────────────────────────────────
    circles = _nms_circles(raw_circles, overlap_thresh=0.6)

    boxes: list[tuple[int, int, int, int]] = []
    for cx, cy, r in circles:
        x1 = max(0, cx - r)
        y1 = max(0, cy - r)
        x2 = min(w, cx + r)
        y2 = min(h, cy + r)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))

    boxes.sort(key=lambda b: b[0])
    return boxes


# ── Debug output ──────────────────────────────────────────────────────────────

_DEBUG_INTERVAL: int = 2
_debug_frame_counter: int = 0


def _save_death_panel_debug(
    frame_rgb: np.ndarray,
    icon_boxes: list[tuple[int, int, int, int]],
    assigned: dict[str, tuple[int, int, int, int]],
    ncc_scores: dict[str, float],
) -> None:
    """Save an annotated death-panel debug image to debug_crops/death_panel.jpg."""
    return  # debug_crops disabled

    try:
        os.makedirs("debug_crops", exist_ok=True)
        # Upscale so small panels are readable
        scale = max(1, 120 // max(frame_rgb.shape[0], 1))
        vis   = cv2.resize(frame_rgb,
                           (frame_rgb.shape[1] * scale, frame_rgb.shape[0] * scale),
                           interpolation=cv2.INTER_NEAREST)
        dbg   = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

        assigned_set = set(map(id, assigned.values()))

        # All detected circles in dim white
        for (x1, y1, x2, y2) in icon_boxes:
            cx = (x1 + x2) // 2 * scale
            cy = (y1 + y2) // 2 * scale
            r  = max(1, (x2 - x1) // 2 * scale)
            cv2.circle(dbg, (cx, cy), r, (180, 180, 180), 1)

        # Matched circles in green with champion name + NCC score
        for key, (x1, y1, x2, y2) in assigned.items():
            score = ncc_scores.get(key, 0.0)
            cx    = (x1 + x2) // 2 * scale
            cy    = (y1 + y2) // 2 * scale
            r     = max(1, (x2 - x1) // 2 * scale)
            cv2.circle(dbg, (cx, cy), r, (0, 220, 60), 2)
            label = f"{key} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            lx = cx - tw // 2
            ly = max(cy - r - 4, th + 2)
            cv2.rectangle(dbg, (lx - 1, ly - th - 2), (lx + tw + 2, ly + 2),
                          (0, 0, 0), cv2.FILLED)
            cv2.putText(dbg, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 80), 1,
                        cv2.LINE_AA)

        # Header
        header = (f"death_panel  circles={len(icon_boxes)}"
                  f"  matched={len(assigned)}"
                  f"  frame={frame_rgb.shape[1]}x{frame_rgb.shape[0]}")
        cv2.putText(dbg, header, (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imwrite("debug_crops/death_panel.jpg", dbg)
    except Exception:
        pass


# ── Player death portrait (below enemy strip) ─────────────────────────────────

_PLAYER_PORTRAIT_TMPL_MIN = 0.38
_PLAYER_PORTRAIT_HSV_MIN  = 0.50
_PLAYER_PORTRAIT_TMPL_SOFT = 0.28


def player_death_timer_red_present(player_frame_rgb: np.ndarray) -> bool:
    """
    True when the respawn-timer red is visible on the player portrait crop.

    Dead portraits are grayscale with a bright red countdown digit along the
    bottom (R ≫ G,B, roughly #FF0000). Alive / off-strip crops lack that red.

    Two conditions must both be true:
    1. Enough pure-red pixels exist in the bottom band (timer digits).
    2. The non-red pixels in that band are predominantly grayscale — the dead
       portrait is desaturated.  An alive champion portrait with red in its art
       (Jinx, Ahri, etc.) fails this check because the rest of the art is
       colourful.
    """
    if player_frame_rgb is None or player_frame_rgb.size == 0:
        return False
    h, w = player_frame_rgb.shape[:2]
    if h < 8 or w < 8:
        return False
    y0 = int(h * (1.0 - _PLAYER_DEATH_RED_BAND_FRAC))
    band = player_frame_rgb[y0:, :]
    r = band[:, :, 0].astype(np.int16)
    g = band[:, :, 1].astype(np.int16)
    b = band[:, :, 2].astype(np.int16)
    diff = r - np.maximum(g, b)
    red_mask = (
        (diff >= _PLAYER_DEATH_RED_DIFF)
        & (r >= _PLAYER_DEATH_RED_R_MIN)
        & (g <= _PLAYER_DEATH_RED_G_MAX)
        & (b <= _PLAYER_DEATH_RED_B_MAX)
    )
    if int(red_mask.sum()) < _PLAYER_DEATH_RED_MIN_PIX:
        return False

    # Gray-background check: non-red pixels must be mostly desaturated.
    # Dead portrait art is monochrome; alive portrait art is full-colour.
    non_red = ~red_mask
    n_non_red = int(non_red.sum())
    if n_non_red > 0:
        nr_f = band[:, :, 0][non_red].astype(np.float32)
        ng_f = band[:, :, 1][non_red].astype(np.float32)
        nb_f = band[:, :, 2][non_red].astype(np.float32)
        mx = np.maximum(np.maximum(nr_f, ng_f), nb_f)
        mn = np.minimum(np.minimum(nr_f, ng_f), nb_f)
        sat = np.where(mx > 0, (mx - mn) / mx, 0.0)
        gray_fraction = float((sat < 0.35).mean())
        if gray_fraction < 0.55:   # less than 55 % gray → alive portrait
            return False

    return True


def player_portrait_match_scores(
    player_frame_rgb: np.ndarray,
    library: "ChampionLibrary",
) -> tuple[float, float]:
    """
    Template (gray/edge/ORB) + HSV Bhattacharyya vs local player icon.
    Returns (template_score, hsv_bc).
    """
    if player_frame_rgb is None or player_frame_rgb.size == 0:
        return 0.0, 0.0

    from tracker import _crop_hsv_hist

    player_key = library.roster.player.key
    _ensure_tmpl_refs(library)
    ref = library._tmpl_refs.get(player_key)  # type: ignore[attr-defined]
    if ref is None:
        tmpl_sc = 0.0
    else:
        feat = _extract_features(player_frame_rgb)
        tmpl_sc = _combined_score(feat, ref)

    h, w = player_frame_rgb.shape[:2]
    sqrt_crop = np.sqrt(_crop_hsv_hist(player_frame_rgb, (0, 0, w, h)))
    hsv_bc = float(library.sqrt_hist[0] @ sqrt_crop)  # player is index 0

    return tmpl_sc, hsv_bc


def scan_player_death_portrait(
    player_frame_rgb: np.ndarray,
    library:   "ChampionLibrary",
    now:       float | None = None,
) -> bool | None:
    """
    Respawn-timer red on the player portrait ⇒ dead; no red ⇒ alive.
    """
    if now is None:
        now = time.perf_counter()
    if player_frame_rgb is None or player_frame_rgb.size == 0:
        return None

    player_key = library.roster.player.key
    st = library._state[player_key]
    timer_red = player_death_timer_red_present(player_frame_rgb)

    if timer_red:
        st.dead_confirms += 1
        st.dead_clears = 0
        if not st.dead and st.dead_confirms >= _PLAYER_DEAD_CONFIRM_MIN:
            st.dead = True
            st.suppress_marker_until_map = True
            st.infer_stack_key = None
    else:
        st.dead_confirms = 0
        if st.dead:
            st.dead_clears += 1
            if st.dead_clears >= _PLAYER_DEAD_CLEAR_MIN:
                st.dead = False
                st.dead_clears = 0

    return timer_red


def apply_player_off_map_fallback(
    library: "ChampionLibrary",
    results: list[dict],
    now: float,
    player_frame_rgb: np.ndarray | None = None,
) -> list[dict]:
    """
    When YOLO does not pick up the player on the minimap:
      - respawn-timer red on death portrait ⇒ dead (no position change)
      - otherwise ⇒ alive; stick player marker to nearest on-map icon (follow it while stacked)
    """
    player_key = library.roster.player.key
    st = library._state[player_key]

    if any(
        r.get("key") == player_key and not r.get("inferred_from")
        for r in results
    ):
        st.infer_stack_key = None
        return results

    if player_frame_rgb is not None and player_frame_rgb.size > 0:
        scan_player_death_portrait(player_frame_rgb, library, now=now)

    if st.dead or st.suppress_marker_until_map:
        return results

    if st.pos is None:
        return results

    others = [r for r in results if r.get("key") != player_key]
    if not others:
        if st.infer_stack_key and st.pos is not None:
            library._update(player_key, st.pos, 0.0, now)
        return results

    ax, ay = st.pos

    def _dist(r: dict) -> float:
        ex, ey = r["location"]
        return ((ex - ax) ** 2 + (ey - ay) ** 2) ** 0.5

    def _icon_w(r: dict) -> float:
        bx1, by1, bx2, by2 = r.get("box", (0, 0, 0, 0))
        return max(8.0, float(bx2 - bx1))

    best: dict | None = None
    if st.infer_stack_key:
        for r in others:
            if r.get("key") == st.infer_stack_key:
                best = r
                break
    if best is None:
        best = min(others, key=_dist)

    # Reject the snap if the candidate is more than _SNAP_MAX_ICON_RADII icon
    # widths away from the player's last known position — no viable stacking
    # candidate nearby.  Suppress the marker until YOLO sees the player again.
    if _dist(best) > _icon_w(best) * _SNAP_MAX_ICON_RADII:
        st.infer_stack_key = None
        st.suppress_marker_until_map = True
        return results

    st.infer_stack_key = best["key"]
    cx, cy = best["location"]
    bx1, by1, bx2, by2 = best.get("box", (cx - 8, cy - 8, cx + 8, cy + 8))
    hw = max(6, (bx2 - bx1) // 4)
    hh = max(6, (by2 - by1) // 4)
    box = (cx - hw, cy - hh, cx + hw, cy + hh)

    library._update(player_key, (cx, cy), float(best.get("score", 0.5)), now)

    out = list(results)
    out.append({
        "champion": library.roster.player.name,
        "key":      player_key,
        "team":     "player",
        "score":    best.get("score", 0.5),
        "orb":      best.get("orb", 0),
        "box":      box,
        "location": (cx, cy),
        "inferred_from": best["key"],
    })
    return out


# ── Main entry point ──────────────────────────────────────────────────────────

def scan_death_panel(
    frame_rgb: np.ndarray,
    library:   "ChampionLibrary",
    now:       float | None = None,
    player_frame_rgb: np.ndarray | None = None,
) -> dict[str, int | None]:
    """Enemy death strip disabled — kept for API compatibility only."""
    if now is None:
        now = time.perf_counter()
    if player_frame_rgb is not None and player_frame_rgb.size > 0:
        scan_player_death_portrait(player_frame_rgb, library, now=now)
    return {}
