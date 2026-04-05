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

# Red text bounds for timer digits
_TIMER_R_DIFF    = 55    # R - max(G, B) must exceed this to count as red
_TIMER_R_MIN     = 110
_TIMER_B_MAX     = 110
_TIMER_MIN_PIX   = 8     # minimum red pixels needed to even try OCR


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


# ── Main entry point ──────────────────────────────────────────────────────────

def scan_death_panel(
    frame_rgb: np.ndarray,
    library:   "ChampionLibrary",
    now:       float | None = None,
) -> dict[str, int | None]:
    """
    Scan one frame from the death panel capture region.

    Updates library._state[key].dead in-place for all enemies.
    Returns {key: respawn_secs} for newly-confirmed dead enemies.

    Matching uses NCC template matching (cv2.TM_CCOEFF_NORMED) on 64×64
    grayscale crops — captures spatial structure, unlike histograms.
    A champion must match for _DEAD_CONFIRM_MIN consecutive scans before
    dead=True is committed, eliminating single-frame false positives.
    """
    if now is None:
        now = time.perf_counter()

    if frame_rgb is None or frame_rgb.size == 0:
        return {}

    _ensure_tmpl_refs(library)

    icon_boxes = _find_icon_boxes(frame_rgb)

    # Extract all query features once (gray + edges + ORB per detected orb)
    blob_feats: list[dict | None] = []
    for (x1, y1, x2, y2) in icon_boxes:
        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            blob_feats.append(None)
        else:
            blob_feats.append(_extract_features(crop))

    # Greedy best-match: assign each enemy to the blob with highest combined score
    assigned:   dict[str, tuple[int, int, int, int]] = {}
    combo_scores: dict[str, float]                   = {}
    used_boxes: set[int] = set()

    for ci, c in enumerate(library.all):
        if library.team[ci] != "enemy":
            continue

        ref = library._tmpl_refs.get(c.key)  # type: ignore[attr-defined]
        if ref is None:
            continue

        best_score = -1.0
        best_bi    = -1
        for bi, feat in enumerate(blob_feats):
            if bi in used_boxes or feat is None:
                continue
            score = _combined_score(feat, ref)
            if score > best_score:
                best_score = score
                best_bi    = bi

        if best_score >= _TMPL_MATCH_MIN and best_bi >= 0:
            assigned[c.key]     = icon_boxes[best_bi]
            combo_scores[c.key] = best_score
            used_boxes.add(best_bi)

    # Update _State for all enemies
    results: dict[str, int | None] = {}
    for ci, c in enumerate(library.all):
        if library.team[ci] != "enemy":
            continue
        st = library._state[c.key]

        if c.key in assigned:
            # Blob matched — require _DEAD_CONFIRM_MIN consecutive hits
            st.dead_confirms += 1
            st.dead_clears    = 0
            if not st.dead and st.dead_confirms >= _DEAD_CONFIRM_MIN:
                st.dead = True
                results[c.key] = None
        else:
            # Blob absent — reset confirm streak; count clears if already dead
            st.dead_confirms = 0
            if st.dead:
                st.dead_clears += 1
                if st.dead_clears >= _DEAD_CLEAR_MIN:
                    # Confirmed respawn — place ghost at base with fresh timer
                    st.pos          = library.enemy_base_px
                    st.last_seen    = now
                    st.ghost_active = True
                    st.dead         = False
                    st.dead_clears  = 0

    global _debug_frame_counter
    _debug_frame_counter += 1
    if _debug_frame_counter % _DEBUG_INTERVAL == 0:
        pass  # _save_death_panel_debug(frame_rgb, icon_boxes, assigned, combo_scores)

    return results
