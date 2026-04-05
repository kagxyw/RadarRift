"""
tracker.py — Real-time LoL minimap champion tracker.

Identification pipeline per detected box
──────────────────────────────────────────
Stage 1 — HSV histogram shortlist  (vectorised, ~0.1 ms / frame)
    Bhattacharyya coefficient:  BC(a,b) = Σ √(aᵢ·bᵢ)  ∈ [0,1]
    Vectorised as √ref_mat @ √crop_mat.T  (n_champ × n_box matrix multiply).
    Scale- and crop-invariant.  Narrows each detection down to top-K
    candidates from the 10-champion roster.

Stage 2 — ORB verification  (precomputed refs, ~1 ms / box / candidate)
    Reference ORB keypoints+descriptors are computed ONCE at library init.
    Per-frame: compute query ORB on each YOLO crop, then BFMatcher against
    the K shortlisted references.  ORB is scale- and rotation-invariant,
    so it distinguishes champions that share a similar colour scheme
    (e.g. Shen vs Sejuani — both dark/blue-grey).

Combined score = BC  +  λ·ORB_norm  +  position bonus
    ORB_norm = min(orb_matches / _ORB_SCALE, 1.0)
    λ = _ORB_W weights the ORB contribution relative to BC.

Assignment: greedy on combined score; BC alone must clear _BC_THRESHOLD.
Champion stays ON MAP for 2 s after last detection.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from champions import Champion, ChampionRoster

# ── Tracking constants ────────────────────────────────────────────────────────

# Base spawn positions (fraction of minimap width/height)
_BASE_BLUE = (0.07, 0.93)   # bottom-left — blue side base
_BASE_RED  = (0.93, 0.07)   # top-right  — red  side base

_HIGH_CONF      = 0.60   # combined score above this → position fully trusted
_OFF_TIMEOUT    = 5.0    # seconds without detection before champion → OFF MAP
_MAX_DIST_FRAC  = 0.25   # _MAX_DIST = minimap_size * this fraction
_POS_W_HIGH     = 0.45   # position bonus weight when last pos was high-confidence
_POS_W_LOW      = 0.25   # position bonus weight when last pos was low-confidence

# Debounce / trail
_TRAIL_LEN        = 3    # number of confirmed positions kept for direction arrow
_MIN_CONFIRM      = 5    # consecutive matches needed before committing a position
_CLOSE_PX         = 18   # movement ≤ this many px is accepted immediately (same champ)
# Stage 1 — HSV histogram (Bhattacharyya)
_BC_THRESHOLD   = 0.3   # reject if BC below this
_HIST_TOP_K     = 5      # candidates forwarded from histogram to ORB
_HIST_BINS      = [16, 8, 8]
_HIST_RANGES    = [0, 180, 0, 256, 0, 256]

# Stage 2 — ORB
_ORB_SIZE       = 128    # resize icons + crops to this square size for ORB
_ORB_FEATURES   = 200    # max keypoints per image
_ORB_SCALE      = 60.0   # ORB match count that maps to "perfect" (score=1.0)
_ORB_W          = 0.60   # weight of ORB_norm added to BC in combined score

# ── Icon cache ────────────────────────────────────────────────────────────────

ICON_DIR   = Path(__file__).parent / "cache" / "icons"
_REGISTRY_PATH = Path(__file__).parent / "cache" / "champion_registry.json"
_CHAMP_REGISTRY: dict[str, dict] | None = None


def _champ_registry_data() -> dict[str, dict]:
    """Bundled id → {name, ...} (cache/champion_registry.json), no network."""
    global _CHAMP_REGISTRY
    if _CHAMP_REGISTRY is None:
        if _REGISTRY_PATH.exists():
            try:
                raw = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
                data = raw.get("data", raw)
                _CHAMP_REGISTRY = data if isinstance(data, dict) else {}
            except Exception:
                _CHAMP_REGISTRY = {}
        else:
            _CHAMP_REGISTRY = {}
    return _CHAMP_REGISTRY


def canon_champ_key(name_or_key: str) -> str:
    """
    Resolve roster / UI strings to the filename stem used in cache/icons/{key}.png.
    Tries exact key, then registry key match, then registry display name (e.g. Wukong → MonkeyKing).
    """
    raw = name_or_key.strip()
    if not raw:
        return raw
    if (ICON_DIR / f"{raw}.png").exists():
        return raw
    lo = raw.lower()
    data = _champ_registry_data()
    for k in data:
        if k.lower() == lo:
            return k
    for k, meta in data.items():
        if isinstance(meta, dict):
            nm = meta.get("name")
            if isinstance(nm, str) and nm.lower() == lo:
                return k
    return raw


def fetch_icon(key: str) -> Image.Image:
    """Load square icon from cache/icons only (run rebuild_cache — no network)."""
    ICON_DIR.mkdir(parents=True, exist_ok=True)
    exact = canon_champ_key(key)
    path  = ICON_DIR / f"{exact}.png"
    if path.exists():
        return Image.open(path).convert("RGB")
    return Image.new("RGB", (64, 64), (80, 80, 80))


# ── HSV histogram helpers ─────────────────────────────────────────────────────

def _img_hsv_hist(img: Image.Image) -> np.ndarray:
    """
    L1-normalised 3-D HSV histogram (1024-dim float32).

    L1 (sum-to-1) normalisation ensures the Bhattacharyya coefficient
        BC(a, b) = Σ √(aᵢ · bᵢ)
    stays in [0, 1] by Cauchy-Schwarz.
    """
    bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h   = cv2.calcHist([hsv], [0, 1, 2], None, _HIST_BINS, _HIST_RANGES)
    cv2.normalize(h, h, norm_type=cv2.NORM_L1)
    return h.flatten().astype(np.float32)


def _crop_hsv_hist(frame_rgb: np.ndarray,
                   box:       tuple[int, int, int, int]) -> np.ndarray:
    """HSV histogram from a bounding-box crop of the minimap frame."""
    x1, y1, x2, y2 = box
    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(np.prod(_HIST_BINS), dtype=np.float32)
    return _img_hsv_hist(Image.fromarray(crop))


# ── ORB helpers ───────────────────────────────────────────────────────────────

_orb_detector: cv2.ORB | None = None


def _get_orb() -> cv2.ORB:
    global _orb_detector
    if _orb_detector is None:
        _orb_detector = cv2.ORB_create(nfeatures=_ORB_FEATURES,
                                        scaleFactor=1.1,
                                        nlevels=4,
                                        edgeThreshold=10,
                                        scoreType=cv2.ORB_HARRIS_SCORE,
                                        fastThreshold=10)
    return _orb_detector


def _to_orb_gray(img: Image.Image) -> np.ndarray:
    """Resize a PIL image to _ORB_SIZE × _ORB_SIZE uint8 grayscale."""
    return cv2.resize(
        np.array(img.convert("L")),
        (_ORB_SIZE, _ORB_SIZE),
        interpolation=cv2.INTER_AREA,
    )


def _img_orb_desc(gray: np.ndarray) -> np.ndarray | None:
    """Compute ORB descriptors for an already-resized grayscale array."""
    _, des = _get_orb().detectAndCompute(gray, None)
    return des   # None if no keypoints


def _orb_match_count(des_query: np.ndarray | None,
                     des_ref:   np.ndarray | None) -> int:
    """Number of cross-checked BFMatcher hits (Hamming distance)."""
    if des_query is None or des_ref is None:
        return 0
    if len(des_query) < 2 or len(des_ref) < 2:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    return len(bf.match(des_query, des_ref))


# ── Per-champion state ────────────────────────────────────────────────────────

@dataclasses.dataclass
class _State:
    pos:          tuple[int, int] | None = None   # last confirmed minimap centre
    pos_trail:    object = dataclasses.field(     # ring-buffer of confirmed positions
        default_factory=lambda: collections.deque(maxlen=_TRAIL_LEN)
    )
    last_seen:    float = 0.0    # perf_counter timestamp
    high_conf:    bool  = False  # last BC was >= _HIGH_CONF
    # Cached movement direction (unit vector) — persists after champion goes off-map
    # so the arrow still draws even if the trail only has 1 distinct position.
    last_dir:     tuple[float, float] | None = None
    # Ghost latch: True once off-timeout is reached; only cleared after
    # _MIN_CONFIRM consecutive confirmed frames (same bar as position commits)
    ghost_active: bool  = False
    ghost_clears: int   = 0      # consecutive confirmed frames since reappearing
    # Death state — set by death_panel.scan_death_panel(), cleared on respawn
    dead:          bool = False
    dead_confirms: int  = 0   # consecutive frames icon matched; commits dead after threshold
    dead_clears:   int  = 0   # consecutive frames icon absent; clears dead after threshold


# ── Champion reference library ────────────────────────────────────────────────

class ChampionLibrary:
    def __init__(self, roster: ChampionRoster, minimap_size: int = 400):
        """
        Tracks all 10 champions: player (team="player") +
        4 allies (team="ally") + 5 enemies (team="enemy").

        minimap_size — capture-region width/height in pixels; controls
        how far a champion may jump between frames before the position
        heuristic stops boosting its score.
        """
        self.roster      = roster
        self.max_dist    = minimap_size * _MAX_DIST_FRAC
        self.off_timeout = _OFF_TIMEOUT   # may be overridden per-frame by track_frame

        # Pre-compute base pixel positions for dead-champion rendering
        fx, fy = _BASE_BLUE if roster.enemy_side == "blue" else _BASE_RED
        self.enemy_base_px = (int(minimap_size * fx), int(minimap_size * fy))

        # All 10 tracked champions, player first
        self.all:  list[Champion] = (
            [roster.player] + roster.allies + roster.enemies
        )
        self.team: list[str] = (
            ["player"]
            + ["ally"]  * len(roster.allies)
            + ["enemy"] * len(roster.enemies)
        )

        # Load cached square icons (wiki originals in cache/icons); build HSV + ORB refs
        # print("  Loading champion icons…")
        hists:    list[np.ndarray]          = []
        orb_refs: list[np.ndarray | None]   = []
        # Icons stored at _GHOST_BASE_SZ; _draw resizes to the requested ghost_size
        # at render time so the UI slider takes effect without rebuilding the library.
        self.icon_imgs: dict[str, np.ndarray] = {}

        _GHOST_BASE_SZ = 64   # source resolution stored in memory

        for c in self.all:
            try:
                icon = fetch_icon(c.key)
            except Exception as e:
                # print(f"    Warning: no icon for {c.key!r}: {e}")
                icon = Image.new("RGB", (64, 64), (80, 80, 80))

            hists.append(_img_hsv_hist(icon))
            orb_refs.append(_img_orb_desc(_to_orb_gray(icon)))

            self.icon_imgs[c.key] = np.array(
                icon.resize((_GHOST_BASE_SZ, _GHOST_BASE_SZ), Image.LANCZOS)
            )

        hist_mat       = np.stack(hists)
        self.sqrt_hist = np.sqrt(hist_mat).astype(np.float32)  # (n_champ, 1024)
        self.orb_refs  = orb_refs                               # list[ndarray|None]

        # Per-champion tracking state — persists across frames
        self._state:   dict[str, _State] = {c.key: _State() for c in self.all}
        # Consecutive-match counter for debounce (key → int)
        self._confirm: dict[str, int]    = {c.key: 0 for c in self.all}

    # ── position heuristic ───────────────────────────────────────────────────

    def _pos_bonus(self, key: str, cx: int, cy: int) -> float:
        """Spatial-continuity bonus: boost score when detection is near last pos."""
        st = self._state.get(key)
        if st is None or st.pos is None:
            return 0.0
        dx, dy = cx - st.pos[0], cy - st.pos[1]
        dist   = (dx * dx + dy * dy) ** 0.5
        w      = _POS_W_HIGH if st.high_conf else _POS_W_LOW
        return w * max(0.0, 1.0 - dist / self.max_dist)

    def _update(self, key: str, pos: tuple[int, int], bc: float, now: float):
        st = self._state[key]
        # Append to trail only when the champion has meaningfully moved
        if st.pos is None or st.pos != pos:
            st.pos_trail.append(pos)
        st.pos       = pos
        st.last_seen = now
        st.high_conf = bc >= _HIGH_CONF
        # Update cached direction from trail whenever we have >= 2 distinct points
        trail = list(st.pos_trail)
        if len(trail) >= 2:
            dx = trail[-1][0] - trail[0][0]
            dy = trail[-1][1] - trail[0][1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > 3:
                st.last_dir = (dx / dist, dy / dist)

    # ── public state queries ─────────────────────────────────────────────────

    def is_visible(self, key: str, now: float,
                   off_timeout: float = _OFF_TIMEOUT) -> bool:
        """True if the champion was matched within the last off_timeout seconds."""
        st = self._state.get(key)
        return st is not None and (now - st.last_seen) < off_timeout

    def last_pos(self, key: str) -> tuple[int, int] | None:
        st = self._state.get(key)
        return st.pos if st else None


# ── YOLO detection ────────────────────────────────────────────────────────────

_yolo_model = None


def _get_yolo(model=None):
    global _yolo_model
    if model is not None:
        return model
    if _yolo_model is None:
        from backend import load_model
        _yolo_model = load_model()
    return _yolo_model


# ── Per-frame tracking ────────────────────────────────────────────────────────

_debug_frame_counter = 0
_DEBUG_INTERVAL = 1   # save every N frames


def _save_debug_frames(frame_bgr: np.ndarray,
                       dets: list[dict],
                       boxes: list,
                       bc_mat: np.ndarray,
                       library: "ChampionLibrary",
                       results: "list[dict] | None" = None,
                       combined: "np.ndarray | None" = None,
                       orb_extra: "np.ndarray | None" = None) -> None:
    """
    Write debug images to debug_crops/:
      yolo_boxes.jpg    — raw YOLO detections
      hsv_matches.jpg   — best HSV histogram match per box
      orb_top3.jpg      — top-3 combined scores per box with BC+ORB breakdown
      tracker_output.jpg— confirmed champion identifications (team-coloured)
      {team}_{key}.png  — individual crop for each confirmed detection
    """
    try:
        import sys as _sys
        base      = Path(__file__).parent if not getattr(_sys, "frozen", False) \
                    else Path(_sys.executable).parent
        crops_dir = base / "debug_crops"
        crops_dir.mkdir(exist_ok=True)

        _COLORS_BGR = {           # BGR
            "player": ( 50, 220, 255),
            "ally":   (255, 220,  80),
            "enemy":  ( 80,  80, 255),
        }

        # ── 1: raw YOLO detections ────────────────────────────────────────────
        yolo_img = frame_bgr.copy()
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            label = f"{d.get('class_name', '')} {d['conf']:.2f}"
            cv2.rectangle(yolo_img, (x1, y1), (x2, y2), (0, 255, 80), 2)
            cv2.putText(yolo_img, label, (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 80), 1)
        cv2.imwrite(str(crops_dir / "yolo_boxes.jpg"), yolo_img)

        # ── 2: HSV best-match per box ─────────────────────────────────────────
        hsv_img = frame_bgr.copy()
        for bi, b in enumerate(boxes):
            x1, y1, x2, y2 = b
            top_ci  = int(bc_mat[:, bi].argmax())
            top_bc  = float(bc_mat[top_ci, bi])
            name    = library.all[top_ci].key if top_ci < len(library.all) else "?"
            label   = f"{name} {top_bc:.2f}"
            cv2.rectangle(hsv_img, (x1, y1), (x2, y2), (255, 160, 0), 2)
            cv2.putText(hsv_img, label, (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 160, 0), 1)
        cv2.imwrite(str(crops_dir / "hsv_matches.jpg"), hsv_img)

        # ── 3: ORB top-3 combined scores per box ─────────────────────────────
        if combined is not None and orb_extra is not None and len(boxes) > 0:
            orb_img = frame_bgr.copy()
            for bi, b in enumerate(boxes):
                x1, y1, x2, y2 = b
                top3_ci = np.argsort(combined[:, bi])[-3:][::-1]
                cv2.rectangle(orb_img, (x1, y1), (x2, y2), (0, 200, 255), 2)
                for rank, ci in enumerate(top3_ci):
                    name    = library.all[ci].key if ci < len(library.all) else "?"
                    comb_sc = float(combined[ci, bi])
                    bc_sc   = float(bc_mat[ci, bi])
                    orb_sc  = float(orb_extra[ci, bi])
                    label   = f"#{rank+1} {name} C:{comb_sc:.2f} BC:{bc_sc:.2f} O:{orb_sc:.2f}"
                    ty = y2 + 14 + rank * 14
                    cv2.putText(orb_img, label, (x1, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)
            cv2.imwrite(str(crops_dir / "orb_top3.jpg"), orb_img)

        # ── 4: confirmed tracker output ───────────────────────────────────────
        if results:
            track_img = frame_bgr.copy()
            for r in results:
                x1, y1, x2, y2 = r["box"]
                col   = _COLORS_BGR.get(r.get("team", ""), (200, 200, 200))
                team1 = r.get("team", "?")[0].upper()
                label = f"{r.get('champion', '?')} ({team1}) {r.get('score', 0):.2f}"
                cv2.rectangle(track_img, (x1, y1), (x2, y2), col, 2)
                cv2.putText(track_img, label, (x1, max(y1 - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)
            cv2.imwrite(str(crops_dir / "tracker_output.jpg"), track_img)

            # ── 4: individual crops for each confirmed detection ───────────────
            for r in results:
                x1, y1, x2, y2 = r["box"]
                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                team  = r.get("team",  "unk")
                key   = r.get("key",   r.get("champion", "unknown"))
                fname = f"{team}_{key}.png"
                cv2.imwrite(str(crops_dir / fname), crop)

    except Exception:
        print("Error saving debug frames")

def track_frame(
    frame_rgb:   np.ndarray,
    library:     ChampionLibrary,
    model=None,
    threshold:   float        = _BC_THRESHOLD,
    conf:        float        = 0.25,
    now:         float | None = None,
    off_timeout: float        = _OFF_TIMEOUT,
) -> list[dict]:
    """
    Detect and identify champions on the minimap for one frame.

    Returns a list of matched detections and updates ChampionLibrary's
    internal per-champion state.

    Two-stage matching per detected box
    ─────────────────────────────────────
    Stage 1 — HSV Bhattacharyya  (vectorised matrix multiply, very fast)
        bc_mat = √ref_mat @ √crop_mat.T           (n_champ × n_box)
        Shortlists to top-_HIST_TOP_K candidates per box.

    Stage 2 — ORB verification  (precomputed refs, fast BFMatcher per pair)
        Query ORB descriptors computed once per detected box.
        Matched against each shortlisted reference.
        ORB_norm = min(match_count / _ORB_SCALE, 1.0)

    Combined = BC + _ORB_W × ORB_norm + position_bonus
    """
    if now is None:
        now = time.perf_counter()

    library.off_timeout = off_timeout   # make it available to _draw

    from backend import infer as _infer
    global _debug_frame_counter
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    dets = _infer(_get_yolo(model), frame_bgr, conf=conf)
    if not dets:
        return []

    boxes   = [d["box"] for d in dets]
    centres = [((b[0] + b[2]) // 2, (b[1] + b[3]) // 2) for b in boxes]
    n_box   = len(boxes)
    n_champ = len(library.all)

    # ── Stage 1: HSV Bhattacharyya matrix  (n_champ × n_box) ─────────────────
    crop_hists = np.stack([_crop_hsv_hist(frame_rgb, b) for b in boxes])
    sqrt_crop  = np.sqrt(crop_hists)                       # (n_box, 1024)
    bc_mat     = library.sqrt_hist @ sqrt_crop.T           # (n_champ, n_box)

    # ── Stage 2: ORB verification for top-K histogram candidates ─────────────
    # Compute query ORB descriptors once per box
    query_descs: list[np.ndarray | None] = []
    for b in boxes:
        x1, y1, x2, y2 = b
        crop_arr = frame_rgb[y1:y2, x1:x2]
        if crop_arr.size == 0:
            query_descs.append(None)
        else:
            gray = cv2.resize(crop_arr[:, :, ::-1] if crop_arr.ndim == 3
                              else crop_arr,
                              (_ORB_SIZE, _ORB_SIZE),
                              interpolation=cv2.INTER_LINEAR)
            gray_u8 = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY) if gray.ndim == 3 else gray
            query_descs.append(_img_orb_desc(gray_u8))

    # Build combined score matrix; ORB only computed for shortlisted pairs
    combined  = bc_mat.copy()
    orb_extra = np.zeros_like(bc_mat)

    for bi in range(n_box):
        # shortlist: indices of top-K champions by BC for this box
        shortlist = np.argsort(bc_mat[:, bi])[-_HIST_TOP_K:]
        for ci in shortlist:
            matches         = _orb_match_count(query_descs[bi],
                                               library.orb_refs[ci])
            orb_norm        = min(matches / _ORB_SCALE, 1.0)
            orb_extra[ci, bi] = orb_norm

    combined += _ORB_W * orb_extra

    # ── Position bonus ────────────────────────────────────────────────────────
    for ci, c in enumerate(library.all):
        for bi, (cx, cy) in enumerate(centres):
            combined[ci, bi] += library._pos_bonus(c.key, cx, cy)

    # ── Greedy assignment; BC alone must clear threshold ──────────────────────
    quads = [
        (combined[ci, bi], bc_mat[ci, bi], ci, bi)
        for ci in range(n_champ)
        for bi in range(n_box)
    ]
    quads.sort(reverse=True)

    assigned_champ: set[int] = set()
    assigned_box:   set[int] = set()
    results: list[dict]      = []

    matched_keys: set[str] = set()

    for comb_sc, bc_sc, ci, bi in quads:
        if bc_sc < threshold:
            break
        if ci in assigned_champ or bi in assigned_box:
            continue
        c      = library.all[ci]
        cx, cy = centres[bi]
        orb_sc = orb_extra[ci, bi]

        # ── Debounce: require _MIN_CONFIRM consecutive matches before committing
        #    a position change.  Tiny movements (same champion drifting slightly)
        #    are accepted immediately; large jumps need confirmation.
        st    = library._state[c.key]
        close = (st.pos is not None and
                 ((cx - st.pos[0])**2 + (cy - st.pos[1])**2) ** 0.5 <= _CLOSE_PX)
        library._confirm[c.key] = library._confirm.get(c.key, 0) + 1
        confirmed = close or library._confirm[c.key] >= _MIN_CONFIRM

        if confirmed:
            results.append({
                "champion": c.name,
                "key":      c.key,
                "team":     library.team[ci],
                "score":    round(float(bc_sc), 3),
                "orb":      int(round(orb_sc * _ORB_SCALE)),
                "box":      boxes[bi],
                "location": (cx, cy),
            })
            library._update(c.key, (cx, cy), float(bc_sc + _ORB_W * orb_sc), now)

        matched_keys.add(c.key)
        assigned_champ.add(ci)
        assigned_box.add(bi)

    # Reset confirm counter for champions not matched this frame
    for c in library.all:
        if c.key not in matched_keys:
            library._confirm[c.key] = 0

    # ── Ghost latch maintenance ───────────────────────────────────────────────
    confirmed_keys = {r["key"] for r in results}
    for c in library.all:
        st = library._state[c.key]
        if c.key in confirmed_keys:
            st.ghost_clears += 1
            if st.ghost_clears >= _MIN_CONFIRM:
                st.ghost_active = False
                st.dead = False   # confirmed alive on map — clear death state
        else:
            st.ghost_clears = 0
            # Latch ghost ON once the champion exceeds the off-timeout window
            if st.pos is not None and not library.is_visible(c.key, now, off_timeout):
                st.ghost_active = True

    # ── Debug snapshot (every _DEBUG_INTERVAL frames) ────────────────────────
    # _debug_frame_counter += 1
    # if _debug_frame_counter % _DEBUG_INTERVAL == 0:
    #     _save_debug_frames(frame_bgr, dets, boxes, bc_mat, library, results,
    #                        combined=combined, orb_extra=orb_extra)

    return results


# ── Roster helpers ────────────────────────────────────────────────────────────

def _roster_from_names(
    enemy_names: list[str],
    ally_names:  list[str] | None = None,
) -> ChampionRoster:
    dd_path  = Path(__file__).parent / "cache" / "champions.json"
    name_map = {}
    if dd_path.exists():
        with open(dd_path, encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.get("data", {}).items():
            name_map[k.lower()] = v["name"]

    def _make(raw: str) -> Champion:
        key     = raw.strip()
        display = name_map.get(key.lower(), key)
        return Champion(name=display, key=key)

    return ChampionRoster(
        player  = Champion(name="(you)", key=""),
        allies  = [_make(n) for n in (ally_names or [])],
        enemies = [_make(n) for n in enemy_names],
    )
