#!/usr/bin/env python3
"""Cross-match a few cache/icons with the same HSV + ORB path tracker uses.

Run from repo root:
    python -m tools.test_hsv_orb_icons
    python -m tools.test_hsv_orb_icons Lux Pantheon Fiddlesticks
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np
from PIL import Image

from tracker import (
    _BC_THRESHOLD,
    _ORB_SCALE,
    _ORB_W,
    _img_hsv_hist,
    _img_orb_desc,
    _orb_match_count,
    _to_orb_gray,
)

ICONS = _ROOT / "cache" / "icons"
DEFAULT = ("Lux", "Pantheon", "Fiddlesticks")


def _load(name: str) -> tuple[str, Image.Image]:
    p = ICONS / f"{name}.png"
    if not p.is_file():
        # case-insensitive fallback
        hits = [f for f in ICONS.glob("*.png") if f.stem.lower() == name.lower()]
        if not hits:
            raise FileNotFoundError(f"missing icon: {p}")
        p = hits[0]
    return p.stem, Image.open(p).convert("RGB")


def main(names: list[str]) -> int:
    loaded = [_load(n) for n in names]
    keys = [k for k, _ in loaded]
    hists = [_img_hsv_hist(img) for _, img in loaded]
    orbs = [_img_orb_desc(_to_orb_gray(img)) for _, img in loaded]
    sqrt = np.sqrt(np.stack(hists).astype(np.float32))

    n = len(keys)
    bc = sqrt @ sqrt.T
    orb = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            m = _orb_match_count(orbs[i], orbs[j])
            orb[i, j] = min(m / _ORB_SCALE, 1.0)
    comb = bc + _ORB_W * orb

    print(f"BC threshold = {_BC_THRESHOLD}   ORB_W = {_ORB_W}   ORB_SCALE = {_ORB_SCALE}")
    print(f"icons: {keys}\n")

    def _table(title: str, mat: np.ndarray) -> None:
        print(title)
        hdr = f"{'query\\\\ref':<14}" + "".join(f"{k:>14}" for k in keys)
        print(hdr)
        for i, q in enumerate(keys):
            row = f"{q:<14}" + "".join(f"{mat[i, j]:14.3f}" for j in range(n))
            print(row)
        print()

    _table("HSV Bhattacharyya (self should be ~1.0)", bc)
    _table("ORB_norm (self should be high)", orb)
    _table(f"Combined = BC + {_ORB_W}*ORB", comb)

    print("Nearest-neighbour (each query → best other ref):")
    ok = True
    for i, q in enumerate(keys):
        scores = comb[i].copy()
        scores[i] = -1.0
        j = int(scores.argmax())
        margin = float(comb[i, i] - comb[i, j])
        # Self-match must beat every other AND clear BC threshold on self.
        self_ok = float(bc[i, i]) >= _BC_THRESHOLD and margin > 0
        status = "OK" if self_ok else "CONFUSED"
        if not self_ok:
            ok = False
        print(
            f"  {q:<14} self={comb[i, i]:.3f}  "
            f"best_other={keys[j]} ({comb[i, j]:.3f})  "
            f"margin={margin:+.3f}  [{status}]"
        )

    # Also: treat each icon as a "crop" matched against the full shortlist of
    # the three, mimicking tracker's greedy assignment rule (BC must clear).
    print("\nTracker-style assignment (BC must clear "
          f"{_BC_THRESHOLD}; pick max combined):")
    assigned = set()
    pairs = []
    for i in range(n):
        for j in range(n):
            if bc[i, j] >= _BC_THRESHOLD:
                pairs.append((float(comb[i, j]), float(bc[i, j]), i, j))
    pairs.sort(reverse=True)
    used_q, used_r = set(), set()
    for score, bc_sc, i, j in pairs:
        if i in used_q or j in used_r:
            continue
        used_q.add(i)
        used_r.add(j)
        match = "correct" if i == j else "WRONG"
        if i != j:
            ok = False
        print(f"  {keys[i]:<14} → {keys[j]:<14}  "
              f"comb={score:.3f} bc={bc_sc:.3f}  [{match}]")
    for i, q in enumerate(keys):
        if i not in used_q:
            print(f"  {q:<14} → (unassigned)  best BC={bc[i].max():.3f}")
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    names = sys.argv[1:] or list(DEFAULT)
    # No "Locke" in LoL — accept common typo → Lux
    names = ["Lux" if n.lower() == "locke" else n for n in names]
    raise SystemExit(main(names))
