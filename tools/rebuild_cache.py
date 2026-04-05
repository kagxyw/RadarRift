"""
rebuild_cache.py — Download all skin portraits and rebuild the identification cache.

Step 1 — skin portraits (default: wiki.leagueoflegends.com)
    Category:Champion_loading_screens: only the first layer of “… loading screens”
    subcategories (no deeper nesting); JPGs there only; skips root files; WR/old/etc.

    Alternate:  python -m tools.rebuild_cache --fandom-skins
        Uses leagueoflegends.fandom.com instead.

Step 2 — download_icons()
    Minimap square icons from wiki.leagueoflegends.com (Category:Champion_squares,
    *OriginalSquare.png) → cache/icons/{ChampionKey}.png
    (Uses cache/champion_registry.json for display-name → key mapping.)

Step 3 — build_matrix()
    For each usable *.jpg in cache/ computes NCC + HSV histogram vectors.
    Saves thumb_matrix.npy, hist_matrix.npy, thumb_index.json.

Usage (from repo root):
    python -m tools.rebuild_cache
    python -m tools.rebuild_cache --fandom-skins
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


CACHE_DIR    = Path(__file__).resolve().parent.parent / "cache"
THUMB_MATRIX = CACHE_DIR / "thumb_matrix.npy"
HIST_MATRIX  = CACHE_DIR / "hist_matrix.npy"   # HSV histograms for shortlist
THUMB_INDEX  = CACHE_DIR / "thumb_index.json"
_WIKI_API        = "https://leagueoflegends.fandom.com/api.php"
_LOLG_WIKI_API   = "https://wiki.leagueoflegends.com/api.php"
_WIKI_UA         = "RadarRiftCacheBot/1.0 (local cache rebuild; educational)"
_FANDOM_CATEGORY = "Category:Champion_loading_screens"
_LOLG_LOADING_CATEGORY = "Category:Champion_loading_screens"
_LOLG_ICON_CATEGORY = "Category:Champion_squares"

# Subcategories under Category:Champion_loading_screens to NOT recurse into
# (misc / legacy / Wild Rift / unused — not SR champion loading art we need).
_LOADING_TREE_SKIP_SUBCATEGORIES: frozenset[str] = frozenset(
    {
        "Category:Loading_screen_images",
        "Category:Old_champion_loading_screens",
        "Category:Special_champion_loading_screens",
        "Category:Unused_champion_loading_screens",
        "Category:WR_champion_loading_screens",
    }
)

# ── preprocessing constants ───────────────────────────────────────────────────

_PORTRAIT_TOP_H = 480   # top rows of tall loading art (matches in-game card crop)
_BORDER_FRAC = 0.04  # trim 4 % from each edge
NCC_SIZE     = (128, 173)  # reference NCC resize target

# HSV histogram bins:  H=16, S=8, V=8  →  1024-element vector
_HIST_BINS   = [16, 8, 8]


# ── HSV histogram ─────────────────────────────────────────────────────────────

def _hsv_hist(img: Image.Image) -> np.ndarray:
    """
    L1-normalised 3-D HSV histogram (1024-dim float32, sums to 1).

    L1 normalisation keeps the Bhattacharyya coefficient in [0, 1]:
        BC(a,b) = Σ √(aᵢ · bᵢ) ≤ 1  (Cauchy-Schwarz, both sum to 1)

    Uses the full portrait so the reference captures the champion's
    complete colour palette regardless of skin/variant.
    """
    bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h   = cv2.calcHist([hsv], [0, 1, 2], None, _HIST_BINS, [0, 180, 0, 256, 0, 256])
    cv2.normalize(h, h, norm_type=cv2.NORM_L1)
    return h.flatten().astype(np.float32)


# ── NCC vector (kept as fast secondary verification) ─────────────────────────

def _ncc_vec(img: Image.Image) -> np.ndarray:
    """NCC vector using the top _PORTRAIT_TOP_H rows of the loading portrait."""
    img = img.convert("RGB")
    w, h = img.size
    if h > _PORTRAIT_TOP_H:
        img = img.crop((0, 0, w, _PORTRAIT_TOP_H))
    w2, h2 = img.size
    px = max(1, int(w2 * _BORDER_FRAC))
    py = max(1, int(h2 * _BORDER_FRAC))
    img = img.crop((px, py, w2 - px, h2 - py))
    arr = np.array(img.resize(NCC_SIZE, Image.LANCZOS), dtype=np.float32)
    arr -= arr.mean()
    return (arr / (np.linalg.norm(arr) + 1e-8)).flatten()


def _load_champion_data_for_icons() -> dict:
    """Champion id → {name} map (same shape as cache/champion_registry.json `data`)."""
    path = CACHE_DIR / "champion_registry.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path} — required for minimap icon name→key mapping.",
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    data = raw.get("data", raw)
    if not isinstance(data, dict):
        raise ValueError("champion_registry.json must contain a top-level 'data' object.")
    return data


# ── Fandom wiki (League of Legends Wiki) skin scraper ─────────────────────────

def _mw_get(api_url: str, params: dict) -> dict:
    """GET request to a MediaWiki API with a proper User-Agent."""
    q = {**params, "format": "json"}
    url = api_url + "?" + urllib.parse.urlencode(q)
    req = urllib.request.Request(url, headers={"User-Agent": _WIKI_UA})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.loads(r.read())


def _mw_list_category_members_all(
    api_url: str,
    cmtitle: str,
    cmtype: str,
) -> list[dict]:
    """Paginate categorymembers for one category (cmtype e.g. 'file', 'subcat', 'file|subcat')."""
    out: list[dict] = []
    continue_params: dict = {}
    cat = cmtitle.replace(" ", "_")
    if not cat.startswith("Category:"):
        cat = "Category:" + cat
    while True:
        q: dict = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": cat,
            "cmtype": cmtype,
            "cmlimit": "500",
        }
        q.update(continue_params)
        data = _mw_get(api_url, q)
        out.extend(data.get("query", {}).get("categorymembers", []))
        cont = data.get("continue")
        if not cont:
            break
        continue_params = {k: v for k, v in cont.items() if k != "batchcomplete"}
        time.sleep(0.12)
    return out


def _norm_category_title(title: str) -> str:
    """Category title with underscores for comparison to _LOADING_TREE_SKIP_SUBCATEGORIES."""
    t = title.strip().replace(" ", "_")
    if not t.startswith("Category:"):
        t = "Category:" + t
    return t


def _category_ends_with_loading_screens(title: str) -> bool:
    """True if the category name ends with “loading screens” (excludes “Loading screen images”, etc.)."""
    if not title.startswith("Category:"):
        return False
    rest = title[len("Category:") :].replace("_", " ").strip().lower()
    return rest.endswith("loading screens")


def _mw_collect_loading_jpg_titles_recursive(
    api_url: str,
    root_category: str,
    verbose: bool = False,
    *,
    skip_direct_root_files: bool = False,
    max_subcategory_depth: int | None = None,
) -> list[str]:
    """
    Breadth-first walk from root_category, collecting File:… .jpg titles.

    max_subcategory_depth:
        None — follow every nested “… loading screens” subcategory (deep tree).
        1 — only direct children of root (e.g. per-champion “X loading screens”);
            JPGs there are collected but deeper subcategories are not entered.

    If skip_direct_root_files is True, JPGs filed directly on the root category
    are ignored.

    Only follows subcategories whose titles end with “loading screens”.

    Skips known non-SR subtrees — see _LOADING_TREE_SKIP_SUBCATEGORIES.
    """
    root = root_category.strip().replace(" ", "_")
    if not root.startswith("Category:"):
        root = "Category:" + root

    seen_categories: set[str] = set()
    file_titles: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(root, 0)])

    while queue:
        cat, depth = queue.popleft()
        if cat in seen_categories:
            continue
        seen_categories.add(cat)

        members = _mw_list_category_members_all(api_url, cat, "file|subcat")
        for m in members:
            t = m.get("title", "")
            if not t:
                continue
            if t.startswith("Category:"):
                if max_subcategory_depth is not None and depth >= max_subcategory_depth:
                    continue
                if not _category_ends_with_loading_screens(t):
                    continue
                sub = t.replace(" ", "_")
                if _norm_category_title(sub) in _LOADING_TREE_SKIP_SUBCATEGORIES:
                    continue
                if sub not in seen_categories:
                    queue.append((sub, depth + 1))
            elif t.startswith("File:") and t.lower().endswith(".jpg"):
                if skip_direct_root_files and cat == root:
                    continue
                file_titles.add(t)

        if verbose and len(seen_categories) % 30 == 0:
            print(
                f"  … {len(seen_categories)} categories scanned, "
                f"{len(file_titles)} .jpg file(s) found so far",
            )
        time.sleep(0.06)

    return sorted(file_titles)


def _mw_batch_image_urls(api_url: str, file_titles: list[str]) -> dict[str, str]:
    """Map full page title → direct image URL (latest revision). ~50 titles per query."""
    out: dict[str, str] = {}
    for i in range(0, len(file_titles), 50):
        batch = file_titles[i : i + 50]
        pipe = "|".join(batch)
        data = _mw_get(
            api_url,
            {
                "action": "query",
                "titles": pipe,
                "prop": "imageinfo",
                "iiprop": "url",
            },
        )
        for page in data.get("query", {}).get("pages", {}).values():
            title = page.get("title")
            if not title or "imageinfo" not in page:
                continue
            info = page["imageinfo"][0]
            if "url" in info:
                out[title] = info["url"]
        time.sleep(0.12)
    return out


def _wiki_file_title_to_cache_name(title: str) -> str:
    """File:Ahri OriginalLoading.jpg → Ahri_OriginalLoading.jpg"""
    name = title[5:] if title.startswith("File:") else title
    return name.replace(" ", "_")


def _lol_wiki_list_original_square_titles() -> list[str]:
    """All default minimap squares in Category:Champion_squares (excludes Arcane/TFT variants)."""
    titles: list[str] = []
    continue_params: dict = {}
    suffix = " OriginalSquare.png"
    while True:
        q: dict = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": _LOLG_ICON_CATEGORY.replace(" ", "_"),
            "cmtype": "file",
            "cmlimit": "500",
        }
        q.update(continue_params)
        data = _mw_get(_LOLG_WIKI_API, q)
        for m in data.get("query", {}).get("categorymembers", []):
            t = m.get("title", "")
            if t.startswith("File:") and t.endswith(suffix):
                titles.append(t)
        cont = data.get("continue")
        if not cont:
            break
        continue_params = {k: v for k, v in cont.items() if k != "batchcomplete"}
        time.sleep(0.12)
    return titles


def _champion_display_name_to_key(champ_data: dict) -> dict[str, str]:
    """Map champion display `name` → id key (wiki files use the same spelling, except Nunu)."""
    m = {v["name"]: k for k, v in champ_data.items()}
    m["Nunu"] = "Nunu"
    return m


def _download_url_to_file(url: str, path: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": _WIKI_UA})
    with urllib.request.urlopen(req, timeout=90) as r:
        path.write_bytes(r.read())


def download_skins_wiki(
    verbose: bool = True,
    on_progress: callable | None = None,
    workers: int = 8,
    api_url: str = _LOLG_WIKI_API,
    root_category: str = _LOLG_LOADING_CATEGORY,
    skip_direct_root_files: bool | None = None,
    max_subcategory_depth: int | None = None,
) -> int:
    """
    Download loading-screen JPGs from a League wiki via the MediaWiki API.

    Default wiki.leagueoflegends.com + Category:Champion_loading_screens: direct
    subcategories only (depth 1), no nested “… loading screens” under those;
    root page JPGs skipped. Fandom default keeps full recursion unless overridden.

    Filenames: spaces → underscores, e.g. Ahri_OriginalLoading.jpg (build_matrix-compatible).

    Returns count of newly written files (skips existing non-empty files).
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _lol_loading = (
        api_url == _LOLG_WIKI_API
        and _norm_category_title(root_category)
        == _norm_category_title(_LOLG_LOADING_CATEGORY)
    )
    if skip_direct_root_files is None:
        skip_direct_root_files = _lol_loading
    if max_subcategory_depth is None:
        max_subcategory_depth = 1 if _lol_loading else None

    CACHE_DIR.mkdir(exist_ok=True)
    if verbose:
        host = "wiki.leagueoflegends.com" if "wiki.league" in api_url else urllib.parse.urlparse(api_url).netloc
        if max_subcategory_depth == 1:
            sub = "first layer of subcategories only"
        elif max_subcategory_depth is None:
            sub = "full recursion"
        else:
            sub = f"subcategory depth ≤ {max_subcategory_depth}"
        if skip_direct_root_files:
            sub += "; root JPGs skipped"
        print(f"Listing {root_category} on {host} ({sub}) …")
    titles = _mw_collect_loading_jpg_titles_recursive(
        api_url,
        root_category,
        verbose=verbose,
        skip_direct_root_files=skip_direct_root_files,
        max_subcategory_depth=max_subcategory_depth,
    )
    if verbose:
        print(f"  Found {len(titles)} .jpg file(s) total.")

    if not titles:
        return 0

    if verbose:
        print("Resolving image URLs (batched) …")
    url_map = _mw_batch_image_urls(api_url, titles)

    todo: list[tuple[str, str, Path]] = []
    for t in titles:
        url = url_map.get(t)
        if not url:
            if verbose:
                print(f"  !! No URL for {t}")
            continue
        fname = _wiki_file_title_to_cache_name(t)
        dest = CACHE_DIR / fname
        if dest.exists() and dest.stat().st_size > 512:
            continue
        todo.append((t, url, dest))

    total = len(todo)
    if total == 0:
        if verbose:
            print("All wiki loading screens already cached (or unresolved).")
        return 0

    if verbose:
        print(f"Downloading {total} missing file(s) with {workers} workers …")

    downloaded = 0
    completed = 0
    lock = threading.Lock()

    def _job(item: tuple[str, str, Path]) -> bool:
        _t, u, p = item
        try:
            _download_url_to_file(u, p)
            return True
        except Exception as e:
            if verbose:
                print(f"  !! {p.name}: {e}")
            return False

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_job, it): it for it in todo}
        for fut in as_completed(futs):
            try:
                ok = fut.result()
                with lock:
                    completed += 1
                    if ok:
                        downloaded += 1
                        if verbose and downloaded % 25 == 0:
                            print(f"  … {downloaded}/{total}")
            except Exception:
                with lock:
                    completed += 1
            if on_progress:
                with lock:
                    on_progress(completed, total)

    if verbose:
        print(f"Downloaded {downloaded} new loading screen(s) from wiki.")
    return downloaded


def download_skins_fandom(
    verbose: bool = True,
    on_progress: callable | None = None,
    workers: int = 8,
) -> int:
    """Same as download_skins_wiki but uses Fandom’s API (leagueoflegends.fandom.com)."""
    return download_skins_wiki(
        verbose=verbose,
        on_progress=on_progress,
        workers=workers,
        api_url=_WIKI_API,
        root_category=_FANDOM_CATEGORY,
        skip_direct_root_files=False,
        max_subcategory_depth=None,
    )


# ── matrix builder ────────────────────────────────────────────────────────────

def _is_cache_skin_jpg(path: Path) -> bool:
    """
    Numeric skin id:  Champion_12.jpg
    Wiki loading art: Champion_SkinNameLoading.jpg
    """
    stem = path.stem
    if stem.lower().endswith("loading"):
        return True
    parts = stem.rsplit("_", 1)
    return len(parts) == 2 and parts[-1].isdigit()


def build_matrix(verbose: bool = True,
                 on_progress: callable | None = None) -> None:
    """Build thumb_matrix.npy, hist_matrix.npy + thumb_index.json.

    on_progress(done, total) is called after each image is processed.
    """
    jpgs = sorted(p for p in CACHE_DIR.glob("*.jpg") if _is_cache_skin_jpg(p))
    if not jpgs:
        print("No .jpg files found in cache/.")
        return

    if verbose:
        print(f"Building cache from {len(jpgs)} skins …")

    ncc_vecs:  list[np.ndarray] = []
    hist_vecs: list[np.ndarray] = []
    index:     list[dict]       = []

    for i, jpg in enumerate(jpgs):
        if verbose and ((i + 1) % 200 == 0 or i == 0):
            print(f"  [{i+1}/{len(jpgs)}] {jpg.stem}")
        try:
            img = Image.open(jpg)
            img.verify()
            img = Image.open(jpg).convert("RGB")
        except Exception:
            print(f"  !! Skipping corrupt file: {jpg.name}")
            img = Image.new("RGB", (308, 560), (0, 0, 0))

        ncc_vecs.append(_ncc_vec(img))
        hist_vecs.append(_hsv_hist(img))
        key = jpg.stem.split("_")[0]
        skin_id = jpg.stem[len(key) + 1 :] if "_" in jpg.stem else ""
        index.append({"key": key, "name": key, "skin": skin_id, "file": jpg.name})
        if on_progress:
            on_progress(i + 1, len(jpgs))

    np.save(THUMB_MATRIX, np.stack(ncc_vecs).astype(np.float32))
    np.save(HIST_MATRIX,  np.stack(hist_vecs).astype(np.float32))
    with open(THUMB_INDEX, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)

    if verbose:
        print(f"\nSaved NCC  {np.stack(ncc_vecs).shape}  →  {THUMB_MATRIX}")
        print(f"Saved HIST {np.stack(hist_vecs).shape}  →  {HIST_MATRIX}")
        print(f"Saved {len(index)} entries  →  {THUMB_INDEX}")


# ── minimap icon pre-downloader ───────────────────────────────────────────────

def download_icons(
    verbose: bool = True,
    on_progress: callable | None = None,
    workers: int = 8,
) -> int:
    """
    Download minimap square icons into cache/icons/{ChampionKey}.png.

    Uses wiki.leagueoflegends.com Category:Champion_squares (*OriginalSquare.png)
    via the MediaWiki API. Name→key mapping comes from cache/champion_registry.json.

    on_progress(done, total) is called after each download attempt finishes.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    icon_dir = CACHE_DIR / "icons"
    icon_dir.mkdir(parents=True, exist_ok=True)
    champ_data = _load_champion_data_for_icons()

    # ── LoL Wiki (Weird Gloop) — default skin squares only ───────────────
    name_to_key = _champion_display_name_to_key(champ_data)
    suffix = " OriginalSquare.png"

    if verbose:
        print(f"Listing {_LOLG_ICON_CATEGORY} (*OriginalSquare.png) …")
    titles = _lol_wiki_list_original_square_titles()
    if verbose:
        print(f"  Found {len(titles)} default square file(s).")

    if not titles:
        return 0

    if verbose:
        print("Resolving image URLs (batched) …")
    url_map = _mw_batch_image_urls(_LOLG_WIKI_API, titles)

    todo: list[tuple[str, str, Path]] = []
    for t in titles:
        display_name = t[5 : -len(suffix)] if t.startswith("File:") else t[: -len(suffix)]
        key = name_to_key.get(display_name)
        if not key:
            continue
        url = url_map.get(t)
        if not url:
            if verbose:
                print(f"  !! No URL for {t}")
            continue
        dest = icon_dir / f"{key}.png"
        if dest.exists() and dest.stat().st_size > 64:
            continue
        todo.append((t, url, dest))

    total = len(todo)
    if total == 0:
        if verbose:
            print("All wiki minimap icons already cached (or unresolved).")
        return 0

    if verbose:
        print(f"Downloading {total} missing icon(s) with {workers} workers …")

    downloaded = 0
    completed = 0
    lock = threading.Lock()

    def _job(item: tuple[str, str, Path]) -> bool:
        _t, u, p = item
        try:
            _download_url_to_file(u, p)
            return True
        except Exception as e:
            if verbose:
                print(f"  !! {p.name}: {e}")
            return False

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_job, it): it for it in todo}
        for fut in as_completed(futs):
            try:
                ok = fut.result()
                with lock:
                    completed += 1
                    if ok:
                        downloaded += 1
                        if verbose and downloaded % 25 == 0:
                            print(f"  … {downloaded}/{total}")
            except Exception:
                with lock:
                    completed += 1
            if on_progress:
                with lock:
                    on_progress(completed, total)

    if verbose:
        cached = len(list(icon_dir.glob("*.png")))
        print(f"Downloaded {downloaded} new icon(s). Total icons cached: {cached}")
    return downloaded


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Rebuild skin cache + identification matrices.")
    ap.add_argument(
        "--fandom-skins",
        action="store_true",
        help="Use Fandom wiki for loading screens instead of wiki.leagueoflegends.com.",
    )
    args = ap.parse_args()

    print("=== Step 1: download missing skin portraits ===")
    if args.fandom_skins:
        print("(source: Fandom — Category:Champion_loading_screens, recursive)")
        download_skins_fandom(verbose=True)
    else:
        print(
            "(source: wiki.leagueoflegends.com — first-layer subcategories of "
            "Category:Champion_loading_screens only)",
        )
        download_skins_wiki(verbose=True)
    print()
    print("=== Step 2: download missing minimap icons ===")
    print("(source: wiki.leagueoflegends.com — Category:Champion_squares)")
    download_icons(verbose=True)
    print()
    print("=== Step 3: build NCC matrix + HSV hist index ===")
    build_matrix(verbose=True)


if __name__ == "__main__":
    main()
