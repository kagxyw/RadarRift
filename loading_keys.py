"""
wiki_loading_keys.py — Map League wiki loading-screen filenames to Data Dragon keys.

Wiki stems use underscores, apostrophes, and periods (e.g. ``Lee_Sin_``, ``Dr._Mundo_``,
``Cho'Gath_``). Registry / minimap keys are alphanumeric: join champion segments and strip
``'``, ``.``, and spaces so ``Dr._Mundo`` + ``OriginalLoading`` → ``DrMundo``.

``champion_registry.json`` must be passed as the inner ``data`` dict (id → {name, …}).
"""

from __future__ import annotations


def _strip_wiki_segment(seg: str) -> str:
    """Remove punctuation/spaces Riot omits from Data Dragon champion ids."""
    return seg.replace("'", "").replace(".", "").replace(" ", "")


def _candidate_key(champ_parts: list[str]) -> str:
    return "".join(_strip_wiki_segment(p) for p in champ_parts)


def _match_registry_key(cand: str, reg_keys: frozenset[str]) -> str | None:
    """Exact id match, then case-insensitive (``ChoGath`` → ``Chogath``)."""
    if cand in reg_keys:
        return cand
    cl = cand.lower()
    for k in reg_keys:
        if k.lower() == cl:
            return k
    return None


# Old thumb_index used only stem.split("_")[0] — truncated keys that are not fixed by
# stripping punctuation (must merge with following segments we no longer have).
LEGACY_THUMB_KEY_TO_REGISTRY: dict[str, str] = {
    "Jarvan": "JarvanIV",
    "Lee": "LeeSin",
    "Master": "MasterYi",
    "Miss": "MissFortune",
    "Twisted": "TwistedFate",
    "Xin": "XinZhao",
    "Dr.": "DrMundo",
}


def resolve_wiki_loading_stem(
    stem: str,
    registry: dict[str, dict],
) -> tuple[str, str, str]:
    """
    Return (registry_key, display_name, skin_id_tail).

    Tries longest champion prefix first: ``Lee_Sin_OriginalLoading`` → ``LeeSin`` +
    ``OriginalLoading``. Apostrophes/periods are stripped only inside segment joins.
    """
    reg_keys = frozenset(registry.keys())
    parts = stem.split("_")
    if len(parts) < 2:
        first = parts[0] if parts else ""
        rkey = LEGACY_THUMB_KEY_TO_REGISTRY.get(first)
        if rkey is None:
            m = _match_registry_key(_strip_wiki_segment(first), reg_keys)
            rkey = m if m is not None else first
        info = registry.get(rkey) or {}
        return rkey, info.get("name", rkey), ""

    # Longest champion prefix first (avoids ``Jarvan`` before ``JarvanIV``).
    for i in range(len(parts) - 1, 0, -1):
        cand = _candidate_key(parts[:i])
        rkey = _match_registry_key(cand, reg_keys)
        if rkey is not None:
            skin_id = "_".join(parts[i:])
            info = registry[rkey] or {}
            name = info.get("name", rkey)
            return rkey, name, skin_id

    first = parts[0]
    tail = "_".join(parts[1:])
    rkey = LEGACY_THUMB_KEY_TO_REGISTRY.get(first, first)
    stripped = _strip_wiki_segment(first)
    m = _match_registry_key(stripped, reg_keys)
    if m is not None:
        rkey = m
    info = registry.get(rkey) or {}
    name = info.get("name", first if rkey == first else rkey)
    return rkey, name, tail


def normalize_legacy_thumb_key(
    key: str,
    registry_keys: frozenset[str] | None = None,
) -> str:
    """Map a thumb_index ``key`` from an older build to the registry key."""
    if key in LEGACY_THUMB_KEY_TO_REGISTRY:
        return LEGACY_THUMB_KEY_TO_REGISTRY[key]
    stripped = _strip_wiki_segment(key)
    if registry_keys is not None:
        m = _match_registry_key(stripped, registry_keys)
        if m is not None:
            return m
    return key
