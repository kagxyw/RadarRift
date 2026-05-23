"""
Generate champion name speech (edge-tts online or pyttsx3 offline).

  pip install edge-tts          # neural MP3 (needs speech.platform.bing.com)
  pip install pyttsx3           # offline WAV fallback

One name (auto: try edge, then offline WAV):
  python tools/test_tts_champions.py --mp3 Yasuo

Force offline if Bing speech times out / is blocked:
  python tools/test_tts_champions.py --offline --mp3 Yasuo
  python tools/test_tts_champions.py --offline --all

edge-tts (online):
  python tools/test_tts_champions.py --backend edge --mp3 "Miss Fortune" --voice en-US-AriaNeural
  python tools/test_tts_champions.py --sample
  python tools/test_tts_champions.py --all

Output: tools/tts_out/*.mp3 or *.wav
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_REG = _ROOT / "cache" / "champion_registry.json"
_OUT = Path(__file__).resolve().parent / "tts_out"

DEFAULT_VOICE = "en-US-GuyNeural"
EDGE_RETRIES = 3
EDGE_TIMEOUT_S = 90

_FALLBACK_NAMES = [
    "Aatrox", "Ahri", "Akali", "Alistar", "Amumu", "Anivia", "Annie",
    "Aphelios", "Ashe", "Azir", "Bard", "Bel'Veth", "Blitzcrank",
    "Brand", "Braum", "Caitlyn", "Camille", "Cassiopeia", "Cho'Gath",
    "Corki", "Darius", "Diana", "Dr. Mundo", "Draven", "Ekko",
    "Elise", "Evelynn", "Ezreal", "Fiddlesticks", "Fiora", "Fizz",
    "Galio", "Gangplank", "Garen", "Gnar", "Gragas", "Graves",
    "Gwen", "Hecarim", "Heimerdinger", "Hwei", "Illaoi", "Irelia",
    "Ivern", "Janna", "Jarvan IV", "Jax", "Jayce", "Jhin", "Jinx",
    "K'Sante", "Kai'Sa", "Kalista", "Karma", "Karthus", "Kassadin",
    "Katarina", "Kayle", "Kayn", "Kennen", "Kha'Zix", "Kindred",
    "Kled", "Kog'Maw", "LeBlanc", "Lee Sin", "Leona", "Lillia",
    "Lissandra", "Lucian", "Lulu", "Lux", "Malphite", "Malzahar",
    "Maokai", "Master Yi", "Miss Fortune", "Mordekaiser", "Morgana",
    "Naafiri", "Nami", "Nasus", "Nautilus", "Neeko", "Nidalee",
    "Nilah", "Nocturne", "Nunu & Willump", "Olaf", "Orianna",
    "Ornn", "Pantheon", "Poppy", "Pyke", "Qiyana", "Quinn",
    "Rakan", "Rammus", "Rek'Sai", "Rell", "Renata Glasc",
    "Renekton", "Rengar", "Riven", "Rumble", "Ryze", "Samira",
    "Sejuani", "Senna", "Seraphine", "Sett", "Shaco", "Shen",
    "Shyvana", "Singed", "Sion", "Sivir", "Skarner", "Smolder",
    "Sona", "Soraka", "Swain", "Sylas", "Syndra", "Tahm Kench",
    "Taliyah", "Talon", "Taric", "Teemo", "Thresh", "Tristana",
    "Trundle", "Tryndamere", "Twisted Fate", "Twitch", "Udyr",
    "Urgot", "Varus", "Vayne", "Veigar", "Vel'Koz", "Vex",
    "Vi", "Viego", "Viktor", "Vladimir", "Volibear", "Warwick",
    "Wukong", "Xayah", "Xerath", "Xin Zhao", "Yasuo", "Yone",
    "Yorick", "Yuumi", "Zac", "Zed", "Zeri", "Ziggs", "Zilean",
    "Zoe", "Zyra",
]

_SAMPLE_NAMES = [
    "Miss Fortune", "Cho'Gath", "Kai'Sa", "Lee Sin", "Aphelios", "Nunu & Willump",
]

_SAMPLE_ROSTER = ["Garen", "Lee Sin", "Syndra", "Jinx", "Thresh"]

# Display name → words spoken (output filename still from display name)
SPEAK_OVERRIDES: dict[str, str] = {
    "Xin Zhao": "Shin Zhao",
    "Nunu & Willump": "Nunu",
    "Jarvan IV": "Jarvan",
}

_pyttsx3_engine = None


def speak_text_for(display_name: str) -> str:
    return SPEAK_OVERRIDES.get(display_name, display_name)


def load_champion_names() -> list[str]:
    if _REG.is_file():
        try:
            raw = json.loads(_REG.read_text(encoding="utf-8"))
            data = raw.get("data", raw)
            if isinstance(data, dict):
                names = sorted(
                    {str(v.get("name", k)).strip() for k, v in data.items()
                     if isinstance(v, dict) and v.get("name")},
                )
                if names:
                    print(f"Loaded {len(names)} names from {_REG}")
                    return names
        except Exception as e:
            print(f"Could not read registry ({e}), using built-in list.")
    print(f"Using built-in list ({len(_FALLBACK_NAMES)} names).")
    return list(_FALLBACK_NAMES)


def _safe_filename(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().replace(" ", "_")
    return s[:60] or "speech"


def _edge_network_error(exc: BaseException) -> bool:
    msg = f"{type(exc).__name__}: {exc}".lower()
    return any(
        k in msg
        for k in (
            "timeout", "connection", "connect", "bing.com",
            "speech.platform", "clientconnector", "cannot connect",
        )
    )


def _print_edge_help() -> None:
    print(
        "\nedge-tts could not reach Microsoft speech (speech.platform.bing.com).\n"
        "Common causes: firewall, region/network blocking, or no internet.\n"
        "Use offline mode (local Windows voice, WAV files):\n"
        "  python tools/test_tts_champions.py --offline --mp3 Yasuo\n"
        "  python tools/test_tts_champions.py --offline --all\n"
    )


def _get_pyttsx3():
    global _pyttsx3_engine
    if _pyttsx3_engine is None:
        import pyttsx3

        _pyttsx3_engine = pyttsx3.init()
        rate = _pyttsx3_engine.getProperty("rate")
        _pyttsx3_engine.setProperty("rate", max(120, int(rate * 0.95)))
    return _pyttsx3_engine


def make_voice_offline(text: str, path: Path) -> Path:
    path = path.with_suffix(".wav")
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  → {text!r}  [offline/pyttsx3]  →  {path.name}")
    engine = _get_pyttsx3()
    engine.save_to_file(text, str(path))
    engine.runAndWait()
    if not path.is_file():
        raise RuntimeError(f"WAV was not created: {path}")
    return path


async def make_voice_edge(text: str, path: Path, voice: str) -> Path:
    import edge_tts

    path = path.with_suffix(".mp3")
    path.parent.mkdir(parents=True, exist_ok=True)
    last_err: BaseException | None = None

    for attempt in range(1, EDGE_RETRIES + 1):
        try:
            print(f"  → {text!r}  [{voice}]  →  {path.name}"
                  + (f"  (try {attempt}/{EDGE_RETRIES})" if attempt > 1 else ""))
            communicate = edge_tts.Communicate(text, voice)
            await asyncio.wait_for(
                communicate.save(str(path)),
                timeout=EDGE_TIMEOUT_S,
            )
            if not path.is_file():
                raise RuntimeError(f"MP3 was not created: {path}")
            return path
        except Exception as e:
            last_err = e
            if attempt < EDGE_RETRIES and _edge_network_error(e):
                wait = 2 ** attempt
                print(f"     retry in {wait}s ({type(e).__name__})")
                await asyncio.sleep(wait)
                continue
            raise

    assert last_err is not None
    raise last_err


async def make_voice(
    text: str,
    path: Path,
    voice: str,
    backend: str,
) -> Path:
    if backend == "offline":
        return make_voice_offline(text, path)

    try:
        return await make_voice_edge(text, path, voice)
    except Exception as e:
        if backend == "edge":
            if _edge_network_error(e):
                _print_edge_help()
            raise
        print(f"     edge failed ({type(e).__name__}), using offline…")
        return make_voice_offline(text, path)


async def generate_many(
    names: list[str],
    voice: str,
    backend: str,
    *,
    force: bool = False,
    prefix: str = "",
) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    ok, skip = 0, 0
    ext = ".wav" if backend == "offline" else ".mp3"

    for name in names:
        stem = f"{prefix}{_safe_filename(name)}" if prefix else _safe_filename(name)
        path = _OUT / f"{stem}{ext}"
        if backend == "auto":
            mp3 = _OUT / f"{stem}.mp3"
            wav = _OUT / f"{stem}.wav"
            if not force and (mp3.is_file() or wav.is_file()):
                print(f"  skip (exists) {stem}")
                skip += 1
                continue
        elif path.is_file() and not force:
            print(f"  skip (exists) {path.name}")
            skip += 1
            continue

        await make_voice(speak_text_for(name), path, voice, backend)
        ok += 1
        if backend == "edge":
            time.sleep(0.2)

    print(f"Done: {ok} created, {skip} skipped → {_OUT.resolve()}")


async def list_voices() -> None:
    import edge_tts

    voices = await edge_tts.list_voices()
    voices = [v for v in voices if v["Locale"].startswith("en")]
    for v in voices:
        print(f"  {v['ShortName']:28}  {v['Gender']:6}  {v['Locale']}")


def print_menu(names: list[str], limit: int = 30) -> None:
    print(f"\nFirst {limit} champions:")
    for i, name in enumerate(names[:limit], 1):
        print(f"  {i:3}. {name}")
    if len(names) > limit:
        print(f"  … and {len(names) - limit} more")
    print("  sample | roster | all | voices | q")


async def interactive_main(names: list[str], voice: str, backend: str, force: bool) -> None:
    print(f"\nChampion TTS  backend={backend}  voice={voice}")
    print("Output → tools/tts_out/")
    print_menu(names)

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        low = line.lower()
        if low in ("q", "quit", "exit"):
            break
        if low == "voices":
            try:
                await list_voices()
            except Exception as e:
                print(f"list-voices failed: {e}")
                _print_edge_help()
            continue
        if low == "sample":
            await generate_many(_SAMPLE_NAMES, voice, backend, force=force)
            continue
        if low == "roster":
            await generate_many(_SAMPLE_ROSTER, voice, backend, force=force, prefix="roster_")
            continue
        if low == "all":
            await generate_many(names, voice, backend, force=force)
            continue
        if line.isdigit():
            idx = int(line) - 1
            if 0 <= idx < len(names):
                n = names[idx]
                p = _OUT / _safe_filename(n)
                await make_voice(speak_text_for(n), p, voice, backend)
            else:
                print(f"Pick 1–{len(names)}")
            continue

        await make_voice(speak_text_for(line), _OUT / _safe_filename(line), voice, backend)

    print("Done.")


def _resolve_backend(args: argparse.Namespace) -> str:
    if args.offline:
        return "offline"
    return args.backend


def main() -> None:
    ap = argparse.ArgumentParser(description="Champion name TTS (edge-tts or pyttsx3)")
    ap.add_argument("--mp3", metavar="TEXT", help="Generate speech for this text")
    ap.add_argument("--out", metavar="PATH", help="Output path")
    ap.add_argument("--voice", default=DEFAULT_VOICE, help=f"edge voice (default {DEFAULT_VOICE})")
    ap.add_argument(
        "--backend",
        choices=("auto", "edge", "offline"),
        default="auto",
        help="auto=edge then offline on failure (default)",
    )
    ap.add_argument("--offline", action="store_true", help="Same as --backend offline")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--roster", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list-voices", action="store_true")
    args = ap.parse_args()

    backend = _resolve_backend(args)
    voice = args.voice

    if args.list_voices:
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            print("Install edge-tts:  pip install edge-tts")
            sys.exit(1)
        try:
            asyncio.run(list_voices())
        except Exception as e:
            _print_edge_help()
            print(f"Error: {e}")
            sys.exit(1)
        return

    if backend in ("auto", "edge"):
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            if backend == "edge":
                print("Install edge-tts:  pip install edge-tts")
                sys.exit(1)
            print("edge-tts not installed; using offline only.")
            backend = "offline"

    if backend == "offline":
        try:
            import pyttsx3  # noqa: F401
        except ImportError:
            print("Install pyttsx3:  pip install pyttsx3")
            sys.exit(1)

    async def _one(text: str, out: Path | None) -> None:
        path = out or (_OUT / _safe_filename(text))
        spoken = speak_text_for(text)
        try:
            result = await make_voice(spoken, path, voice, backend)
            print(f"Open: {result.resolve()}")
        except Exception as e:
            if _edge_network_error(e):
                _print_edge_help()
            raise

    if args.mp3:
        text = args.mp3
        if text.lower() == "roster":
            asyncio.run(generate_many(_SAMPLE_ROSTER, voice, backend, force=args.force, prefix="roster_"))
            return
        out = Path(args.out) if args.out else None
        asyncio.run(_one(text, out))
        return

    if args.sample:
        asyncio.run(generate_many(_SAMPLE_NAMES, voice, backend, force=args.force))
        return

    if args.roster:
        asyncio.run(generate_many(_SAMPLE_ROSTER, voice, backend, force=args.force, prefix="roster_"))
        return

    if args.all:
        names = load_champion_names()
        asyncio.run(generate_many(names, voice, backend, force=args.force))
        return

    names = load_champion_names()
    asyncio.run(interactive_main(names, voice, backend, args.force))


if __name__ == "__main__":
    main()
