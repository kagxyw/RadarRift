# -*- mode: python ; coding: utf-8 -*-
# Build deps: pip install -r requirements-build.txt
#
# Run from repository root (close dist\RadarRift\RadarRift.exe if it is running):
#   python -m PyInstaller packaging/RadarRift.spec --noconfirm
#
from PyInstaller.utils.hooks import collect_all, collect_data_files

import os
import shutil
import subprocess
import sys as _sys

_PACK = os.path.dirname(os.path.abspath(SPECPATH))


def _project_root(pack_dir: str) -> str:
    """RadarRift root = directory that contains main.py (parent of packaging/)."""
    d = os.path.abspath(pack_dir)
    for _ in range(6):
        if os.path.isfile(os.path.join(d, "main.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    raise RuntimeError(
        "Could not find project root (main.py). Run PyInstaller from the RadarRift "
        "repo root, e.g.  python -m PyInstaller packaging/RadarRift.spec"
    )


_ROOT = _project_root(_PACK)

# Export YOLO .pt → ONNX into cache/ when weights are present (requires ultralytics).
print("--- tools.build_bundle_onnx (optional) ---")
_bx = subprocess.run(
    [_sys.executable, "-m", "tools.build_bundle_onnx"],
    cwd=_ROOT,
)
if _bx.returncode != 0:
    print(
        "WARNING: ONNX export exited with code %s — cache/ may lack .onnx files.\n"
        "  Install ultralytics, place .pt in cache/ or train runs/, or add .onnx manually."
        % (_bx.returncode,)
    )

# Exclude minimap_yolo11n.pt from the bundled cache — users download it on first run
_cache_src = os.path.join(_ROOT, "cache")
_cache_tmp = os.path.join(_ROOT, "cache_bundle_tmp")
if os.path.exists(_cache_tmp):
    shutil.rmtree(_cache_tmp)
# Only copy runtime-needed files; skip skin portrait JPEGs (only needed to
# build matrices, not at runtime) to keep the dist folder small and fast to zip.
_KEEP_EXTS = {".npy", ".json", ".onnx", ".png"}


def _cache_ignore(src, names):
    ignored = set()
    for n in names:
        ext = os.path.splitext(n)[1].lower()
        full = os.path.join(src, n)
        if os.path.isfile(full) and ext not in _KEEP_EXTS:
            ignored.add(n)
    return ignored


shutil.copytree(_cache_src, _cache_tmp, ignore=_cache_ignore)
# Exclude .pt weight files — only ONNX files are bundled for inference
for _excl in (
    "minimap_yolo11n.pt",
    "minimap_detection.pt",
    "minimap_detection.onnx",
    "splash_detection.pt",
    "champion_yolo11n.pt",
):
    _p = os.path.join(_cache_tmp, _excl)
    if os.path.exists(_p):
        os.remove(_p)

_assets = os.path.join(_ROOT, "assets")

datas = [(_cache_tmp, "cache")]
_tts_out = os.path.join(_ROOT, "tts_out")
if os.path.isdir(_tts_out):
    datas.append((_tts_out, "tts_out"))
    print("Bundling tts_out/ (%d mp3)" % len(
        [f for f in os.listdir(_tts_out) if f.lower().endswith(".mp3")]))
else:
    print("WARNING: tts_out/ not found — Read name alerts will need MP3s beside the exe.")
datas += [
    (os.path.join(_assets, "屏幕截图 2026-03-07 060822.png"), "."),
    (os.path.join(_assets, "arrow_up.svg"), "."),
    (os.path.join(_assets, "arrow_down.svg"), "."),
    (os.path.join(_assets, "kayn_slay_square.png"), "."),
    (os.path.join(_assets, "kayn_ass_square.png"), "."),
    (os.path.join(_assets, "kayle_square_lvl11.png"), "."),
]
datas += collect_data_files("setuptools", include_py_files=False)
binaries = []
hiddenimports = ["PIL._tkinter_finder"]

# ── onnxruntime: let collect_all handle binaries; the runtime hook
# (rthook_onnxruntime.py) registers onnxruntime/capi via add_dll_directory
# before the pyd is imported, which is the correct Python 3.8+ fix.
tmp_ret = collect_all("onnxruntime")
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]
tmp_ret = collect_all("dxcam")
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]
# Fallback screen capture + alert sounds (requirements.txt)
tmp_ret = collect_all("mss")
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]
tmp_ret = collect_all("pygame")
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

a = Analysis(
    [os.path.join(_ROOT, "main.py")],
    pathex=[_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports
    + [
        "tools",
        "tools.startup_check",
        "tools.rebuild_cache",
        "backend",
        "yolo_champion",
        "death_panel",
        "overlay_qt",
        "tracker",
        "capture",
        "champions",
        "constants",
        "match_start",
        "loading_keys",
        "wiki_loading_keys",
        "splash_model",
        "onnx_model",
        "select_minimap",
        "win_lol",
        "ui",
        "app",
        "alert_audio",
        "PyQt6",
        "PyQt6.QtCore",
        "PyQt6.QtGui",
        "PyQt6.QtWidgets",
        "PyQt6.sip",
        "mss",
        "pygame",
        "pygame.mixer",
    ],
    hookspath=[],
    hooksconfig={},
    # Anchor to _ROOT — SPECPATH dirname can resolve to repo root on some PyInstaller versions.
    runtime_hooks=[os.path.join(_ROOT, "packaging", "rthook_onnxruntime.py")],
    excludes=[
        "torch",
        "torchvision",
        "torchaudio",
        "ultralytics",
        "bitsandbytes",
        "datasets",
        "transformers",
        "accelerate",
        "diffusers",
        "tokenizers",
        "huggingface_hub",
        "onnxruntime.training",
        "nltk",
        "sklearn",
        "PyQt5",
        "PySide2",
        "PySide6",
        "matplotlib",
        "pandas",
        "scipy",
        "IPython",
        "jupyter",
        "notebook",
        "nbconvert",
        "altair",
        "bokeh",
        "panel",
        "holoviews",
        "plotly",
        "pyarrow",
        "tables",
        "h5py",
        "sqlalchemy",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RadarRift",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="RadarRift",
    overwrite_spec=True,
)

# Clean up the temporary cache copy used during bundling
if os.path.exists(_cache_tmp):
    shutil.rmtree(_cache_tmp)
