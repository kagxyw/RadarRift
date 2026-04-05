# -*- mode: python ; coding: utf-8 -*-
# Build deps: pip install -r requirements-build.txt
from PyInstaller.utils.hooks import collect_all, collect_data_files

import os, shutil
# Exclude minimap_yolo11n.pt from the bundled cache — users download it on first run
_cache_src = 'cache'
_cache_tmp = 'cache_bundle_tmp'
if os.path.exists(_cache_tmp):
    shutil.rmtree(_cache_tmp)
# Only copy runtime-needed files; skip skin portrait JPEGs (only needed to
# build matrices, not at runtime) to keep the dist folder small and fast to zip.
_KEEP_EXTS = {'.npy', '.json', '.onnx', '.png'}
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
for _excl in ('minimap_yolo11n.pt', 'minimap_detection.pt',
              'minimap_detection.onnx', 'splash_detection.pt',
              'champion_yolo11n.pt'):
    _p = os.path.join(_cache_tmp, _excl)
    if os.path.exists(_p):
        os.remove(_p)

datas = [(_cache_tmp, 'cache')]
datas += [('屏幕截图 2026-03-07 060822.png', '.')]
datas += [('arrow_up.svg',           '.')]
datas += [('arrow_down.svg',         '.')]
datas += [('kayn_slay_square.png',   '.')]
datas += [('kayn_ass_square.png',    '.')]
datas += [('kayle_square_lvl11.png', '.')]
datas += collect_data_files('setuptools', include_py_files=False)
binaries = []
hiddenimports = ['PIL._tkinter_finder']

# ── onnxruntime: let collect_all handle binaries; the runtime hook
# (rthook_onnxruntime.py) registers onnxruntime/capi via add_dll_directory
# before the pyd is imported, which is the correct Python 3.8+ fix.
tmp_ret = collect_all('onnxruntime')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('dxcam')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
# Fallback screen capture + alert sounds (requirements.txt)
tmp_ret = collect_all('mss')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pygame')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + [
        'startup_check', 'rebuild_cache',
        'backend', 'yolo_champion',
        'death_panel', 'overlay_qt',
        'tracker', 'capture', 'champions', 'constants',
        'match_start', 'splash_model', 'onnx_model',
        'select_minimap', 'win_lol',
        'ui', 'app',
        'PyQt6', 'PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets',
        'PyQt6.sip',
        'mss', 'pygame', 'pygame.mixer',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['rthook_onnxruntime.py'],
    excludes=['torch', 'torchvision', 'torchaudio', 'ultralytics',
              'bitsandbytes', 'datasets', 'transformers', 'accelerate',
              'diffusers', 'tokenizers', 'huggingface_hub',
              'onnxruntime.training',
              'nltk', 'sklearn', 'PyQt5', 'PySide2', 'PySide6',
              'matplotlib', 'pandas', 'scipy', 'IPython', 'jupyter',
              'notebook', 'nbconvert', 'altair', 'bokeh', 'panel',
              'holoviews', 'plotly', 'pyarrow', 'tables', 'h5py', 'sqlalchemy'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RadarRift',
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
    name='RadarRift',
    overwrite_spec=True,
)

# Clean up the temporary cache copy used during bundling
if os.path.exists(_cache_tmp):
    shutil.rmtree(_cache_tmp)
