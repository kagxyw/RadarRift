#!/usr/bin/env python3
"""Remake ONNX files from .pt weights. Run from project root.

Delegates to tools.build_bundle_onnx (discovery + export).

Usage:
  python -m tools.remake_onnx        # export only if .onnx missing
  python -m tools.remake_onnx --force
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.build_bundle_onnx import main

if __name__ == "__main__":
    raise SystemExit(main())
