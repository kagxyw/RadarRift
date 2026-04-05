"""
PyInstaller runtime hook – pre-loads onnxruntime native DLLs before any
Python module import touches onnxruntime_pybind11_state.pyd.

Three mechanisms are used (belt-and-suspenders):
  1. os.add_dll_directory()  – Python 3.8+ DLL search path
  2. PATH prepend            – legacy DLL search fallback
  3. ctypes.WinDLL()         – explicit pre-load so the pyd finds its deps
"""
import ctypes
import os
import sys

if hasattr(sys, "_MEIPASS"):
    _capi = os.path.join(sys._MEIPASS, "onnxruntime", "capi")
    _dirs = [_capi, sys._MEIPASS]

    for _d in _dirs:
        if os.path.isdir(_d):
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(_d)
                except OSError:
                    pass

    os.environ["PATH"] = _capi + os.pathsep + os.environ.get("PATH", "")

    for _dll in (
        "vcruntime140.dll",
        "vcruntime140_1.dll",
        "msvcp140.dll",
        "onnxruntime_providers_shared.dll",
        "onnxruntime.dll",
        "DirectML.dll",
    ):
        _p = os.path.join(_capi, _dll)
        if not os.path.isfile(_p):
            _p = os.path.join(sys._MEIPASS, _dll)
        if os.path.isfile(_p):
            try:
                ctypes.WinDLL(_p)
            except OSError:
                pass
