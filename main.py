"""RadarRift — League of Legends minimap tracker."""

import os
import sys

# Run ONNX path from source (same as build):  python main.py --onnx
if "--onnx" in sys.argv:
    os.environ["RADARRIFT_BACKEND"] = "onnx"
    sys.argv.remove("--onnx")

os.environ.setdefault("YOLO_AUTOINSTALL", "0")
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.window=false")
import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

# Single QApplication for the whole process — startup_check, App, and overlay
# all reuse this instance via QApplication.instance().
from PyQt6.QtWidgets import QApplication
_qt_app = QApplication(sys.argv)

# from startup_check import ensure_ready
from app import App

if __name__ == "__main__":
    # if ensure_ready():
    App().run()
