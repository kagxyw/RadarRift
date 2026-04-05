from __future__ import annotations

from pathlib import Path
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QCheckBox, QComboBox,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFrame,
    QSpinBox, QDoubleSpinBox, QSlider, QSizePolicy,
    QScrollArea
)

from constants import BG, FG, DIM, ALLY, ENE, ACT

# Absolute paths to arrow SVGs (forward slashes required for Qt stylesheets)
_HERE       = Path(__file__).parent
_ARROW_UP   = (_HERE / "arrow_up.svg"  ).as_posix()
_ARROW_DOWN = (_HERE / "arrow_down.svg").as_posix()


class SectionHeader(QWidget):
    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        label = QLabel(text)
        label.setStyleSheet("""
            color: #7f849c;
            font-size: 11px;
            font-weight: bold;
        """)

        rule = QFrame()
        rule.setFrameShape(QFrame.Shape.HLine)
        rule.setStyleSheet("color: #313244; background: #313244; min-height: 1px;")

        layout.addWidget(label)
        layout.addWidget(rule, 1)


class RosterRow(QWidget):
    def __init__(self, name: str, name_color: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.dot = QLabel("○")
        self.dot.setFixedWidth(18)
        self.dot.setStyleSheet(f"color: {DIM}; font-size: 12px;")

        self.name_lbl = QLabel(name)
        self.name_lbl.setStyleSheet(f"color: {name_color}; font-size: 13px;")

        self.state_lbl = QLabel("off map")
        self.state_lbl.setStyleSheet(f"color: {DIM}; font-size: 13px;")

        layout.addWidget(self.dot)
        layout.addWidget(self.name_lbl)
        layout.addWidget(self.state_lbl, 1)

    def set_on_map(self, on: bool):
        self.dot.setText("●" if on else "○")
        self.dot.setStyleSheet(f"color: {ACT if on else DIM}; font-size: 12px;")
        self.state_lbl.setText("ON MAP" if on else "off map")
        self.state_lbl.setStyleSheet(f"color: {ACT if on else DIM}; font-size: 13px;")


class AppWindow(QWidget):
    def __init__(self):
        super().__init__()

        # state placeholders
        self.roster = None
        self.capture = None
        self.running = False
        self.alert_radius = 0
        self.alert_sound = None
        self._overlay = None
        self._library = None

        self._roster_widgets: dict[str, RosterRow] = {}

        self.setWindowTitle("RadarRift")
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {BG};
                color: {FG};
                font-family: "Segoe UI";
                font-size: 13px;
            }}
            QPushButton {{
                border: none;
                border-radius: 8px;
                padding: 8px 12px;
                font-weight: bold;
            }}
            QComboBox {{
                background: #313244;
                color: {FG};
                border: 1px solid #45475a;
                border-radius: 6px;
                min-width: 48px;
            }}
            QSpinBox, QDoubleSpinBox {{
                background: #313244;
                color: {FG};
                border: 1px solid #45475a;
                border-radius: 6px;
                min-width: 48px;
                min-height: 24px;
            }}
            QSpinBox::up-button, QDoubleSpinBox::up-button {{
                width: 24px;
                height: 14px;
                subcontrol-position: top right;
                subcontrol-origin: border;
                border-left: 1px solid #45475a;
                background: #45475a;
                border-top-right-radius: 6px;
            }}
            QSpinBox::down-button, QDoubleSpinBox::down-button {{
                width: 24px;
                height: 14px;
                subcontrol-position: bottom right;
                subcontrol-origin: border;
                border-left: 1px solid #45475a;
                background: #45475a;
                border-bottom-right-radius: 6px;
            }}
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
                background: #585b70;
            }}
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
                image: url({_ARROW_UP});
                width: 10px;
                height: 6px;
            }}
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
                image: url({_ARROW_DOWN});
                width: 10px;
                height: 6px;
            }}
            QCheckBox {{
                color: {DIM};
            }}
            QLabel.dim {{
                color: {DIM};
            }}
            QFrame.sep {{
                background: #313244;
                min-height: 1px;
                max-height: 1px;
            }}
        """)

        self._build_ui()

    def _make_sep(self) -> QFrame:
        sep = QFrame()
        sep.setObjectName("sep")
        sep.setProperty("class", "sep")
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background: #313244; min-height: 1px; max-height: 1px;")
        return sep

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(8)

        # ── status bar ─────────────────────────────────────────────
        status_bar = QWidget()
        sb = QHBoxLayout(status_bar)
        sb.setContentsMargins(0, 0, 0, 0)
        sb.setSpacing(6)

        self._phase_dot = QLabel("●")
        self._phase_dot.setStyleSheet(f"color: {DIM};")
        self.watch_lbl = QLabel("Waiting for League of Legends…")
        self.watch_lbl.setStyleSheet(f"color: {DIM};")

        self.fps_lbl = QLabel("")
        self.fps_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.fps_lbl.setFixedWidth(70)
        self.fps_lbl.setStyleSheet(f"color: {DIM}; font-weight: bold;")

        sb.addWidget(self._phase_dot)
        sb.addWidget(self.watch_lbl, 1)
        sb.addWidget(self.fps_lbl)

        root.addWidget(status_bar)

        # ── title row ──────────────────────────────────────────────
        title_row = QWidget()
        tl = QHBoxLayout(title_row)
        tl.setContentsMargins(2, 0, 2, 0)
        tl.setSpacing(8)

        title = QLabel("RadarRift")
        title.setStyleSheet("color: #cba6f7; font-size: 22px; font-weight: bold;")

        self.capture_hidden_cb = QCheckBox("Invisible to capture")
        self.capture_hidden_cb.setChecked(True)
        self.capture_hidden_cb.stateChanged.connect(self._apply_capture_exclusion)
        self.capture_hidden_cb.setToolTip(
            "When on, this RadarRift window is excluded from screen capture "
            "(so it does not appear in your own recordings or screenshots).",
        )

        fps_label = QLabel("FPS")
        fps_label.setStyleSheet(f"color: {DIM};")
        fps_label.setToolTip("Target cap for minimap inference / tracking loop.")

        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["1", "5", "15", "30", "Unlimited"])
        self.fps_combo.setCurrentText("30")
        self.fps_combo.setFixedWidth(100)
        self.fps_combo.setToolTip("Maximum frames per second for processing the minimap region.")

        tl.addWidget(title)
        tl.addStretch(1)
        tl.addWidget(self.capture_hidden_cb)
        tl.addWidget(fps_label)
        tl.addWidget(self.fps_combo)

        root.addWidget(title_row)
        root.addWidget(self._make_sep())

        # ── capture ────────────────────────────────────────────────
        root.addWidget(SectionHeader("CAPTURE"))

        cap_row = QWidget()
        cap = QHBoxLayout(cap_row)
        cap.setContentsMargins(0, 0, 0, 0)
        cap.setSpacing(6)

        self.sel_btn = QPushButton("⊞  Region")
        self.sel_btn.setStyleSheet("background: #89b4fa; color: #1e1e2e;")
        self.sel_btn.clicked.connect(self._select_region)
        self.sel_btn.setToolTip(
            "Open a fullscreen overlay: pick bottom-left or bottom-right minimap corner "
            "and square size, then confirm. Defines what area is captured for detection.",
        )

        self.run_btn = QPushButton("▶  Start")
        self.run_btn.setStyleSheet(f"background: {ACT}; color: #1e1e2e;")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._toggle_run)
        self.run_btn.setToolTip(
            "Start or stop live minimap tracking and the in-game overlay if automatic start/stop fails. "
        )

        cap.addWidget(self.sel_btn, 1)
        cap.addWidget(self.run_btn, 1)

        self.region_lbl = QLabel("No region selected")
        self.region_lbl.setStyleSheet(f"color: {DIM}; font-size: 12px;")
        self.region_lbl.setToolTip(
            "Current minimap rectangle: width×height and top-left (x, y) in screen pixels.",
        )

        root.addWidget(cap_row)
        root.addWidget(self.region_lbl)

        death_row = QWidget()
        dl = QHBoxLayout(death_row)
        dl.setContentsMargins(0, 0, 0, 0)
        dl.setSpacing(6)

        death_btn = QPushButton("⊡  Death Strip")
        death_btn.setStyleSheet("background: #585b70; color: #cdd6f4;")
        death_btn.clicked.connect(self._select_death_region)
        death_btn.setToolTip(
            "Fullscreen drag overlay: draw the strip where the death recap / timer appears. "
            "Used when death-panel scanning is enabled.",
        )

        auto_btn = QPushButton("Auto")
        auto_btn.setStyleSheet("background: #45475a; color: #cdd6f4;")
        auto_btn.clicked.connect(self._auto_death_region)
        auto_btn.setToolTip(
            "Clear manual death strip and derive its position from the current minimap region "
            "(built-in layout offset).",
        )

        self.death_region_lbl = QLabel("pending minimap region…")
        self.death_region_lbl.setStyleSheet(f"color: {DIM}; font-size: 12px;")
        self.death_region_lbl.setToolTip("Death-strip capture area: size and position.")

        dl.addWidget(death_btn)
        dl.addWidget(auto_btn)
        dl.addWidget(self.death_region_lbl, 1)

        root.addWidget(death_row)
        root.addWidget(self._make_sep())

        # ── alert ─────────────────────────────────────────────────
        root.addWidget(SectionHeader("ALERT"))

        radius_row = QWidget()
        rl = QHBoxLayout(radius_row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)

        radius_btn = QPushButton("Set Radius")
        radius_btn.setStyleSheet("background: #fab387; color: #1e1e2e;")
        radius_btn.clicked.connect(self._select_alert_radius)
        radius_btn.setToolTip(
            "Open a dialog to set the danger ring radius around your champion on the minimap. "
            "When an enemy enters the ring, an alert can play (if sound is set). 0 = off.",
        )

        self._radius_lbl = QLabel("Off")
        self._radius_lbl.setStyleSheet(f"color: {DIM}; font-weight: bold;")
        self._radius_lbl.setFixedWidth(80)
        self._radius_lbl.setToolTip("Current alert radius in minimap pixels (or Off).")

        rl.addWidget(radius_btn, 1)
        rl.addWidget(self._radius_lbl)

        root.addWidget(radius_row)

        sound_row = QWidget()
        sl = QHBoxLayout(sound_row)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(8)

        sound_btn = QPushButton("♪  Sound")
        sound_btn.setStyleSheet("background: #89dceb; color: #1e1e2e;")
        sound_btn.clicked.connect(self._pick_alert_sound)
        sound_btn.setToolTip("Choose a WAV (or compatible) file played when the alert radius triggers.")

        self._sound_lbl = QLabel("need file")
        self._sound_lbl.setStyleSheet(f"color: {DIM}; font-size: 12px;")
        self._sound_lbl.setToolTip("Filename of the current alert sound.")

        vol_lbl = QLabel("Vol:")
        vol_lbl.setStyleSheet(f"color: {DIM};")
        vol_lbl.setToolTip("Alert sound volume.")

        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        self.volume_slider.setFixedWidth(90)
        self.volume_slider.valueChanged.connect(
            lambda v: self._vol_lbl.setText(f"{v}%")
        )
        self.volume_slider.setToolTip("Playback volume for the radius alert sound.")

        self._vol_lbl = QLabel("80%")
        self._vol_lbl.setStyleSheet(f"color: {DIM};")
        self._vol_lbl.setFixedWidth(36)

        test_btn = QPushButton("▶")
        test_btn.setStyleSheet("background: #45475a; color: #cdd6f4;")
        test_btn.setFixedWidth(34)
        test_btn.clicked.connect(self._test_alert_sound)
        test_btn.setToolTip("Play the selected alert sound once to check volume and file.")

        sl.addWidget(sound_btn)
        sl.addWidget(self._sound_lbl, 1)
        sl.addWidget(vol_lbl)
        sl.addWidget(self.volume_slider)
        sl.addWidget(self._vol_lbl)
        sl.addWidget(test_btn)

        root.addWidget(sound_row)

        cd_row = QWidget()
        cl = QHBoxLayout(cd_row)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(8)

        _cd_lbl = QLabel("Alert CD:")
        _cd_lbl.setToolTip(
            "Minimum seconds between alert sounds when enemies keep entering the radius.",
        )
        cl.addWidget(_cd_lbl)
        self.cooldown_spin = QDoubleSpinBox()
        self.cooldown_spin.setRange(1.0, 120.0)
        self.cooldown_spin.setSingleStep(1.0)
        self.cooldown_spin.setValue(10.0)
        self.cooldown_spin.setFixedWidth(80)
        self.cooldown_spin.setToolTip(
            "Cooldown between radius alerts (seconds). Prevents sound spam.",
        )

        cl.addWidget(self.cooldown_spin)
        cl.addWidget(QLabel("s"))

        cl.addSpacing(12)

        _off_lbl = QLabel("Draw marker after:")
        _off_lbl.setToolTip(
            "Seconds without seeing a champion on the minimap before showing off-map ghost marker.",
        )
        cl.addWidget(_off_lbl)
        self.off_timeout_spin = QDoubleSpinBox()
        self.off_timeout_spin.setRange(0.5, 60.0)
        self.off_timeout_spin.setSingleStep(0.5)
        self.off_timeout_spin.setValue(5.0)
        self.off_timeout_spin.setFixedWidth(80)
        self.off_timeout_spin.setToolTip(
            "Off-map / ghost overlay timing: higher = wait longer before treating as missing.",
        )

        cl.addWidget(self.off_timeout_spin)
        cl.addWidget(QLabel("s"))
        cl.addStretch(1)

        root.addWidget(cd_row)
        root.addWidget(self._make_sep())

        # ── overlay ───────────────────────────────────────────────
        root.addWidget(SectionHeader("OVERLAY"))

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        _SW = 80   # fixed width for all overlay spinboxes

        self.champ_size_spin = QSpinBox()
        self.champ_size_spin.setRange(8, 64)
        self.champ_size_spin.setSingleStep(4)
        self.champ_size_spin.setValue(36)
        self.champ_size_spin.setFixedWidth(_SW)
        self.champ_size_spin.setToolTip(
            "Pixel size of off-map champion icons drawn on the overlay.",
        )

        self.ghost_alpha_spin = QSpinBox()
        self.ghost_alpha_spin.setRange(0, 100)
        self.ghost_alpha_spin.setSingleStep(5)
        self.ghost_alpha_spin.setValue(50)
        self.ghost_alpha_spin.setFixedWidth(_SW)
        self.ghost_alpha_spin.setToolTip("Opacity of off-map champion portraits (0–100%).")

        self.dot_size_spin = QSpinBox()
        self.dot_size_spin.setRange(0, 16)
        self.dot_size_spin.setValue(5)
        self.dot_size_spin.setFixedWidth(_SW)
        self.dot_size_spin.setToolTip(
            "Radius of on-map detection dots (0 hides dots). Applies when overlay markers are visible.",
        )

        self.timer_size_spin = QSpinBox()
        self.timer_size_spin.setRange(4, 32)
        self.timer_size_spin.setValue(10)
        self.timer_size_spin.setFixedWidth(_SW)
        self.timer_size_spin.setToolTip("Font size for the seconds-since-seen label under ghost icons.")

        self.timer_alpha_spin = QSpinBox()
        self.timer_alpha_spin.setRange(0, 100)
        self.timer_alpha_spin.setSingleStep(5)
        self.timer_alpha_spin.setValue(100)
        self.timer_alpha_spin.setFixedWidth(_SW)
        self.timer_alpha_spin.setToolTip("Opacity of the timer text (0–100%).")

        self.arrow_size_spin = QSpinBox()
        self.arrow_size_spin.setRange(0, 10)
        self.arrow_size_spin.setValue(2)
        self.arrow_size_spin.setFixedWidth(_SW)
        self.arrow_size_spin.setToolTip(
            "Thickness / scale of the direction arrow on off-map ghosts (0 hides arrow line).",
        )

        self.arrow_alpha_spin = QSpinBox()
        self.arrow_alpha_spin.setRange(0, 100)
        self.arrow_alpha_spin.setSingleStep(5)
        self.arrow_alpha_spin.setValue(100)
        self.arrow_alpha_spin.setFixedWidth(_SW)
        self.arrow_alpha_spin.setToolTip("Reserved / paired with arrow styling (opacity %).")

        self.ring_thickness_spin = QSpinBox()
        self.ring_thickness_spin.setRange(0, 8)
        self.ring_thickness_spin.setValue(2)
        self.ring_thickness_spin.setFixedWidth(_SW)
        self.ring_thickness_spin.setToolTip(
            "Line thickness of the alert radius ring on the minimap (0 hides the ring).",
        )

        _g_champ = QLabel("Champ:")
        _g_champ.setToolTip(self.champ_size_spin.toolTip())
        grid.addWidget(_g_champ, 0, 0)
        grid.addWidget(self.champ_size_spin, 0, 1)
        _g_galpha_l = QLabel("Opacity:")
        _g_galpha_l.setToolTip(self.ghost_alpha_spin.toolTip())
        grid.addWidget(_g_galpha_l, 0, 2)
        grid.addWidget(self.ghost_alpha_spin, 0, 3)
        _g_pct0 = QLabel("%")
        _g_pct0.setToolTip("Unit for ghost portrait opacity (0 = invisible, 100 = opaque).")
        grid.addWidget(_g_pct0, 0, 4)
        _g_dot = QLabel("Dot:")
        _g_dot.setToolTip(self.dot_size_spin.toolTip())
        grid.addWidget(_g_dot, 0, 5)
        grid.addWidget(self.dot_size_spin, 0, 6)

        _g_timer = QLabel("Timer:")
        _g_timer.setToolTip(self.timer_size_spin.toolTip())
        grid.addWidget(_g_timer, 1, 0)
        grid.addWidget(self.timer_size_spin, 1, 1)
        _g_talpha_l = QLabel("Opacity:")
        _g_talpha_l.setToolTip(self.timer_alpha_spin.toolTip())
        grid.addWidget(_g_talpha_l, 1, 2)
        grid.addWidget(self.timer_alpha_spin, 1, 3)
        _g_pct1 = QLabel("%")
        _g_pct1.setToolTip("Unit for timer text opacity.")
        grid.addWidget(_g_pct1, 1, 4)

        _g_arrow = QLabel("Arrow:")
        _g_arrow.setToolTip(self.arrow_size_spin.toolTip())
        grid.addWidget(_g_arrow, 2, 0)
        grid.addWidget(self.arrow_size_spin, 2, 1)
        _g_aalpha_l = QLabel("Opacity:")
        _g_aalpha_l.setToolTip(self.arrow_alpha_spin.toolTip())
        grid.addWidget(_g_aalpha_l, 2, 2)
        grid.addWidget(self.arrow_alpha_spin, 2, 3)
        _g_pct2 = QLabel("%")
        _g_pct2.setToolTip("Unit for arrow opacity.")
        grid.addWidget(_g_pct2, 2, 4)
        _g_ring = QLabel("Radius Thickness:")
        _g_ring.setToolTip(self.ring_thickness_spin.toolTip())
        grid.addWidget(_g_ring, 2, 5)
        grid.addWidget(self.ring_thickness_spin, 2, 6)

        grid_wrap = QWidget()
        grid_wrap.setLayout(grid)
        root.addWidget(grid_wrap)

        # All minimap overlay drawing: ghosts, dots, alert radius (see App for logic)
        ghost_row = QWidget()
        gr = QHBoxLayout(ghost_row)
        gr.setContentsMargins(0, 4, 0, 0)
        gr.setSpacing(8)

        _om_lbl = QLabel("Overlay markers:")
        _om_lbl.setToolTip(
            "Controls visibility of the whole minimap overlay layer: ghosts, dots, and alert ring.",
        )
        gr.addWidget(_om_lbl)
        self.ghost_marker_mode_combo = QComboBox()
        self.ghost_marker_mode_combo.setMinimumWidth(200)
        for _label, _data in (
            ("Always show", "always"),
            ("Never show", "never"),
            ("While holding key", "hold"),
            ("Toggle with key", "toggle"),
        ):
            self.ghost_marker_mode_combo.addItem(_label, _data)
        gr.addWidget(self.ghost_marker_mode_combo, 1)
        self.ghost_marker_mode_combo.setToolTip(
            "Always: draw whenever tracking. Never: hide overlay art. "
            "Hold: show while a key is held. Toggle: key press flips show/hide.",
        )

        self.ghost_marker_key_btn = QPushButton("Set key…")
        self.ghost_marker_key_btn.setStyleSheet(
            "background: #45475a; color: #cdd6f4; min-width: 120px;"
        )
        self.ghost_marker_key_btn.setToolTip(
            "Key used for Hold or Toggle overlay modes. "
            "Applies to off-map icons, on-map dots, and the alert radius. "
            "Uses Windows key state in-game. Click, then press a key.",
        )
        gr.addWidget(self.ghost_marker_key_btn)

        root.addWidget(ghost_row)

        # preview_btn = QPushButton("Preview Overlay")
        # preview_btn.setStyleSheet("background: #cba6f7; color: #1e1e2e;")
        # preview_btn.clicked.connect(self._preview_overlay)
        # root.addWidget(preview_btn)

        root.addWidget(self._make_sep())

        # ── roster ────────────────────────────────────────────────
        root.addWidget(SectionHeader("ROSTER"))

        manual_btn = QPushButton("Enter Manually")
        manual_btn.setStyleSheet("background: #cba6f7; color: #1e1e2e;")
        manual_btn.clicked.connect(self._enter_manually)
        manual_btn.setToolTip(
            "Pick all 10 champions from lists when automatic loading-screen detection fails.",
        )
        root.addWidget(manual_btn)

        self._roster_container = QWidget()
        self._roster_layout = QVBoxLayout(self._roster_container)
        self._roster_layout.setContentsMargins(0, 0, 0, 0)
        self._roster_layout.setSpacing(4)

        root.addWidget(self._roster_container, 1)

        self.resize(620, 720)

    # ── roster display ────────────────────────────────────────────

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout(child_layout)

    def _build_roster_display(self) -> None:
        self._clear_layout(self._roster_layout)
        self._roster_widgets.clear()

        if not self.roster:
            lbl = QLabel("No champions loaded yet.")
            lbl.setStyleSheet(f"color: {DIM}; font-style: italic;")
            self._roster_layout.addWidget(lbl)
            self._roster_layout.addStretch(1)
            return

        def add_section(title: str, champs, name_color: str):
            hdr = QLabel(title)
            hdr.setStyleSheet(f"color: {DIM}; font-size: 11px; font-weight: bold;")
            self._roster_layout.addWidget(hdr)

            for c in champs:
                row = RosterRow(c.name, name_color)
                self._roster_layout.addWidget(row)
                self._roster_widgets[c.name] = row
                self._after_roster_row(c, row)

        add_section("— Player", [self.roster.player], "#f9e2af")
        add_section("— Allies", self.roster.allies, ALLY)
        add_section("— Enemies", self.roster.enemies, ENE)
        self._roster_layout.addStretch(1)

    def _after_roster_row(self, c, row) -> None:
        """Hook called after each champion row is added. Override in App."""
        pass

    def _refresh_roster_display(self) -> None:
        if not self.roster:
            return
        from champions import STATUS_ON_MAP

        for c in [self.roster.player] + self.roster.allies + self.roster.enemies:
            row = self._roster_widgets.get(c.name)
            if row is None:
                continue
            row.set_on_map(c.status == STATUS_ON_MAP)

    # ── status ────────────────────────────────────────────────────

    _PHASE_COLORS: dict[str, str] = {
        "wait": DIM,
        "scan": "#fab387",
        "id":   "#cba6f7",
        "run":  ACT,
        "stop": "#f38ba8",
    }

    def _set_status(self, msg: str, phase: str = "wait") -> None:
        color = self._PHASE_COLORS.get(phase, DIM)
        self.watch_lbl.setText(msg)
        self.watch_lbl.setStyleSheet(f"color: {color};")
        self._phase_dot.setStyleSheet(f"color: {color};")

    # ── placeholders for your existing logic ─────────────────────
    def _apply_capture_exclusion(self):
        hidden = self.capture_hidden_cb.isChecked()
        # move your existing Win32 affinity logic here
        print("capture hidden:", hidden)

    def _select_region(self):
        print("select region")

    def _start(self):
        print("start")

    def _stop(self):
        print("stop")

    def _toggle_run(self):
        print("toggle run")

    def _select_death_region(self):
        print("select death strip")

    def _auto_death_region(self):
        print("auto death strip")

    def _select_alert_radius(self):
        print("select alert radius")

    def _pick_alert_sound(self):
        print("pick sound")

    def _test_alert_sound(self):
        print("test sound")

    def _preview_overlay(self):
        print("preview overlay")

    def _enter_manually(self):
        print("manual entry")