#!/usr/bin/env python3
import sys
import json
import time
import numpy as np

# ===================== Qt =====================
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGroupBox, QFormLayout, QSlider, QLabel, QPushButton,
    QColorDialog, QComboBox, QCheckBox, QFrame, QSizePolicy,
    QScrollArea, QFileDialog
)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import QTimer, Qt

# ===================== OpenGL =====================
from OpenGL.GL import *
from OpenGL.GLU import *

# ===================== Audio =====================
import sounddevice as sd
from scipy.signal import butter, lfilter

# ===================== Audio Globals =====================
RATE = 44100
BUFFER = 1024

beat_energy = 0.0
vocal_energy = 0.0


def bandpass(low, high, fs, data):
    b, a = butter(2, [low / (fs / 2), high / (fs / 2)], btype="band")
    return lfilter(b, a, data)


def audio_callback(indata, frames, time_info, status):
    """
    Hybrid energy metric:
      energy = 0.7 * RMS + 0.3 * Peak
    Then EMA-smoothed.
    """
    global beat_energy, vocal_energy

    mono = np.mean(indata, axis=1)
    bass = bandpass(20, 150, RATE, mono)
    vocals = bandpass(300, 3000, RATE, mono)

    bass_rms = float(np.sqrt(np.mean(bass ** 2)))
    bass_peak = float(np.max(np.abs(bass)) + 1e-12)
    bass_e = 0.7 * bass_rms + 0.3 * bass_peak

    voc_rms = float(np.sqrt(np.mean(vocals ** 2)))
    voc_peak = float(np.max(np.abs(vocals)) + 1e-12)
    voc_e = 0.7 * voc_rms + 0.3 * voc_peak

    beat_energy = 0.9 * beat_energy + 0.1 * bass_e
    vocal_energy = 0.9 * vocal_energy + 0.1 * voc_e


def clamp(x, a, b):
    return max(a, min(b, x))


def smoothstep(edge0, edge1, x):
    if edge0 == edge1:
        return 0.0
    t = (x - edge0) / (edge1 - edge0)
    t = clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ===================== OpenGL Widget =====================
class SphereGL(QOpenGLWidget):
    def __init__(self):
        super().__init__()

        self.time_offset = 0.0
        self.camera_angle = 0.0

        # Ensure GL widget doesn't steal UI interaction in fullscreen
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Audio sensitivity
        self.beat_gain = 6.0
        self.vocal_gain = 4.0

        # Motion
        self.flow_speed = 0.015
        self.camera_orbit = False

        # Zoom (camera distance)
        self.camera_dist = 4.0

        # Model translation (positioning)
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_z = 0.0

        # Rotation system: constant + audio-driven + smooth random drift
        self.spin_base = 0.45
        self.spin_audio = 1.65
        self.axis_wander = 0.28
        self.jitter = 0.10

        self.rot_x = 0.0
        self.rot_y = 0.0
        self.rot_z = 0.0

        self.axis_x = True
        self.axis_y = True
        self.axis_z = True

        # Colors
        self.mesh_color = [0.4, 0.7, 1.0]
        self.wire_color = [1.0, 1.0, 1.0]
        self.wire_alpha = 0.35
        self.wire_radius = 1.28

        # Mode
        self.flow_mode = "wormhole"  # "sphere" or "wormhole"

        # Wormhole (contained) params
        self.wh_tunnel_radius = 0.18
        self.wh_mouth_start = 0.60
        self.wh_softness = 0.45
        self.wh_twist = 1.10
        self.wh_strength = 1.00

        # Rendering / resolution (structural)
        self.mesh_layers = 120
        self.mesh_segments = 170
        self.wire_rings = 24
        self.wire_segments = 36

        # Auto quality (adaptive rendering)
        self.auto_quality = True
        self.target_fps = 60
        self._fps_ema = 60.0
        self._last_ts = time.perf_counter()
        self._quality_cooldown = 0.0

        # Hard bounds (don’t let it go crazy)
        self._min_layers, self._max_layers = 60, 240
        self._min_segments, self._max_segments = 80, 300
        self._min_wire_rings, self._max_wire_rings = 10, 80
        self._min_wire_segments, self._max_wire_segments = 12, 120

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(16)

    # ---------- State Save/Load ----------
    def get_state(self) -> dict:
        return {
            "beat_gain": self.beat_gain,
            "vocal_gain": self.vocal_gain,
            "flow_speed": self.flow_speed,
            "camera_orbit": self.camera_orbit,
            "camera_dist": self.camera_dist,
            "pos_x": self.pos_x,
            "pos_y": self.pos_y,
            "pos_z": self.pos_z,
            "spin_base": self.spin_base,
            "spin_audio": self.spin_audio,
            "axis_wander": self.axis_wander,
            "jitter": self.jitter,
            "axis_x": self.axis_x,
            "axis_y": self.axis_y,
            "axis_z": self.axis_z,
            "mesh_color": list(self.mesh_color),
            "wire_color": list(self.wire_color),
            "wire_alpha": self.wire_alpha,
            "wire_radius": self.wire_radius,
            "flow_mode": self.flow_mode,
            "wh_tunnel_radius": self.wh_tunnel_radius,
            "wh_mouth_start": self.wh_mouth_start,
            "wh_softness": self.wh_softness,
            "wh_twist": self.wh_twist,
            "wh_strength": self.wh_strength,
            "mesh_layers": int(self.mesh_layers),
            "mesh_segments": int(self.mesh_segments),
            "wire_rings": int(self.wire_rings),
            "wire_segments": int(self.wire_segments),
            "auto_quality": self.auto_quality,
            "target_fps": int(self.target_fps),
        }

    def apply_state(self, s: dict) -> None:
        # Use .get with current defaults so older preset files still work.
        self.beat_gain = float(s.get("beat_gain", self.beat_gain))
        self.vocal_gain = float(s.get("vocal_gain", self.vocal_gain))
        self.flow_speed = float(s.get("flow_speed", self.flow_speed))
        self.camera_orbit = bool(s.get("camera_orbit", self.camera_orbit))
        self.camera_dist = float(s.get("camera_dist", self.camera_dist))

        self.pos_x = float(s.get("pos_x", self.pos_x))
        self.pos_y = float(s.get("pos_y", self.pos_y))
        self.pos_z = float(s.get("pos_z", self.pos_z))

        self.spin_base = float(s.get("spin_base", self.spin_base))
        self.spin_audio = float(s.get("spin_audio", self.spin_audio))
        self.axis_wander = float(s.get("axis_wander", self.axis_wander))
        self.jitter = float(s.get("jitter", self.jitter))

        self.axis_x = bool(s.get("axis_x", self.axis_x))
        self.axis_y = bool(s.get("axis_y", self.axis_y))
        self.axis_z = bool(s.get("axis_z", self.axis_z))

        mc = s.get("mesh_color", self.mesh_color)
        wc = s.get("wire_color", self.wire_color)
        if isinstance(mc, (list, tuple)) and len(mc) == 3:
            self.mesh_color = [float(mc[0]), float(mc[1]), float(mc[2])]
        if isinstance(wc, (list, tuple)) and len(wc) == 3:
            self.wire_color = [float(wc[0]), float(wc[1]), float(wc[2])]

        self.wire_alpha = float(s.get("wire_alpha", self.wire_alpha))
        self.wire_radius = float(s.get("wire_radius", self.wire_radius))
        self.flow_mode = str(s.get("flow_mode", self.flow_mode))

        self.wh_tunnel_radius = float(s.get("wh_tunnel_radius", self.wh_tunnel_radius))
        self.wh_mouth_start = float(s.get("wh_mouth_start", self.wh_mouth_start))
        self.wh_softness = float(s.get("wh_softness", self.wh_softness))
        self.wh_twist = float(s.get("wh_twist", self.wh_twist))
        self.wh_strength = float(s.get("wh_strength", self.wh_strength))

        self.mesh_layers = int(clamp(int(s.get("mesh_layers", self.mesh_layers)), self._min_layers, self._max_layers))
        self.mesh_segments = int(clamp(int(s.get("mesh_segments", self.mesh_segments)), self._min_segments, self._max_segments))
        self.wire_rings = int(clamp(int(s.get("wire_rings", self.wire_rings)), self._min_wire_rings, self._max_wire_rings))
        self.wire_segments = int(clamp(int(s.get("wire_segments", self.wire_segments)), self._min_wire_segments, self._max_wire_segments))

        self.auto_quality = bool(s.get("auto_quality", self.auto_quality))
        self.target_fps = int(clamp(int(s.get("target_fps", self.target_fps)), 15, 240))

        self.update()

    # ---------- OpenGL ----------
    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glEnable(GL_LINE_SMOOTH)
        glClearColor(0.03, 0.05, 0.1, 1)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / max(h, 1), 0.1, 100)
        glMatrixMode(GL_MODELVIEW)

    def _auto_quality_tick(self):
        """
        Adaptive quality: keep the look sharp but avoid frame drops.
        Strategy:
          - track EMA FPS
          - if FPS < target*0.92 -> reduce detail (segments/layers)
          - if FPS > target*1.12 -> increase detail
          - cooldown so it doesn’t “thrash”
        """
        if not self.auto_quality:
            return

        now = time.perf_counter()
        dt = now - self._last_ts
        self._last_ts = now

        if dt <= 0:
            return

        fps = 1.0 / dt
        self._fps_ema = 0.92 * self._fps_ema + 0.08 * fps

        # cooldown (seconds)
        self._quality_cooldown = max(0.0, self._quality_cooldown - dt)
        if self._quality_cooldown > 0.0:
            return

        tgt = float(self.target_fps)
        lo = tgt * 0.92
        hi = tgt * 1.12

        # Adjust in small steps (keeps visuals stable)
        if self._fps_ema < lo:
            # downshift
            self.mesh_layers = max(self._min_layers, int(self.mesh_layers) - 6)
            self.mesh_segments = max(self._min_segments, int(self.mesh_segments) - 10)
            self.wire_rings = max(self._min_wire_rings, int(self.wire_rings) - 2)
            self.wire_segments = max(self._min_wire_segments, int(self.wire_segments) - 2)
            self._quality_cooldown = 0.35
        elif self._fps_ema > hi:
            # upshift (more sharpness / definition)
            self.mesh_layers = min(self._max_layers, int(self.mesh_layers) + 6)
            self.mesh_segments = min(self._max_segments, int(self.mesh_segments) + 10)
            self.wire_rings = min(self._max_wire_rings, int(self.wire_rings) + 2)
            self.wire_segments = min(self._max_wire_segments, int(self.wire_segments) + 2)
            self._quality_cooldown = 0.45

    def paintGL(self):
        self._auto_quality_tick()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        if self.camera_orbit:
            self.camera_angle += 0.25
            cx = np.sin(np.radians(self.camera_angle)) * self.camera_dist
            cz = np.cos(np.radians(self.camera_angle)) * self.camera_dist
            gluLookAt(cx, 0, cz, 0, 0, 0, 0, 1, 0)
        else:
            glTranslatef(0, 0, -float(self.camera_dist))

        # Model position (viewable/controllable)
        glTranslatef(float(self.pos_x), float(self.pos_y), float(self.pos_z))

        audio = beat_energy * self.beat_gain * 0.6 + vocal_energy * self.vocal_gain * 0.6
        audio = float(np.clip(audio, 0.0, 2.0))

        scale = 1.0 + audio * 0.22
        glScalef(scale, scale, scale)

        self.time_offset += self.flow_speed
        t = self.time_offset

        ax = np.sin(t * 0.37) * self.axis_wander + np.sin(t * 1.33) * self.jitter
        ay = np.cos(t * 0.29) * self.axis_wander + np.cos(t * 1.11) * self.jitter
        az = np.sin(t * 0.41) * self.axis_wander + np.cos(t * 0.97) * self.jitter

        spin = self.spin_base + self.spin_audio * audio

        self.rot_x += spin * (0.78 + 0.50 * ax)
        self.rot_y += spin * (1.00 + 0.45 * ay)
        self.rot_z += spin * (0.70 + 0.55 * az)

        if self.axis_x:
            glRotatef(self.rot_x, 1, 0, 0)
        if self.axis_y:
            glRotatef(self.rot_y, 0, 1, 0)
        if self.axis_z:
            glRotatef(self.rot_z, 0, 0, 1)

        if self.flow_mode == "wormhole":
            self.draw_wormhole_mesh(audio)
        else:
            self.draw_flow_mesh(audio)

        self.draw_wire_sphere()

    def draw_flow_mesh(self, audio):
        layers = int(self.mesh_layers)
        segments = int(self.mesh_segments)

        base_radius = 1.2
        t = self.time_offset
        glLineWidth(1.15)

        for i in range(layers):
            lat = np.pi * (-0.5 + i / layers)
            glBegin(GL_LINE_STRIP)
            for j in range(segments + 1):
                lon = 2 * np.pi * j / segments
                x = np.cos(lat) * np.cos(lon)
                y = np.sin(lat)
                z = np.cos(lat) * np.sin(lon)

                flow = (np.sin(lon * 3 + t) + np.cos(lat * 4 - t * 1.3)) * 0.15
                r = base_radius + flow + audio

                glColor4f(*self.mesh_color, 0.28)
                glVertex3f(r * x, r * y, r * z)
            glEnd()

    def draw_wormhole_mesh(self, audio):
        layers = int(self.mesh_layers)
        segments = int(self.mesh_segments)

        base_radius = 1.18
        t = self.time_offset

        tunnel_r = max(0.03, float(self.wh_tunnel_radius))
        mouth_start_frac = clamp(float(self.wh_mouth_start), 0.10, 0.95)
        softness = clamp(float(self.wh_softness), 0.05, 1.0)
        twist_amt = max(0.0, float(self.wh_twist))
        strength = clamp(float(self.wh_strength), 0.0, 1.5)

        glLineWidth(1.25)

        for i in range(layers):
            lat = np.pi * (-0.5 + i / layers)
            s = np.sin(lat)
            c = np.cos(lat)

            glBegin(GL_LINE_STRIP)
            for j in range(segments + 1):
                lon = 2 * np.pi * j / segments

                flow = (np.sin(lon * 3 + t) + np.cos(lat * 4 - t * 1.3)) * 0.12
                R = base_radius + flow + (audio * 0.78)

                y_s = R * s
                y_abs = abs(y_s)
                sign_y = 1.0 if y_s >= 0 else -1.0

                y_start = mouth_start_frac * R
                blend0 = max(0.0, y_start - softness * 0.25 * R)
                blend1 = R

                w = smoothstep(blend0, blend1, y_abs) * strength
                w = clamp(w, 0.0, 1.0)

                y_limit = float(np.sqrt(max(R * R - tunnel_r * tunnel_r, 0.0)))

                flow_dampen = (1.0 - 0.70 * w)
                R2 = base_radius + (flow * flow_dampen) + (audio * 0.78)

                xz_sphere = R2 * c
                xz_target = tunnel_r

                y_target = min(y_abs, y_limit)
                y_abs2 = (1.0 - w) * y_abs + w * y_target
                xz2 = (1.0 - w) * xz_sphere + w * xz_target

                twist = w * twist_amt * (0.9 + 0.8 * audio)
                ang = lon + twist * np.sin(t * 0.85 + lat * 2.2)

                x2 = xz2 * np.cos(ang)
                z2 = xz2 * np.sin(ang)
                y2 = sign_y * y_abs2

                alpha = 0.20 + 0.22 * w
                glColor4f(*self.mesh_color, alpha)
                glVertex3f(x2, y2, z2)

            glEnd()

    def draw_wire_sphere(self):
        rings = int(self.wire_rings)
        segments = int(self.wire_segments)
        r = float(self.wire_radius)

        glLineWidth(1.0)
        glColor4f(*self.wire_color, float(self.wire_alpha))

        for i in range(1, rings):
            lat = np.pi * (-0.5 + i / rings)
            glBegin(GL_LINE_LOOP)
            for j in range(segments):
                lon = 2 * np.pi * j / segments
                x = r * np.cos(lat) * np.cos(lon)
                y = r * np.sin(lat)
                z = r * np.cos(lat) * np.sin(lon)
                glVertex3f(x, y, z)
            glEnd()

        glPointSize(3.5 + beat_energy * 6.0)
        glBegin(GL_POINTS)
        for i in range(1, rings):
            lat = np.pi * (-0.5 + i / rings)
            for j in range(segments):
                lon = 2 * np.pi * j / segments
                x = r * np.cos(lat) * np.cos(lon)
                y = r * np.sin(lat)
                z = r * np.cos(lat) * np.sin(lon)
                glVertex3f(x, y, z)
        glEnd()


# ===================== UI Helpers =====================
class ControlPanel(QWidget):
    """
    Top menu bar controls (wide + short).
    Put inside a horizontal QScrollArea.
    Includes:
      - Save/Load tuning state to JSON
      - View XYZ/rotation readouts
      - Auto quality (adaptive sharpness vs FPS)
    """
    def __init__(self, gl: SphereGL):
        super().__init__()
        self.gl = gl

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        title_row = QWidget()
        title_lay = QHBoxLayout(title_row)
        title_lay.setContentsMargins(0, 0, 0, 0)
        title_lay.setSpacing(10)

        title = QLabel("Controls")
        title.setStyleSheet("font-size: 18px; font-weight: 650;")
        title_lay.addWidget(title, 1)

        self.btn_save = QPushButton("Save Preset")
        self.btn_load = QPushButton("Load Preset")
        self.btn_save.clicked.connect(self.save_preset)
        self.btn_load.clicked.connect(self.load_preset)
        title_lay.addWidget(self.btn_save)
        title_lay.addWidget(self.btn_load)

        outer.addWidget(title_row)

        row = QWidget()
        row_lay = QHBoxLayout(row)
        row_lay.setContentsMargins(0, 0, 0, 0)
        row_lay.setSpacing(10)

        row_lay.addWidget(self._section_mode())
        row_lay.addWidget(self._section_audio())
        row_lay.addWidget(self._section_motion())
        row_lay.addWidget(self._section_transform())
        row_lay.addWidget(self._section_wormhole())
        row_lay.addWidget(self._section_rendering())
        row_lay.addWidget(self._section_colors())
        row_lay.addStretch(1)

        outer.addWidget(row)
        outer.addStretch(1)

        # live readout updater
        self._readout_timer = QTimer(self)
        self._readout_timer.timeout.connect(self._update_readouts)
        self._readout_timer.start(100)

    def _group(self, name: str):
        g = QGroupBox(name)
        g.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        g.setMinimumWidth(260)
        g.setMaximumWidth(360)
        lay = QFormLayout(g)
        lay.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        lay.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        lay.setVerticalSpacing(4)
        lay.setContentsMargins(8, 8, 8, 8)
        return g, lay

    def _slider_row(self, layout: QFormLayout, label: str, minv: int, maxv: int, start: int, on_change, fmt):
        row = QWidget()
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(8)

        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(minv, maxv)
        s.setValue(start)

        val = QLabel(fmt(start))
        val.setFixedWidth(88)
        val.setStyleSheet("font-family: Consolas, monospace;")
        val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        def _changed(v):
            val.setText(fmt(v))
            on_change(v)

        s.valueChanged.connect(_changed)
        hl.addWidget(s, 1)
        hl.addWidget(val, 0)
        layout.addRow(label, row)
        return s

    # ---------- Presets ----------
    def save_preset(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Preset", "preset.json", "JSON Files (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        data = self.gl.get_state()
        data["_meta"] = {"app": "ReactiveGeometricSphere", "version": 1, "saved_at": time.time()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Preset", "", "JSON Files (*.json)")
        if not path:
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.gl.apply_state(data)

    # ---------- Readouts ----------
    def _update_readouts(self):
        if hasattr(self, "lbl_pos"):
            self.lbl_pos.setText(f"({self.gl.pos_x:+.2f}, {self.gl.pos_y:+.2f}, {self.gl.pos_z:+.2f})")
        if hasattr(self, "lbl_rot"):
            self.lbl_rot.setText(f"({self.gl.rot_x%360:06.2f}, {self.gl.rot_y%360:06.2f}, {self.gl.rot_z%360:06.2f})")
        if hasattr(self, "lbl_fps"):
            self.lbl_fps.setText(f"{self.gl._fps_ema:5.1f} fps")

    # ---------- Sections ----------
    def _section_mode(self):
        g, lay = self._group("Mode")
        mode = QComboBox()
        mode.addItems(["wormhole", "sphere"])
        mode.setCurrentText(self.gl.flow_mode)
        mode.currentTextChanged.connect(lambda t: setattr(self.gl, "flow_mode", t))
        lay.addRow("Flow Mode", mode)

        orbit = QCheckBox("Camera Orbit")
        orbit.setChecked(self.gl.camera_orbit)
        orbit.stateChanged.connect(lambda st: setattr(self.gl, "camera_orbit", st == 2))
        lay.addRow("", orbit)
        return g

    def _section_audio(self):
        g, lay = self._group("Audio")
        self._slider_row(lay, "Bass Sens.", 0, 20, int(self.gl.beat_gain),
                         lambda v: setattr(self.gl, "beat_gain", float(v)),
                         fmt=lambda v: f"{v:.0f}")
        self._slider_row(lay, "Vocal Sens.", 0, 20, int(self.gl.vocal_gain),
                         lambda v: setattr(self.gl, "vocal_gain", float(v)),
                         fmt=lambda v: f"{v:.0f}")

        note = QLabel("Energy: 0.7 RMS + 0.3 Peak (smoothed)")
        note.setWordWrap(True)
        note.setStyleSheet("color:#9aa0a6; font-size: 10px;")
        lay.addRow("", note)
        return g

    def _section_motion(self):
        g, lay = self._group("Motion")

        self._slider_row(lay, "Flow Speed", 0, 100, int(self.gl.flow_speed * 1000),
                         lambda v: setattr(self.gl, "flow_speed", v / 1000.0),
                         fmt=lambda v: f"{v/1000.0:.3f}")

        self._slider_row(lay, "Zoom", 220, 700, int(self.gl.camera_dist * 100),
                         lambda v: setattr(self.gl, "camera_dist", v / 100.0),
                         fmt=lambda v: f"{v/100.0:.2f}")

        self._slider_row(lay, "Spin Base", 0, 250, int(self.gl.spin_base * 100),
                         lambda v: setattr(self.gl, "spin_base", v / 100.0),
                         fmt=lambda v: f"{v/100.0:.2f}")

        self._slider_row(lay, "Spin Audio", 0, 500, int(self.gl.spin_audio * 100),
                         lambda v: setattr(self.gl, "spin_audio", v / 100.0),
                         fmt=lambda v: f"{v/100.0:.2f}")

        self._slider_row(lay, "Axis Wander", 0, 150, int(self.gl.axis_wander * 100),
                         lambda v: setattr(self.gl, "axis_wander", v / 100.0),
                         fmt=lambda v: f"{v/100.0:.2f}")

        self._slider_row(lay, "Jitter", 0, 150, int(self.gl.jitter * 100),
                         lambda v: setattr(self.gl, "jitter", v / 100.0),
                         fmt=lambda v: f"{v/100.0:.2f}")

        ax = QCheckBox("X")
        ay = QCheckBox("Y")
        az = QCheckBox("Z")
        ax.setChecked(self.gl.axis_x)
        ay.setChecked(self.gl.axis_y)
        az.setChecked(self.gl.axis_z)
        ax.stateChanged.connect(lambda st: setattr(self.gl, "axis_x", st == 2))
        ay.stateChanged.connect(lambda st: setattr(self.gl, "axis_y", st == 2))
        az.stateChanged.connect(lambda st: setattr(self.gl, "axis_z", st == 2))

        row = QWidget()
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(10)
        hl.addWidget(QLabel("Axes:"))
        hl.addWidget(ax)
        hl.addWidget(ay)
        hl.addWidget(az)
        hl.addStretch(1)
        lay.addRow("", row)

        return g

    def _section_transform(self):
        g, lay = self._group("Transform (XYZ)")

        self._slider_row(lay, "Pos X", -200, 200, int(self.gl.pos_x * 100),
                         lambda v: setattr(self.gl, "pos_x", v / 100.0),
                         fmt=lambda v: f"{v/100.0:+.2f}")
        self._slider_row(lay, "Pos Y", -200, 200, int(self.gl.pos_y * 100),
                         lambda v: setattr(self.gl, "pos_y", v / 100.0),
                         fmt=lambda v: f"{v/100.0:+.2f}")
        self._slider_row(lay, "Pos Z", -200, 200, int(self.gl.pos_z * 100),
                         lambda v: setattr(self.gl, "pos_z", v / 100.0),
                         fmt=lambda v: f"{v/100.0:+.2f}")

        self.lbl_pos = QLabel("(+0.00, +0.00, +0.00)")
        self.lbl_pos.setStyleSheet("font-family: Consolas, monospace;")
        lay.addRow("Position", self.lbl_pos)

        self.lbl_rot = QLabel("(000.00, 000.00, 000.00)")
        self.lbl_rot.setStyleSheet("font-family: Consolas, monospace;")
        lay.addRow("Rotation", self.lbl_rot)

        return g

    def _section_wormhole(self):
        g, lay = self._group("Wormhole (Contained)")

        self._slider_row(lay, "Tunnel Radius", 3, 60, int(self.gl.wh_tunnel_radius * 100),
                         lambda v: setattr(self.gl, "wh_tunnel_radius", v / 100.0),
                         fmt=lambda v: f"{v/100.0:.2f}")

        self._slider_row(lay, "Mouth Start", 10, 95, int(self.gl.wh_mouth_start * 100),
                         lambda v: setattr(self.gl, "wh_mouth_start", v / 100.0),
                         fmt=lambda v: f"{v/100.0:.2f}")

        self._slider_row(lay, "Softness", 5, 100, int(self.gl.wh_softness * 100),
                         lambda v: setattr(self.gl, "wh_softness", v / 100.0),
                         fmt=lambda v: f"{v/100.0:.2f}")

        self._slider_row(lay, "Twist", 0, 300, int(self.gl.wh_twist * 100),
                         lambda v: setattr(self.gl, "wh_twist", v / 100.0),
                         fmt=lambda v: f"{v/100.0:.2f}")

        self._slider_row(lay, "Strength", 0, 150, int(self.gl.wh_strength * 100),
                         lambda v: setattr(self.gl, "wh_strength", v / 100.0),
                         fmt=lambda v: f"{v/100.0:.2f}")

        tip = QLabel("Contained tunnel: increase Strength, keep Tunnel Radius modest (0.10–0.22).")
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#9aa0a6; font-size: 10px;")
        lay.addRow("", tip)
        return g

    def _section_rendering(self):
        g, lay = self._group("Rendering (Adaptive)")

        aq = QCheckBox("Auto Quality")
        aq.setChecked(self.gl.auto_quality)
        aq.stateChanged.connect(lambda st: setattr(self.gl, "auto_quality", st == 2))
        lay.addRow("", aq)

        self._slider_row(lay, "Target FPS", 15, 240, int(self.gl.target_fps),
                         lambda v: setattr(self.gl, "target_fps", int(v)),
                         fmt=lambda v: f"{v:d}")

        self.lbl_fps = QLabel(" 60.0 fps")
        self.lbl_fps.setStyleSheet("font-family: Consolas, monospace;")
        lay.addRow("Measured", self.lbl_fps)

        # Manual resolution controls still available (Auto Quality can override gently)
        self._slider_row(lay, "Mesh Layers", self.gl._min_layers, self.gl._max_layers, int(self.gl.mesh_layers),
                         lambda v: setattr(self.gl, "mesh_layers", int(v)),
                         fmt=lambda v: f"{v:d}")

        self._slider_row(lay, "Mesh Segments", self.gl._min_segments, self.gl._max_segments, int(self.gl.mesh_segments),
                         lambda v: setattr(self.gl, "mesh_segments", int(v)),
                         fmt=lambda v: f"{v:d}")

        self._slider_row(lay, "Wire Rings", self.gl._min_wire_rings, self.gl._max_wire_rings, int(self.gl.wire_rings),
                         lambda v: setattr(self.gl, "wire_rings", int(v)),
                         fmt=lambda v: f"{v:d}")

        self._slider_row(lay, "Wire Segments", self.gl._min_wire_segments, self.gl._max_wire_segments, int(self.gl.wire_segments),
                         lambda v: setattr(self.gl, "wire_segments", int(v)),
                         fmt=lambda v: f"{v:d}")

        self._slider_row(lay, "Wire Radius", 80, 220, int(self.gl.wire_radius * 100),
                         lambda v: setattr(self.gl, "wire_radius", v / 100.0),
                         fmt=lambda v: f"{v/100.0:.2f}")

        self._slider_row(lay, "Wire Alpha", 0, 100, int(self.gl.wire_alpha * 100),
                         lambda v: setattr(self.gl, "wire_alpha", v / 100.0),
                         fmt=lambda v: f"{v/100.0:.2f}")

        return g

    def _section_colors(self):
        g, lay = self._group("Colors")

        mesh_btn = QPushButton("Pick Mesh Color")
        wire_btn = QPushButton("Pick Wire Color")

        def pick_mesh():
            c = QColorDialog.getColor()
            if c.isValid():
                self.gl.mesh_color = [c.redF(), c.greenF(), c.blueF()]

        def pick_wire():
            c = QColorDialog.getColor()
            if c.isValid():
                self.gl.wire_color = [c.redF(), c.greenF(), c.blueF()]

        mesh_btn.clicked.connect(pick_mesh)
        wire_btn.clicked.connect(pick_wire)

        lay.addRow("", mesh_btn)
        lay.addRow("", wire_btn)
        return g


# ===================== Main Window =====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reactive Geometric Sphere (Top Controls + Presets + Adaptive Quality)")
        self.resize(1500, 900)

        self.gl = SphereGL()
        panel = ControlPanel(self.gl)

        top_scroll = QScrollArea()
        top_scroll.setWidget(panel)
        top_scroll.setWidgetResizable(True)
        top_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        top_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        top_scroll.setFixedHeight(230)
        top_scroll.setFrameShape(QFrame.Shape.NoFrame)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        top_frame = QFrame()
        top_frame.setFrameShape(QFrame.Shape.StyledPanel)
        top_layout = QVBoxLayout(top_frame)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(top_scroll)

        layout.addWidget(top_frame, 0)
        layout.addWidget(self.gl, 1)

        self.setCentralWidget(root)


# ===================== Run =====================
if __name__ == "__main__":
    try:
        stream = sd.InputStream(
            callback=audio_callback,
            channels=2,
            samplerate=RATE,
            blocksize=BUFFER
        )
        stream.start()
    except Exception as e:
        print("Audio stream failed to start:", e)
        stream = None

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    ret = app.exec()

    if stream is not None:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass

    sys.exit(ret)

