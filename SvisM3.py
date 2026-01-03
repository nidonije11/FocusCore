import sys
import numpy as np

# ===================== Qt =====================
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGroupBox, QFormLayout, QSlider, QLabel, QPushButton,
    QColorDialog, QComboBox, QCheckBox, QFrame, QSizePolicy,
    QScrollArea
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


def audio_callback(indata, frames, time, status):
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
        self.camera_dist = 4.0  # zoom in/out target

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

        # ===================== Wormhole (CONTAINED) params =====================
        # Internal tunnel: a cylindrical throat fully inside the sphere.
        self.wh_tunnel_radius = 0.18     # cylinder radius
        self.wh_mouth_start = 0.60      # where mouth deformation starts (fraction of sphere radius along Y)
        self.wh_softness = 0.45         # blend smoothness
        self.wh_twist = 1.10            # twist intensity
        self.wh_strength = 1.00         # how strongly to enforce the internal tunnel

        # Rendering / resolution (structural)
        self.mesh_layers = 120
        self.mesh_segments = 170
        self.wire_rings = 24
        self.wire_segments = 36

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(16)

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

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        if self.camera_orbit:
            self.camera_angle += 0.25
            cx = np.sin(np.radians(self.camera_angle)) * self.camera_dist
            cz = np.cos(np.radians(self.camera_angle)) * self.camera_dist
            gluLookAt(cx, 0, cz, 0, 0, 0, 0, 1, 0)
        else:
            glTranslatef(0, 0, -float(self.camera_dist))

        # Audio scalar driver
        audio = beat_energy * self.beat_gain * 0.6 + vocal_energy * self.vocal_gain * 0.6
        audio = float(np.clip(audio, 0.0, 2.0))

        # Keep global scale subtle
        scale = 1.0 + audio * 0.22
        glScalef(scale, scale, scale)

        self.time_offset += self.flow_speed
        t = self.time_offset

        # Rotation: constant + audio + smooth axis drift
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
        """
        Wormhole CONTAINED within the sphere:

        We morph the surface near high |y| into a cylinder of radius tunnel_r,
        BUT we clamp the cylinder extent to the sphere interior.

        A cylinder of radius tunnel_r fits inside a sphere of radius R only for:
          |y| <= y_limit = sqrt(R^2 - tunnel_r^2)

        So instead of extending past poles, we "cap" the throat at y_limit.
        This produces a clear internal tunnel without protrusion.
        """
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

                # Base "alive" flow, but damp inside mouth area so the tunnel reads cleanly
                flow = (np.sin(lon * 3 + t) + np.cos(lat * 4 - t * 1.3)) * 0.12

                # Sphere radius with audio
                R = base_radius + flow + (audio * 0.78)

                # Sphere point
                x_s = R * c * np.cos(lon)
                y_s = R * s
                z_s = R * c * np.sin(lon)

                y_abs = abs(y_s)
                sign_y = 1.0 if y_s >= 0 else -1.0

                # Mouth starts at y_start (fraction of R)
                y_start = mouth_start_frac * R
                # Blend window widened by softness
                blend0 = max(0.0, y_start - softness * 0.25 * R)
                blend1 = R

                w = smoothstep(blend0, blend1, y_abs) * strength
                w = clamp(w, 0.0, 1.0)

                # The internal cylinder can only exist up to y_limit inside the sphere
                y_limit = float(np.sqrt(max(R * R - tunnel_r * tunnel_r, 0.0)))

                # Dampen flow in the mouth region so tunnel doesn't look like a wobbling blob
                flow_dampen = (1.0 - 0.70 * w)
                R2 = base_radius + (flow * flow_dampen) + (audio * 0.78)

                # Sphere xz radius at this latitude
                xz_sphere = R2 * c

                # Target: internal cylinder
                xz_target = tunnel_r

                # IMPORTANT: y target is clamped to y_limit, so it stays inside sphere
                y_target = min(y_abs, y_limit)  # never exceed limit

                # Push points near poles DOWN to y_limit (not outward)
                # This creates a clear opening ring and an internal throat.
                y_abs2 = (1.0 - w) * y_abs + w * y_target

                # Blend xz toward the cylinder radius
                xz2 = (1.0 - w) * xz_sphere + w * xz_target

                # Twist primarily where w is high
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
    Left-side always-visible controls.
    Placed inside a QScrollArea so fullscreen never makes controls unusable.
    """
    def __init__(self, gl: SphereGL):
        super().__init__()
        self.gl = gl

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        title = QLabel("Controls")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        outer.addWidget(title)

        outer.addWidget(self._section_mode())
        outer.addWidget(self._section_audio())
        outer.addWidget(self._section_motion())
        outer.addWidget(self._section_wormhole())
        outer.addWidget(self._section_rendering())
        outer.addWidget(self._section_colors())
        outer.addStretch(1)

    def _group(self, name: str):
        g = QGroupBox(name)
        g.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        lay = QFormLayout(g)
        lay.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        lay.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        lay.setVerticalSpacing(6)
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
        val.setFixedWidth(60)
        val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        def _changed(v):
            val.setText(fmt(v))
            on_change(v)

        s.valueChanged.connect(_changed)
        hl.addWidget(s, 1)
        hl.addWidget(val, 0)
        layout.addRow(label, row)
        return s

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

        # Zoom slider (camera distance): smaller = zoom in
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

        # Axis toggles
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
        g, lay = self._group("Rendering")

        self._slider_row(lay, "Mesh Layers", 60, 220, int(self.gl.mesh_layers),
                         lambda v: setattr(self.gl, "mesh_layers", int(v)),
                         fmt=lambda v: f"{v:d}")

        self._slider_row(lay, "Mesh Segments", 80, 260, int(self.gl.mesh_segments),
                         lambda v: setattr(self.gl, "mesh_segments", int(v)),
                         fmt=lambda v: f"{v:d}")

        self._slider_row(lay, "Wire Rings", 8, 60, int(self.gl.wire_rings),
                         lambda v: setattr(self.gl, "wire_rings", int(v)),
                         fmt=lambda v: f"{v:d}")

        self._slider_row(lay, "Wire Segments", 12, 90, int(self.gl.wire_segments),
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
        self.setWindowTitle("Reactive Geometric Sphere (Usable Fullscreen)")
        self.resize(1200, 900)

        self.gl = SphereGL()

        # Left panel scaled down ~25% and placed in a scroll area (fixes fullscreen usability)
        panel = ControlPanel(self.gl)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Smaller panel width (about 25% reduction vs prior ~320)
        scroll.setFixedWidth(240)

        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        left_frame = QFrame()
        left_frame.setFrameShape(QFrame.Shape.StyledPanel)
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(scroll)

        layout.addWidget(left_frame, 0)
        layout.addWidget(self.gl, 1)

        self.setCentralWidget(root)


# ===================== Run =====================
if __name__ == "__main__":
    stream = sd.InputStream(
        callback=audio_callback,
        channels=2,
        samplerate=RATE,
        blocksize=BUFFER
    )
    stream.start()

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

