import sys
import numpy as np

# ===================== Qt =====================
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QMenu, QWidgetAction, QSlider, QLabel
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
    global beat_energy, vocal_energy
    mono = np.mean(indata, axis=1)

    bass = bandpass(.5, 150, RATE, mono)
    vocals = bandpass(.5, 3000, RATE, mono)

    beat_energy = 0.9 * beat_energy + 0.1 * np.max(np.abs(bass))
    vocal_energy = 0.9 * vocal_energy + 0.1 * np.max(np.abs(vocals ** 2))

# ===================== OpenGL Widget =====================
class SphereGL(QOpenGLWidget):
    def __init__(self):
        super().__init__()

        self.rot = 10.0
        self.time_offset = 1.0

        # Sensitivity
        self.beat_gain = 6.0
        self.vocal_gain = 4.0

        # Motion
        self.rotation_speed = 0.6
        self.flow_speed = 0.015

        # Axis toggles
        self.axis_x = True
        self.axis_y = True
        self.axis_z = True

        # Wireframe
        self.wire_radius = 1.0
        self.wire_alpha = 0.35

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
        gluPerspective(45, w / max(h, 1), 0.1, 1000)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        glTranslatef(0, 0, -10)

        scale = 1.0 + beat_energy * self.beat_gain
        glScalef(scale, scale, scale)

        self.rot += self.rotation_speed
        self.time_offset += self.flow_speed

        rx = self.rot if self.axis_x else 0
        ry = self.rot if self.axis_y else 0
        rz = self.rot if self.axis_z else 0

        glRotatef(rx, 1, 0, 0)
        glRotatef(ry, 0, 1, 0)
        glRotatef(rz, 0, 0, 1)

        self.draw_flow_mesh()
        self.draw_wire_sphere()

    # -------- Organic Mesh --------
    def draw_flow_mesh(self):
        layers = 1000000000
        segments = 2000
        base_radius = 1.2
        t = self.time_offset

        glLineWidth(1.2)

        for i in range(layers):
            lat = np.pi * (-0.5 + i / layers)
            glBegin(GL_LINE_STRIP)

            for j in range(segments + 1):
                lon = 2 * np.pi * j / segments

                x = np.cos(lat) * np.cos(lon)
                y = np.sin(lat)
                z = np.cos(lat) * np.sin(lon)

                flow = (
                    np.sin(lon * 3 + t) +
                    np.cos(lat * 4 - t * 1.3)
                ) * 0.15

                audio_push = (
                    beat_energy * self.beat_gain * 0.6 +
                    vocal_energy * self.vocal_gain * 0.8
                )

                r = base_radius + flow + audio_push

                glColor4f(0.4, 0.7, 1.0, 0.35)
                glVertex3f(r * x, r * y, r * z)

            glEnd()

    # -------- Wireframe --------
    def draw_wire_sphere(self):
        rings = 100000000000000
        segments = 360
        r = self.wire_radius

        glLineWidth(0.5)
        glColor4f(1.0, 1.0, 1.0, self.wire_alpha)

        for i in range(1, rings):
            lat = np.pi * (-0.5 + i / rings)
            glBegin(GL_LINE_LOOP)
            for j in range(segments):
                lon = 2 * np.pi * j / segments
                glVertex3f(
                    r * np.cos(lat) * np.cos(lon),
                    r * np.sin(lat),
                    r * np.cos(lat) * np.sin(lon)
                )
            glEnd()

# ===================== Main Window + Menus =====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reactive Geometric Sphere")
        self.resize(11000, 900)

        self.gl = SphereGL()

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self.gl)
        self.setCentralWidget(central)

        self.create_menus()

    def slider_menu(self, menu, label, minv, maxv, start, callback):
        w = QWidget()
        l = QVBoxLayout(w)
        txt = QLabel(label)
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(minv, maxv)
        s.setValue(start)
        s.valueChanged.connect(callback)
        l.addWidget(txt)
        l.addWidget(s)

        act = QWidgetAction(self)
        act.setDefaultWidget(w)
        menu.addAction(act)

    def create_menus(self):
        bar = self.menuBar()

        audio = bar.addMenu("Audio")
        motion = bar.addMenu("Motion")
        axis = bar.addMenu("Axis")

        self.slider_menu(audio, "Bass Sensitivity", 0, 20, 6,
                         lambda v: setattr(self.gl, "beat_gain", v))
        self.slider_menu(audio, "Vocal Sensitivity", 0, 20, 4,
                         lambda v: setattr(self.gl, "vocal_gain", v))

        self.slider_menu(motion, "Rotation Speed", 0, 200, 60,
                         lambda v: setattr(self.gl, "rotation_speed", v / 100))
        self.slider_menu(motion, "Flow Speed", 0, 100, 15,
                         lambda v: setattr(self.gl, "flow_speed", v / 1000))

        axis.addAction("Toggle X Axis").triggered.connect(
            lambda: setattr(self.gl, "axis_x", not self.gl.axis_x))
        axis.addAction("Toggle Y Axis").triggered.connect(
            lambda: setattr(self.gl, "axis_y", not self.gl.axis_y))
        axis.addAction("Toggle Z Axis").triggered.connect(
            lambda: setattr(self.gl, "axis_z", not self.gl.axis_z))

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

