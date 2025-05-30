import vispy
import numpy as np
from vispy import scene, color
from vispy.scene.visuals import Markers, ColorBar, Text
from vispy.scene import widgets
from vispy.app import use_app
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QSlider, QApplication
from PyQt5.QtCore import Qt
from .data_handler import DynamicLoader, get_all_speeds

vispy.use('PyQt5')

MIN_SPEED, SPEED_RANGE, BASE_RADIUS = None, None, None

class ThreeDVisualizer(QMainWindow):
    def __init__(self, sim_callback: DynamicLoader, params: dict, index: int):
        super().__init__()
        self.sim_step = sim_callback
        self.running = True
        self.reverse = False
        self.interval = 1 / 60.0  # 60Hz

        # Set up main window and layout
        self.setWindowTitle("3D Simulation with Slider")
        self.resize(1200, 950)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Create canvas and add to layout
        self.canvas = scene.SceneCanvas(keys='interactive', size=(1200, 900), show=False)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(distance=3)
        self.layout.addWidget(self.canvas.native)
        self.canvas.events.resize.connect(self._on_resize)

        # Create slider and add to layout
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.sim_step.max_step - 1)
        self.slider.valueChanged.connect(self.on_slider_change)
        self.layout.addWidget(self.slider)

        # Canvas events
        self.canvas.events.key_press.connect(self.on_key_press)

        # Colorbar
        # 创建颜色条
        colormap = color.get_colormap('viridis')
        self.colorbar = ColorBar(
            cmap=colormap,
            orientation='right',
            size=(500, 20),
            label='Speed',
            label_color = 'white',
            clim=(0,1),
            border_color='black',
            parent=self.canvas.scene  # 直接添加到 scene
        )
        self.colorbar.label.font_size = 12
        canvas_width = self.canvas.size[0]
        canvas_height = self.canvas.size[1]
        colorbar_width = 20  # 或者你定义的实际宽度
        margin = 85
        self.colorbar.pos = (canvas_width - colorbar_width - margin, canvas_height / 2)  # 你可以调整这个位置来改变颜色条的显示位置
        self.colorbar_initialized = False
        # from vispy.visuals.axis import AxisVisual
        # for child in self.colorbar.children:
        #     if isinstance(child, AxisVisual):
        #         child.visible = False

        # Text
        self.params = params
        param_text = "\n".join(f"{k} = {v}" for k, v in params.items())

        self.param_display = scene.Text(
            param_text,
            color='white',
            font_size=8,
            anchor_x='left',
            anchor_y='top',
            parent=self.canvas.scene
        )
        self.param_display.pos = (10, 270)

        # Index & Speed Data
        self.index = index
        filename = f"1-{index}-data.dat"
        self.all_speeds = get_all_speeds("visualization_data/" + filename)
        self.speed_max = np.round(self.all_speeds.max(), 4)
        self.all_speeds[self.all_speeds < 1e-5] = 2 * self.speed_max
        self.speed_min = np.round(self.all_speeds.min(), 4)
        print(self.speed_min)
        self.all_speeds[self.all_speeds > self.speed_max] = 0
        if self.speed_max > 5: 
            self.speed_max = 5

        # Setup visuals
        self._init_visuals()

        # Timer
        from vispy import app
        self.timer = app.Timer(interval=self.interval, connect=self.update)

        self.base_radius = None
        self.min_speed = None
        self.max_speed = None

    def _on_resize(self, event):
        canvas_width = self.canvas.size[0]
        canvas_height = self.canvas.size[1]
        colorbar_width = 85  # 你设置的颜色条宽度
        margin = 40  # 距离右边的像素
        if hasattr(self, 'colorbar'):
            self.colorbar.pos = (canvas_width - colorbar_width / 2 - margin, canvas_height / 2)

        if hasattr(self, 'param_display'):
            self.title_text.pos = (10, 200)

    def _init_visuals(self):
        positions, speeds, masses = self.sim_step()
        self.sun_visual = Markers(
            pos=positions[0:1],
            edge_color=None,
            face_color=(1, 0.5, 0, 1),
            symbol='disc',
            size=10,
            parent=self.view.scene
        )
        self.asteroid_visual = Markers(
            pos=positions[1:],
            symbol='disc',
            parent=self.view.scene
        )
        self._update_asteroid_data(speeds[1:], masses[1:], positions[1:])

    def _update_asteroid_data(self, speeds, masses, pos):
        global BASE_RADIUS, MIN_SPEED, SPEED_RANGE
        if len(speeds) == 0:
            return
        if not self.colorbar_initialized:
            self.colorbar.clim = (f"{self.speed_min:.4f}", self.speed_max)
            self.colorbar_initialized = True
        # speeds[speeds > self.speed_max] = 5
        # speed_range = self.speed_max - self.speed_min + 1e-8
        # speed_colors = color.get_colormap('viridis').map((speeds - self.speed_min)/speed_range)
        if MIN_SPEED is None:
            MIN_SPEED, SPEED_RANGE = speeds.min() * 0.5, speeds.max() * 2.0 - speeds.min() * 0.5 + 1e-8
        if BASE_RADIUS is None:
            BASE_RADIUS = np.cbrt(np.min(masses))
        speed_colors = color.get_colormap('viridis').map(np.clip((speeds - MIN_SPEED)/ SPEED_RANGE, 0, 1))
        self.asteroid_visual.set_data(
            pos=pos,
            face_color=speed_colors,
            size=5 * np.cbrt(masses) / BASE_RADIUS,
            edge_color=None
        )

    def update(self, event):
        if self.running and self.sim_step.no_more():
            self.running = False
            print('FINISHED')
            return
        if self.running:
            new_pos, new_speed, new_mass = self.sim_step(self.reverse)
            self.slider.blockSignals(True)
            self.slider.setValue(self.sim_step.current_step)
            self.slider.blockSignals(False)

            self.sun_visual.set_data(
                pos=new_pos[0:1],
                face_color=(1, 0.5, 0, 1),
                size=30,
                edge_color=None
            )
            self._update_asteroid_data(new_speed[1:], new_mass[1:], new_pos[1:])
            self.canvas.update()


    def on_key_press(self, event):
        if event.key == ' ':
            self.running = not self.running
            print(f"Simulation {'paused' if not self.running else 'resumed'}")
        elif event.key == 'A':
            if self.interval > 1.0 / 1000.0:
                self.interval /= 2
                self.timer.interval = self.interval
                print("A")
            else:
                print("Max A")
        elif event.key == 'S':
            if self.interval < 1.0 / 5.0:
                self.interval *= 2
                self.timer.interval = self.interval
                print("S")
            else:
                print("Min S")
        elif event.key == 'R':
            self.reverse = not self.reverse
            print("R")
        elif event.key == 'D':
            self.view.camera.distance = 3  #默认值
        elif event.key == 'Right':
            new_val = min(self.slider.value() + 1, self.slider.maximum())
            self.slider.setValue(new_val)
            print("Right")
        elif event.key == 'Left':
            new_val = max(self.slider.value() - 1, self.slider.minimum())
            self.slider.setValue(new_val)
            print("Left")

    def on_slider_change(self, value):
        self.running = False
        self.sim_step.goto(value)
        new_pos, new_speed, new_mass = self.sim_step()
        self.sun_visual.set_data(
            pos=new_pos[0:1],
            face_color=(1, 0.5, 0, 1),
            size=30,
            edge_color=None
        )
        self._update_asteroid_data(new_speed[1:], new_mass[1:], new_pos[1:])
        self.canvas.update()

    def run(self):
        self.timer.start()
        self.show()
        use_app().run()
