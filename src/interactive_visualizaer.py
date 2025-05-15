import vispy
import numpy as np
from vispy import app, scene, color
from vispy.scene.visuals import Markers
from .data_handler import DynamicLoader

vispy.use('PyQt5')


class ThreeDVisualizer:
    def __init__(self, sim_callback: DynamicLoader):
        self.canvas = scene.SceneCanvas(keys='interactive', size=(1200, 900), show=True)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(distance=5) # 需要根据min_mass,max_mass调整？

        # 初始化参数
        self.sim_step = sim_callback  # sim_callback只返回(pos, speed, mass)
        self.running = True
        self._init_visuals()

        # 定时器设置
        self.timer = app.Timer(interval='auto', connect=self.update)
        self.canvas.events.key_press.connect(self.on_key_press)

    def _init_visuals(self):
        positions, speeds, masses = self.sim_step()

        # 自动判断太阳为第一个元素
        self.sun_visual = Markers(
            pos=positions[0:1],  # 取第一个元素保持二维数组结构
            edge_color=None,
            face_color=(1, 1, 0, 1),  # 固定黄色
            symbol='disc',
            size=30,
            parent=self.view.scene
        )

        # 其他天体从第二个元素开始
        self.asteroid_visual = Markers(
            pos=positions[1:],
            symbol='disc',
            parent=self.view.scene
        )
        self._update_asteroid_data(speeds[1:], masses[1:], positions[1:])

    def _update_asteroid_data(self, speeds, masses, pos):
        # 处理空数据情况
        if len(speeds) == 0:
            return

        # 动态调整颜色范围
        speed_min, speed_max = speeds.min(), speeds.max()
        speed_range = speed_max - speed_min + 1e-8  # 防止除零

        # 使用inferno色系增强对比度
        speed_colors = color.get_colormap('inferno').map(
            (speeds - speed_min) / speed_range
        )

        # 更新天体属性
        self.asteroid_visual.set_data(
            pos=pos,
            face_color=speed_colors,
            size=2000 * np.sqrt(masses), # 质量映射，需要根据min_mass, max_mass调整？
            edge_color=None
        )

    def update(self, event):
        if self.running and self.sim_step.no_more():
            self.running = False
            print('FINISHED')
            return
        if self.running:
            new_pos, new_speed, new_mass = self.sim_step()

            # 更新太阳属性（需保持所有视觉参数）
            self.sun_visual.set_data(
                pos=new_pos[0:1],
                face_color=(1, 1, 0, 1),  # 保持黄色
                size=30,  # 固定大小
                edge_color=None  # 保持无边框
            )

            self._update_asteroid_data(new_speed[1:], new_mass[1:], new_pos[1:])

            self.canvas.update()

    def on_key_press(self, event):
        if event.key == ' ':
            self.running = not self.running
            print(f"Simulation {'paused' if not self.running else 'resumed'}")

    def run(self):
        self.timer.start()
        app.run()
