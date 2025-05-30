import vispy
import numpy as np
from vispy import scene, color
from vispy.scene import Node
from vispy.scene.visuals import Markers, ColorBar, Text
from vispy.app import use_app
from sklearn.cluster import DBSCAN
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QSlider, QApplication
from PyQt5.QtCore import Qt
from .data_handler import DynamicLoader, get_all_speeds
import matplotlib.pyplot as plt

vispy.use('PyQt5')


class ThreeDVisualizer(QMainWindow):
    def __init__(self, filename, sim_callback: DynamicLoader, params: dict, initial_num=100):
        super().__init__()
        self.filename = filename
        self.sim_step = sim_callback
        self.running = True
        self.reverse = False
        self.interval = 1 / 60.0

        self.setWindowTitle("3D Simulation with Slider")
        self.resize(1200, 950)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.canvas = scene.SceneCanvas(keys='interactive', size=(1200, 900), show=False)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(distance=3)
        self.layout.addWidget(self.canvas.native)
        self.canvas.events.resize.connect(self._on_resize)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.sim_step.max_step - 1)
        self.slider.valueChanged.connect(self.on_slider_change)
        self.layout.addWidget(self.slider)

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
        self.param_display.pos = (10, 260)

        self.initial_num = initial_num
        self.cloud_group = Node(parent=self.view.scene)

        # 初始化聚类跟踪状态
        self.persistent_clusters = {}
        self.label_counter = 0
        self.persistence_threshold = 5
        self.cluster_show = False

        self._init_visuals()

        from vispy import app
        self.timer = app.Timer(interval=self.interval, connect=self.update)

        self.persistent_clusters = {}
        self.label_counter = 0
        self.persistence_threshold = 5

    def _on_resize(self, event): 
        canvas_width = self.canvas.size[0]
        canvas_height = self.canvas.size[1]
        colorbar_width = 85  # 你设置的颜色条宽度
        margin = 40  # 距离右边的像素
        if hasattr(self, 'colorbar'):
            self.colorbar.pos = (canvas_width - colorbar_width / 2 - margin, canvas_height / 2)

        if hasattr(self, 'param_display'):
            self.title_text.pos = (10, 260)

    def _init_visuals(self):
        positions, speeds, masses = self.sim_step()
        self.sun_visual = Markers(
            pos=positions[0:1],
            edge_color=None,
            face_color=(1, 1, 0, 1),
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
        self._update_cluster_clouds(positions[1:], False)
        self.cloud_group = Node(parent=self.view.scene)

    def _update_asteroid_data(self, speeds, masses, pos):
        if len(speeds) == 0:
            return
        all_speeds = get_all_speeds(self.filename)
        speed_min = np.round(all_speeds.min(), 4)
        speed_max = np.round(all_speeds.max(), 4)
        if speed_max > 5: 
            speed_max = 5
        if not self.colorbar_initialized:
            self.colorbar.clim = (speed_min, speed_max)
            self.colorbar_initialized = True
        speeds[speeds > speed_max] = 5
        speed_range = speed_max - speed_min + 1e-8
        speed_colors = color.get_colormap('viridis').map((speeds - speed_min)/speed_range)
        # speed_colors = color.get_colormap('inferno').map(np.power((np.log(speeds) - np.log(speed_min))/(np.log(speed_max) - np.log(speed_min)), 3))
        self.asteroid_visual.set_data(
            pos=pos,
            face_color=speed_colors,
            size=100 * masses ** 0.3333,
            edge_color=None
        )

    def _update_cluster_clouds(self, pos, if_print=False, do_visualization = True):
        for child in list(self.cloud_group.children):
            child.parent = None

        eps = 0.15
        threshold = 3 * eps
        db = DBSCAN(eps=eps, min_samples=5)
        labels = db.fit_predict(pos)
        unique_labels = set(labels)

        if if_print:
            print(f"DBSCAN found {len(unique_labels)} asteroid cluster(s)")

        current_step_clusters = {}

        for label in unique_labels:
            if label == -1:
                continue

            mask = labels == label
            cluster_points = pos[mask]
            if cluster_points.shape[0] == 0:
                continue

            # 约束 2: 团块太大
            count = np.sum(labels == label)
            if count > self.initial_num / 10:
                if if_print:
                    print(f"[改为噪声] 聚类 {label} 太大，包含 {count} 个点")
                labels[labels == label] = -1
                continue

            # 约束 3: 移除远离质心的点
            centroid = np.mean(cluster_points, axis=0)
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            mean_dist, std_dist = distances.mean(), distances.std()
            far_mask = distances > (mean_dist + 2 * std_dist)
            if far_mask.sum() > 0:
                global_indices = np.where(mask)[0]
                outlier_indices = global_indices[far_mask]
                labels[outlier_indices] = -1
                continue

            # 约束 4: 扁平团块剔除
            broad_1 = np.max(cluster_points, axis=0)
            broad_2 = np.min(cluster_points, axis=0)
            broad_distance = np.round(np.linalg.norm(broad_1 - broad_2), 3)
            if broad_distance > threshold:
                if if_print:
                    print(f"聚类 {label} 太扁平，包含 {count} 个点，最大距离{broad_distance}")
                labels[labels == label] = -1
                continue

            # 匹配历史中最相似的团块
            best_match_label = None
            best_score = -np.inf
            for prev_label, info in self.persistent_clusters.items():
                score = self._cluster_similarity(info['last_points'], cluster_points)
                if score > best_score:
                    best_score = score
                    best_match_label = prev_label

            similarity_threshold = 0.5
            if best_score > similarity_threshold:
                matched_label = best_match_label
                self.persistent_clusters[matched_label]['lifespan'] += 1
                self.persistent_clusters[matched_label]['last_points'] = cluster_points
            else:
                self.label_counter += 1
                matched_label = self.label_counter
                self.persistent_clusters[matched_label] = {
                    'lifespan': 1,
                    'last_points': cluster_points,
                    'first_points': cluster_points.copy(),
                    'ready_to_show': False
                }

            current_step_clusters[matched_label] = cluster_points

            info = self.persistent_clusters[matched_label]
            if not info['ready_to_show'] and info['lifespan'] >= self.persistence_threshold:
                info['ready_to_show'] = True

            if info['ready_to_show'] and do_visualization:
                points_to_show = info['first_points'] if info['lifespan'] == self.persistence_threshold else info['last_points']
                cmap = plt.get_cmap('tab20')
                base_color = cmap(matched_label % 20)[:3]
                face_color = np.hstack([
                    np.tile(base_color, (len(points_to_show), 1)),
                    np.full((len(points_to_show), 1), 0.2)
                ])
                cloud = Markers(
                    pos=points_to_show,
                    face_color=face_color,
                    size=50,
                    edge_color=None,
                    parent=self.cloud_group
                )
                cloud.order = -1

        self._cleanup_stale_clusters(current_step_clusters)

    def _cluster_similarity(self, cluster_a, cluster_b):
        from sklearn.decomposition import PCA
        from scipy.spatial.distance import euclidean

        centroid_a = np.mean(cluster_a, axis=0)
        centroid_b = np.mean(cluster_b, axis=0)
        a_centered = cluster_a - centroid_a
        b_centered = cluster_b - centroid_b

        pca_a = PCA(n_components=3).fit(a_centered)
        pca_b = PCA(n_components=3).fit(b_centered)
        cos_sim = np.abs(np.dot(pca_a.components_[0], pca_b.components_[0]))

        dist_a = np.linalg.norm(a_centered, axis=1)
        dist_b = np.linalg.norm(b_centered, axis=1)
        hist_a, _ = np.histogram(dist_a, bins=10, range=(0, 0.2), density=True)
        hist_b, _ = np.histogram(dist_b, bins=10, range=(0, 0.2), density=True)
        shape_diff = euclidean(hist_a, hist_b)

        return cos_sim - shape_diff

    def _cleanup_stale_clusters(self, current_clusters):
        for label in list(self.persistent_clusters.keys()):
            if label not in current_clusters:
                self.persistent_clusters[label]['lifespan'] -= 1
                if self.persistent_clusters[label]['lifespan'] <= 0:
                    del self.persistent_clusters[label]


    def update(self, event):
        if self.running and self.sim_step.no_more():
            self.running = False
            print('FINISHED')
            return
        if self.running:
            new_pos, new_speed, new_mass = self.sim_step(self.reverse)
            self.slider.blockSignals(True)
            self.slider.setValue(self.sim_step.current_step)
            print("current step", self.sim_step.current_step)
            self.slider.blockSignals(False)

            self.sun_visual.set_data(
                pos=new_pos[0:1],
                face_color=(1, 1, 0, 1),
                size=30,
                edge_color=None
            )
            self._update_asteroid_data(new_speed[1:], new_mass[1:], new_pos[1:])
            self._update_cluster_clouds(new_pos[1:], True, self.cluster_show)
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
        elif event.key == 'C':
            self.cluster_show = not self.cluster_show
            new_pos, new_speed, new_mass = self.sim_step(self.reverse)
            new_pos, new_speed, new_mass = self.sim_step(not self.reverse)
            self._update_cluster_clouds(new_pos[1:], True, self.cluster_show)

    def on_slider_change(self, value):
        self.running = False
        self.sim_step.goto(value)
        new_pos, new_speed, new_mass = self.sim_step()
        self.sun_visual.set_data(
            pos=new_pos[0:1],
            face_color=(1, 1, 0, 1),
            size=30,
            edge_color=None
        )
        self._update_asteroid_data(new_speed[1:], new_mass[1:], new_pos[1:])
        if self.cluster_show: 
            self._update_cluster_clouds(new_pos[1:], False)
        self.canvas.update()

    def run(self):
        self.timer.start()
        self.show()
        use_app().run()
