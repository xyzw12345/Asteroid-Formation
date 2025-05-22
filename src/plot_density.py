from sklearn.neighbors import KDTree
import numpy as np

def compute_neighbor_density_over_time(position_snapshots, radius=0.05, interval=1):
    """
    对每帧粒子位置，统计半径 r 内的邻居数，作为局部密度指标。
    
    参数:
        position_snapshots: List[np.ndarray(N_i, 3)]，每一帧的粒子位置
        radius: float，邻居统计半径
        interval: int，间隔多少帧采样一次（用于下采样）
    
    返回:
        density_data: List[np.ndarray]，每帧每个粒子的邻居数数组
    """
    density_data = []
    for i, positions in enumerate(position_snapshots[::interval]):
        if len(positions) == 0:
            density_data.append(np.array([]))
            continue
        tree = KDTree(positions)
        counts = tree.query_radius(positions, r=radius, count_only=True)
        density_data.append(counts - 1)  # 不计入自身
    return density_data

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def plot_density_surface(density_data, bins=30, filename="density_surface.png"):
    """
    绘制局部密度分布随时间的三维曲面图。
    
    参数:
        density_data: List[np.ndarray]，每帧的邻居数数组
        bins: int，直方图分箱数
        filename: str，保存图片路径
    """
    os.makedirs(os.path.dirname(f'frames/density.png'), exist_ok=True)

    hist_matrix = []
    for counts in density_data:
        if len(counts) == 0:
            hist_matrix.append(np.zeros(bins))
            continue
        hist, edges = np.histogram(counts, bins=bins, range=(0, max(1, counts.max())))
        hist_matrix.append(hist)

    hist_matrix = np.array(hist_matrix)
    T, D = np.meshgrid(
        np.arange(hist_matrix.shape[0]),
        0.5 * (edges[1:] + edges[:-1])
    )

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(D, T, hist_matrix.T, cmap="plasma")

    ax.set_xlabel("Local Neighbor Count")
    ax.set_ylabel("Time Step")
    ax.set_zlabel("Particle Count")
    ax.set_title("Local Density Evolution Over Time")

    plt.tight_layout()
    plt.savefig(f'frames/density.png', dpi=300)
    plt.close()