# import matplotlib.pyplot as plt
# import numpy as np

# def plot_mass_histograms(mass_snapshots, bins=50):
#     for i, masses in enumerate(mass_snapshots):
#         plt.figure()
#         plt.hist(masses[masses<1], bins=bins, log=True)
#         plt.title(f"Step {i}: Mass Distribution")
#         plt.xlabel("Mass")
#         plt.ylabel("Count (log scale)")
#         plt.savefig(f"frames/mass_hist_{i:04d}.png")
#         plt.close()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def plot_mass_histograms(mass_snapshots_initial, bins=100, interval=1, path="frames/mass_hist.png"):
    
    # Step 1: 全部扫描，统一 bin 范围
    all_masses_initial = np.concatenate([m[m < 1] for m in mass_snapshots_initial[::interval]])
    mass_min_initial = np.min(all_masses_initial)

    mass_snapshots = [m/mass_min_initial for m in mass_snapshots_initial[::interval]]
    all_masses = np.concatenate([m for m in mass_snapshots[::interval]])
    
    mass_min, mass_max = np.min(all_masses), np.max(all_masses)
    bin_edges = np.linspace(mass_min, mass_max, bins + 1)
    # 归一化处理

    # Step 2: 构造直方图矩阵
    hist_matrix = []
    for masses in mass_snapshots[::interval]:
        counts, _ = np.histogram(masses, bins=bin_edges)
        hist_matrix.append(counts)
    hist_matrix = np.array(hist_matrix)
    Z = np.log10(hist_matrix + 1)  # 避免 log(0)

    # Step 3: 网格和绘图
    T, M = np.meshgrid(
        np.arange(Z.shape[0]),
        0.5 * (bin_edges[1:] + bin_edges[:-1])
    )

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(M, T, Z.T, cmap='viridis', rstride=1, cstride=1)

    ax.set_xlabel('Mass')
    ax.set_ylabel('Time Step')
    ax.set_zlabel('Count')
    ax.set_title('Mass Histogram Surface Over Time')

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

# def plot_mass_histograms(mass_snapshots, bins=100, interval=1, path="frames/mass_hist.png"):
#     bin_edges = None
#     hist_matrix = []
#     for masses in mass_snapshots[::interval]:
#         masses = masses[masses < 1]  
#         print(np.min(masses), np.max(masses))
#         counts, edges = np.histogram(masses, bins=bins)
#         hist_matrix.append(counts)
#         if bin_edges is None:
#             bin_edges = edges

#     hist_matrix = np.array(hist_matrix)  # shape: (time_steps, bins)

#     # 构造网格
#     T, M = np.meshgrid(
#         np.arange(hist_matrix.shape[0]),                             
#         0.5 * (bin_edges[1:] + bin_edges[:-1])  # bin 中心
#     )

#     # 绘图
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(M, T, hist_matrix.T, cmap='viridis', rstride=1, cstride=1)

#     ax.set_xlabel('Mass')
#     ax.set_ylabel('Time Step')
#     ax.set_zlabel('Count')
#     ax.set_title('Mass Histogram Surface Over Time')

#     plt.tight_layout()
#     plt.savefig(path, dpi=300)
#     plt.close()

# def plot_mass_histograms(mass_snapshots, bins=20, interval=1):
#     from mpl_toolkits.mplot3d import Axes3D

#     bin_edges = None
#     hist_matrix = []
#     for masses in mass_snapshots[::interval]:
#         counts, edges = np.histogram(masses, bins=bins)
#         hist_matrix.append(counts)
#         if bin_edges is None:
#             bin_edges = edges

#     hist_matrix = np.array(hist_matrix)
#     T, M = np.meshgrid(np.arange(hist_matrix.shape[0]), 0.5 * (bin_edges[1:] + bin_edges[:-1]))

#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(M, T, hist_matrix.T, cmap='viridis')
#     ax.set_xlabel('Mass')
#     ax.set_ylabel('Time Step')
#     ax.set_zlabel('Count')
#     ax.set_title('Mass Histogram Surface Over Time')
#     # 保存图像
#     plt.tight_layout()
#     plt.savefig(f"frames/mass_hist.png", dpi=300)
#     plt.close()

def plot_num(mass_snapshots, interval=1, initial_num = 1000):
    asteroid_num = []
    for masses in mass_snapshots[::interval]:
        asteroid_num.append(len(masses))
    plt.figure(figsize=(10, 6))
    plt.plot(asteroid_num, linestyle='-', color='b')

    # 图形装饰
    plt.ylim(0, initial_num)
    plt.title("Asteroid Number Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Asteroid Num")
    plt.grid(True)
    plt.legend()

    # 显示或保存图像
    plt.tight_layout()
    plt.savefig(f"frames/num_plot.png", dpi=300)  # 可选保存




# def plot_mass_histograms(mass_snapshots, bins=50, interval=1):
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     bin_edges = None
#     for t_idx, masses in enumerate(mass_snapshots[::interval]):
#         counts, edges = np.histogram(masses, bins=bins)
#         if bin_edges is None:
#             bin_edges = edges
#         bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

#         xs = bin_centers
#         ys = np.full_like(xs, t_idx)
#         zs = np.zeros_like(xs)
#         dx = (bin_edges[1] - bin_edges[0]) * 0.9
#         dy = 0.9
#         dz = counts

#         ax.bar3d(xs, ys, zs, dx, dy, dz, shade=True)

#     ax.set_xlabel('Mass')
#     ax.set_ylabel('Time Step')
#     ax.set_zlabel('Count')
#     ax.set_title('3D Mass Distribution Over Time')
#     plt.tight_layout()
#     plt.show()
