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

def plot_log_log(mass_snapshots_initial, bins=100, interval=10, path="frames/mass_loglog.png"):
    # Step 1: 统一归一化
    all_masses_initial = np.concatenate([m[m < 1] for m in mass_snapshots_initial[::interval]])
    mass_min_initial = np.min(all_masses_initial)
    mass_snapshots = [m / mass_min_initial for m in mass_snapshots_initial[::interval]]

    # Step 2: 统一 bin
    all_masses = np.concatenate(mass_snapshots)
    mass_min, mass_max = np.min(all_masses), np.max(all_masses)
    bin_edges = np.logspace(np.log10(mass_min), np.log10(mass_max), bins + 1)  # log scale bin

    # Step 3: 绘制 log-log 图
    plt.figure(figsize=(10, 6))
    for i, masses in enumerate(mass_snapshots):
        counts, edges = np.histogram(masses, bins=bin_edges)
        centers = 0.5 * (edges[1:] + edges[:-1])

        # 过滤掉为0的 bin 以避免 log(0)
        mask = counts > 0
        plt.plot(centers[mask], counts[mask], label=f"Step {i * interval}", alpha=0.7)

    # 设置 log-log 轴和图形装饰
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Normalized Mass (log scale)')
    plt.ylabel('Count (log scale)')
    plt.title('Mass Distribution Over Time (log-log)')
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.grid(True, which="both", ls='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
