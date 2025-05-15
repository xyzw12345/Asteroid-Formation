import numpy as np
import matplotlib.pyplot as plt
import os
from .particle_data import ParticleData

OUTPUT_DIR = "./output" # Relative path to output directory from src/

MIN_SPEED = None
MAX_SPEED = None
MAX_RANGE = None

def plot_particles(particles: ParticleData, step: int, time: float, save: bool = True, final: bool = False):
    """
    Generates a 2D scatter plot of active particle positions (XY plane).
    Sizes points by mass (log scale) and colors by speed.
    """
    global MIN_SPEED, MAX_SPEED, MAX_RANGE
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    active_idx = particles.active_indices
    if active_idx.size == 0:
        print("No active particles to plot.")
        return

    pos = particles.position[active_idx]
    vel = particles.velocity[active_idx]
    mass = particles.mass[active_idx]

    # Calculate speeds for coloring
    speeds = np.linalg.norm(vel, axis=1)
    if MIN_SPEED is None:
       MIN_SPEED = 0.3 * (np.min(speeds) if speeds.size > 0 else 0)
    if MAX_SPEED is None:
       MAX_SPEED = 3 * (np.max(speeds) if speeds.size > 0 else 1)

    # Calculate sizes based on mass (use log scale for better visibility)
    # Avoid log(0) for potentially massless particles or very small masses
    min_mass_display = 1e-15
    log_mass = np.log10(np.maximum(mass, min_mass_display))
    min_log_mass = np.min(log_mass) if log_mass.size > 0 else -15
    max_log_mass = np.max(log_mass) if log_mass.size > 0 else -14

    # Normalize size: scale from a minimum size to a maximum size
    min_pt_size = 1
    max_pt_size = 50
    if max_log_mass > min_log_mass:
      sizes = min_pt_size + (log_mass - min_log_mass) / (max_log_mass - min_log_mass) * (max_pt_size - min_pt_size)
    else:
      sizes = np.full(log_mass.shape, (min_pt_size + max_pt_size) / 2.0) # All same size if masses equal


    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        pos[:, 0], pos[:, 1], # x and y coordinates
        s=sizes,
        c=np.clip(speeds, MIN_SPEED, MAX_SPEED),
        cmap='viridis', # Colormap for speed
        alpha=0.7
    )

    # Add a color bar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(f'Speed (min: {MIN_SPEED:.2e}, max: {MAX_SPEED:.2e})')

    # Set plot limits (optional, adjust based on expected system size)
    if MAX_RANGE is None:
        MAX_RANGE = (np.max(np.abs(pos[:, :2])) * 1.1 if pos.size > 0 else 5.0) * 1.5
    ax.set_xlim(-MAX_RANGE, MAX_RANGE)
    ax.set_ylim(-MAX_RANGE, MAX_RANGE)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    title_prefix = "Final State" if final else f"Step {step}"
    ax.set_title(f"{title_prefix}, Simulation Time: {time:.3f}, Active Particles: {len(active_idx)}")
    ax.grid(True, linestyle='--', alpha=0.5)

    if save:
        filename = os.path.join(OUTPUT_DIR, f"particles_{step:05d}.png")
        plt.savefig(filename, dpi=150)
        print(f"Saved plot: {filename}")

    plt.close(fig) # Close the figure to free memory