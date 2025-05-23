from .particle_data import ParticleData
from .initial_conditions import generate_test_disk
from .simulation import Simulation
from cProfile import run
from .data_handler import DynamicWriter, DynamicLoader
# from .interactive_visualizaer import ThreeDVisualizer
from .interactive_visualizer_modified import ThreeDVisualizer
from PyQt5.QtWidgets import QApplication
import sys
from .plot_density import plot_density_surface, compute_neighbor_density_over_time
from .plot_mass_histograms import plot_mass_histograms, plot_num
import os

#def main():
    # --- Simulation Parameters ---

    # NUM_ASTEROIDS = 100     # Number of asteroids
    # MAX_PARTICLES = NUM_ASTEROIDS + 1 # Capacity slightly larger than needed
    # MIN_ORBIT_RADIUS = 0.98
    # MAX_ORBIT_RADIUS = 1.02
    # MIN_MASS = 1e-4
    # MAX_MASS = 1e-4
    # PERTURBATION_SCALE = 0.01
    # ETA_VALUE = 0.1
    # TIME_STEP = 0.001       # Simulation time step in years/2pi
    # NUM_STEPS = 5000       # Period of simulation
    # PLOT_INTERVAL = 1000      # Period of Saving plot
    # MAX_ANGLE = None

    # NUM_ASTEROIDS = 1000     # Number of asteroids
    # MAX_PARTICLES = NUM_ASTEROIDS + 1 # Capacity slightly larger than needed
    # MIN_ORBIT_RADIUS = 0.95
    # MAX_ORBIT_RADIUS = 1.05
    # MIN_MASS = 1e-4
    # MAX_MASS = 1e-3
    # PERTURBATION_SCALE = 0.1
    # ETA_VALUE = 0.3
    # TIME_STEP = 0.001       # Simulation time step in years/2pi
    # NUM_STEPS = 5000        # Period of simulation (default 3000)
    # PLOT_INTERVAL = 100       # Period of Saving plot
    # MAX_ANGLE = None

    # print("--- N-Body Simulation Setup ---")
    # print(f"Number of asteroids: {NUM_ASTEROIDS}")
    # print(f"Time step (dt): {TIME_STEP}")
    # print(f"Number of steps: {NUM_STEPS}")
    # print(f"Plot interval: {PLOT_INTERVAL}")

    # # 1. Generate Initial Conditions
    # particles = generate_test_disk(n_asteroids=NUM_ASTEROIDS, max_particles=MAX_PARTICLES, min_orbit_radius=MIN_ORBIT_RADIUS,
    #                                max_orbit_radius=MAX_ORBIT_RADIUS, min_mass=MIN_MASS, max_mass=MAX_MASS, perturbation_scale=PERTURBATION_SCALE, max_angle=MAX_ANGLE)
    # # particles = ParticleData(3)
    # # particles.add_particle([1, 0, 0], [0, 0, 0], 1, 0.1)
    # # particles.add_particle([-1, 0, 0], [0, 0, 0], 1, 0.1)

    # # 2. Create Simulation Instance
    # saver = DynamicWriter(filename='data.dat')
    # sim = Simulation(particles, saver=saver) # Uses default G and epsilon from simulation.py

    # # 3. Run Simulation
    # # NOTE: If you are using 'cpu_barnes_hut' as the backend, please adjust the hyper-parameter manually in physics.py
    # sim.run(dt_max=TIME_STEP, num_steps=NUM_STEPS, plot_interval=PLOT_INTERVAL,
    #         eta_adaptive_dt=ETA_VALUE, with_plot=False, backend=['cpu_barnes_hut','cpu_spatial_hash','cpu_spatial_hash'])

    # print("--- Simulation Complete ---")

    # os.makedirs("frames", exist_ok=True)
    # plot_mass_histograms(sim.mass_snapshots)
    # plot_num(sim.mass_snapshots, initial_num = NUM_ASTEROIDS)

    # density_data = compute_neighbor_density_over_time(sim.position_snapshots, radius=0.05)
    # plot_density_surface(density_data, bins=30, filename="density_surface.png")

    # print("--- Graph Complete ---")

# def main():
#     loader = DynamicLoader('data.dat')
#     visualizer = ThreeDVisualizer(loader)
#     visualizer.run()

def main():
    app = QApplication(sys.argv)
    loader = DynamicLoader('data.dat')
    visualizer = ThreeDVisualizer(loader, params)
    visualizer.run()

    # os.makedirs("frames", exist_ok=True)
    # plot_mass_histograms(sim.mass_snapshots)

    # density_data = compute_neighbor_density_over_time(sim.position_snapshots, radius=0.05)
    # plot_density_surface(density_data, bins=30, filename="density_surface.png")

if __name__ == "__main__":
    # run('main()')
    NUM_ASTEROIDS = 100     # Number of asteroids
    MAX_PARTICLES = NUM_ASTEROIDS + 1 # Capacity slightly larger than needed
    MIN_ORBIT_RADIUS = 0.98
    MAX_ORBIT_RADIUS = 1.02
    MIN_MASS = 1e-4
    MAX_MASS = 1e-4
    PERTURBATION_SCALE = 0.01
    ETA_VALUE = 0.1
    TIME_STEP = 0.001       # Simulation time step in years/2pi
    NUM_STEPS = 5000       # Period of simulation
    PLOT_INTERVAL = 1000      # Period of Saving plot
    MAX_ANGLE = None
    params = {
        "NUM_ASTEROIDS": NUM_ASTEROIDS,
        "MAX_PARTICLES": MAX_PARTICLES,
        "MIN_ORBIT_RADIUS": MIN_ORBIT_RADIUS,
        "MAX_ORBIT_RADIUS": MAX_ORBIT_RADIUS,
        "MIN_MASS": MIN_MASS,
        "MAX_MASS": MAX_MASS,
        "PERTURBATION_SCALE": PERTURBATION_SCALE,
        "ETA_VALUE": ETA_VALUE,
        "TIME_STEP": TIME_STEP,
        "NUM_STEPS": NUM_STEPS,
        "PLOT_INTERVAL": PLOT_INTERVAL,
        "MAX_ANGLE": MAX_ANGLE
    }
    main()
