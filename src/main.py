import os
import sys
import json
import numpy as np
from PyQt5.QtWidgets import QApplication
from concurrent.futures import ThreadPoolExecutor
from .particle_data import ParticleData
from .initial_conditions import generate_test_disk
from .simulation import Simulation
from cProfile import run
from .data_handler import DynamicWriter, DynamicLoader
# from .interactive_visualizaer import ThreeDVisualizer
from .interactive_visualizer_modified import ThreeDVisualizer
from .plot_mass_histograms import plot_mass_histograms, plot_num
from .plot_density import compute_neighbor_density_over_time, plot_density_surface
from PyQt5.QtWidgets import QApplication
import sys

def simulation(setting: json, run_index: int, verbose = False):
    print(str(setting))
    setting_index = int(setting['id'])
    NUM_ASTEROIDS = int(setting['num_asteroids'])    # Number of asteroids
    MAX_PARTICLES = NUM_ASTEROIDS + 1 # Capacity slightly larger than needed
    MIN_ORBIT_RADIUS = float(setting['min_orbit_radius'])
    MAX_ORBIT_RADIUS = float(setting['max_orbit_radius'])
    MIN_MASS = float(setting['min_mass'])
    MAX_MASS = float(setting['max_mass'])
    DENSITY = 9.280e6 if 'density' not in setting else float(setting['density'])
    PERTURBATION_SCALE = float(setting['perturbation_scale'])
    ETA_VALUE = float(setting['eta'])
    TIME_STEP = float(setting['default_time_step'])       # Simulation time step in years/2pi
    NUM_STEPS = int(setting['num_steps'])    # Period of simulation
    WITH_PLOT = bool(setting['with_plot'])
    PLOT_INTERVAL = int(setting['plot_interval'])     # Period of Saving plot
    MAX_ANGLE = 2 * np.pi if 'max_angle' not in setting else float(setting['max_angle'])
    BACKEND = setting['backend']

    if verbose:
        print("--- N-Body Simulation Setup ---")
        print(f"Number of asteroids: {NUM_ASTEROIDS}")
        print(f"Time step (dt): {TIME_STEP}")
        print(f"Number of steps: {NUM_STEPS}")
        print(f"Plot interval: {PLOT_INTERVAL}")

    # 1. Generate Initial Conditions
    particles = generate_test_disk(n_asteroids=NUM_ASTEROIDS, max_particles=MAX_PARTICLES, min_orbit_radius=MIN_ORBIT_RADIUS,
                                   max_orbit_radius=MAX_ORBIT_RADIUS, min_mass=MIN_MASS, max_mass=MAX_MASS, perturbation_scale=PERTURBATION_SCALE,
                                   density=DENSITY, max_angle=MAX_ANGLE)
    # particles = ParticleData(3)
    # particles.add_particle([1, 0, 0], [0, 0, 0], 1, 0.1)
    # particles.add_particle([-1, 0, 0], [0, 0, 0], 1, 0.1)

    # 2. Create Simulation Instance
    saver = DynamicWriter(filename=f'./visualization_data/{run_index}-{setting_index}-data.dat')
    sim = Simulation(particles, saver=saver) # Uses default G and epsilon from simulation.py

    # 3. Run Simulation
    # NOTE: If you are using 'cpu_barnes_hut' as the backend, please adjust the hyper-parameter manually in physics.py
    sim.run(dt_max=TIME_STEP, num_steps=NUM_STEPS, plot_interval=PLOT_INTERVAL,
            eta_adaptive_dt=ETA_VALUE, with_plot=WITH_PLOT, backend=BACKEND, verbose=verbose)
    
    plot_mass_histograms(sim.mass_snapshots, path=f"./visualization_data/{run_index}-{setting_index}-mass_histogram.png")

    density_data = compute_neighbor_density_over_time(sim.position_snapshots, radius=0.05)
    plot_density_surface(density_data, bins=30, filename=f"./visualization_data/{run_index}-{setting_index}-density_surface.png")

    # if verbose:
    #     print("--- Simulation Complete ---")
    #     app = QApplication(sys.argv)
    #     loader = DynamicLoader(f"./visualization_data/{run_index}-{setting_index}-data.dat")
    #     visualizer = ThreeDVisualizer(loader)
    #     visualizer.run()

if __name__ == "__main__":
    with open('./initial_conditions/1.json', "r") as file:
        json_settings = json.load(file)
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(simulation, json_setting, 1, verbose=True) for json_setting in json_settings]
    app = QApplication(sys.argv)
    loader = DynamicLoader(f"./visualization_data/1-2-data.dat")
    visualizer = ThreeDVisualizer(loader)
    visualizer.run()
    # for json_setting in json_settings:
    #     simulation(json_setting, run_index=1, verbose=True)
    # run('main()')
    # main()
