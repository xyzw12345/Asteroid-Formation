import sys
import json
import numpy as np
from cProfile import run
from concurrent.futures import ThreadPoolExecutor
from PyQt5.QtWidgets import QApplication
from .initial_conditions import generate_test_disk
from .simulation import Simulation
from .data_handler import DynamicWriter, DynamicLoader
from .interactive_visualizer import ThreeDVisualizer
from .plot_mass_histograms import plot_mass_histograms, plot_num, plot_log_log
from .plot_density import compute_neighbor_density_over_time, plot_density_surface
from .analyze_tree import analyze_tree_structure

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
    TIME_PERIOD = float(setting['time_period'])
    WITH_PLOT = bool(setting['with_plot'])
    PLOT_INTERVAL = float(setting['plot_interval'])     # Period of Saving plot
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
    sim.run(dt_max=TIME_STEP, num_steps=NUM_STEPS, time_period=TIME_PERIOD, plot_interval=PLOT_INTERVAL,
            eta_adaptive_dt=ETA_VALUE, with_plot=WITH_PLOT, backend=BACKEND, verbose=verbose)
    
    print(np.sort(particles.mass[particles.active_indices])[-100:])
    plot_mass_histograms(sim.mass_snapshots, path=f"./visualization_data/{run_index}-{setting_index}-mass_histogram.png")
    plot_log_log(sim.mass_snapshots, path=f"./visualization_data/{run_index}-{setting_index}-log_log.png")

    density_data = compute_neighbor_density_over_time(sim.position_snapshots, radius=0.05)
    plot_density_surface(density_data, bins=30, filename=f"./visualization_data/{run_index}-{setting_index}-density_surface.png")

    stats = analyze_tree_structure(sim.particles.tree_structure)
    for (depth, leaves), count in sorted(stats[0].items()):
        print(f"Depth = {depth}, Leaves = {leaves} → {count} trees")
    print(sorted(stats[1]))

if __name__ == "__main__":
    s = input().strip().split()
    if s[0] == 'sim':
        i, j = int(s[1]), int(s[2])
        with open('./initial_conditions/1.json', "r") as file:
            json_settings = json.load(file)
        for json_setting in json_settings:
                if i <= int(json_setting['id']) and int(json_setting['id']) <= j:
                    simulation(json_setting, run_index=1, verbose=True)
            
    if s[0] == 'profile':
        i, j = int(s[1]), int(s[2])
        with open('./initial_conditions/1.json', "r") as file:
            json_settings = json.load(file)
        for json_setting in json_settings:
            if i <= int(json_setting['id']) and int(json_setting['id']) <= j:
                run('simulation(json_setting, run_index=1, verbose=True)')
    elif s[0] == 'show':
        i = int(s[1])
        app = QApplication(sys.argv)
        loader = DynamicLoader(f"./visualization_data/1-{i}-data.dat")
        with open('initial_conditions/1.json', 'r') as f:
            data = json.load(f)
        params = [json.dumps(entry, indent=4) for entry in data if entry['id'] == i]
        setting = json.loads(params[0])
        visualizer = ThreeDVisualizer(filename = f"./visualization_data/1-{i}-data.dat", sim_callback = loader, params=setting)
        visualizer.run()
