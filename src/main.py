from .particle_data import ParticleData
from .initial_conditions import generate_test_disk
from .simulation import Simulation
from cProfile import run
from .data_handler import DynamicWriter, DynamicLoader
from .interactive_visualizaer import ThreeDVisualizer

def main():
    # --- Simulation Parameters ---
    NUM_ASTEROIDS = 100     # Number of asteroids
    MAX_PARTICLES = NUM_ASTEROIDS + 1 # Capacity slightly larger than needed
    MIN_ORBIT_RADIUS = 0.99
    MAX_ORBIT_RADIUS = 1.01
    MIN_MASS = 1e-6
    MAX_MASS = 3e-6
    PERTURBATION_SCALE = 0.01
    ETA_VALUE = 0.1
    TIME_STEP = 0.001       # Simulation time step in years/2pi
    NUM_STEPS = 50000       # Period of simulation
    PLOT_INTERVAL = 100       # Period of Saving plot

    print("--- N-Body Simulation Setup ---")
    print(f"Number of asteroids: {NUM_ASTEROIDS}")
    print(f"Time step (dt): {TIME_STEP}")
    print(f"Number of steps: {NUM_STEPS}")
    print(f"Plot interval: {PLOT_INTERVAL}")

    # 1. Generate Initial Conditions
    particles = generate_test_disk(n_asteroids=NUM_ASTEROIDS, max_particles=MAX_PARTICLES, min_orbit_radius=MIN_ORBIT_RADIUS,
                                   max_orbit_radius=MAX_ORBIT_RADIUS, min_mass=MIN_MASS, max_mass=MAX_MASS, perturbation_scale=PERTURBATION_SCALE)
    # particles = ParticleData(3)
    # particles.add_particle([1, 0, 0], [0, 0, 0], 1)
    # particles.add_particle([-1, 0, 0], [0, 0, 0], 1)

    # 2. Create Simulation Instance
    saver = DynamicWriter(filename='data.dat')
    sim = Simulation(particles, saver=saver) # Uses default G and epsilon from simulation.py

    # 3. Run Simulation
    # NOTE: If you are using 'cpu_barnes_hut' as the backend, please adjust the hyper-parameter manually in physics.py
    sim.run(dt_max=TIME_STEP, num_steps=NUM_STEPS, plot_interval=PLOT_INTERVAL,
            eta_adaptive_dt=ETA_VALUE, with_plot=False, backend='cpu_numpy')

    print("--- Simulation Complete ---")
'''def main():
    loader = DynamicLoader('data.dat')
    visualizer = ThreeDVisualizer(loader)
    visualizer.run()'''

    # os.makedirs("frames", exist_ok=True)
    # plot_mass_histograms(sim.mass_snapshots)

    # density_data = compute_neighbor_density_over_time(sim.position_snapshots, radius=0.05)
    # plot_density_surface(density_data, bins=30, filename="density_surface.png")

if __name__ == "__main__":
    # run('main()')
    main()
