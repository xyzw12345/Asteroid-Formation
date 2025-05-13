from .particle_data import ParticleData
from .initial_conditions import generate_test_disk
from .simulation import Simulation
from cProfile import run

def main():
    # --- Simulation Parameters ---
    NUM_ASTEROIDS = 1000       # Number of asteroids
    MAX_PARTICLES = NUM_ASTEROIDS + 1 # Capacity slightly larger than needed
    MIN_ORBIT_RADIUS = 0.95
    MAX_ORBIT_RADUIS = 1.05
    MIN_MASS = 1e-6
    MAX_MASS = 3e-6
    PERTURBATION_SCALE = 0.3
    ETA_VALUE = 0.3
    TIME_STEP = 0.001       # Simulation time step (e.g., in years/2pi if G=1, SunMass=1, Dist=AU)
    NUM_STEPS = 30000        # Period of simulation
    PLOT_INTERVAL = 400       # Period of Saving plot

    print("--- N-Body Simulation Setup ---")
    print(f"Number of asteroids: {NUM_ASTEROIDS}")
    print(f"Time step (dt): {TIME_STEP}")
    print(f"Number of steps: {NUM_STEPS}")
    print(f"Plot interval: {PLOT_INTERVAL}")

    # 1. Generate Initial Conditions
    particles = generate_test_disk(n_asteroids=NUM_ASTEROIDS, max_particles=MAX_PARTICLES, min_orbit_radius=MIN_ORBIT_RADIUS,
                                   max_orbit_radius=MAX_ORBIT_RADUIS, min_mass=MIN_MASS, max_mass=MAX_MASS, perturbation_scale=PERTURBATION_SCALE)

    # 2. Create Simulation Instance
    sim = Simulation(particles) # Uses default G and epsilon from simulation.py

    # 3. Run Simulation
    sim.run(dt_max=TIME_STEP, num_steps=NUM_STEPS, plot_interval=PLOT_INTERVAL, eta_adaptive_dt=ETA_VALUE)
    # sim.run_interactive(dt_max_vis=TIME_STEP)

    print("--- Simulation Complete ---")

if __name__ == "__main__":
    run('main()')