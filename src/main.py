from .particle_data import ParticleData
from .initial_conditions import generate_test_disk
from .simulation import Simulation

def main():
    # --- Simulation Parameters ---
    NUM_ASTEROIDS = 100       # Number of asteroids
    MAX_PARTICLES = NUM_ASTEROIDS + 1 # Capacity slightly larger than needed
    TIME_STEP = 0.001       # Simulation time step (e.g., in years/2pi if G=1, SunMass=1, Dist=AU)
    NUM_STEPS = 30000        # Period of simulation
    PLOT_INTERVAL = 400       # Period of Saving plot

    print("--- N-Body Simulation Setup ---")
    print(f"Number of asteroids: {NUM_ASTEROIDS}")
    print(f"Time step (dt): {TIME_STEP}")
    print(f"Number of steps: {NUM_STEPS}")
    print(f"Plot interval: {PLOT_INTERVAL}")

    # 1. Generate Initial Conditions
    particles = generate_test_disk(n_asteroids=NUM_ASTEROIDS, max_particles=MAX_PARTICLES)
    # particles = ParticleData(2)
    # particles.add_particle([0, 0, 0], [0, 0, 0], 1, 1e-5)
    # particles.add_particle([1e2, 0, 0], [0, 0, 0], 1, 1e-5)

    # 2. Create Simulation Instance
    sim = Simulation(particles) # Uses default G and epsilon from simulation.py

    # 3. Run Simulation
    sim.run(dt_max=TIME_STEP, num_steps=NUM_STEPS, plot_interval=PLOT_INTERVAL)

    print("--- Simulation Complete ---")

if __name__ == "__main__":
    main()