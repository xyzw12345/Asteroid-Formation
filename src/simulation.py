import time
import numpy as np
from .particle_data import ParticleData
from .physics import compute_accelerations_cpu
from .integrator import kick, drift
from .visualization import plot_particles # Assuming basic plotter exists

# Define constants here or import from a config file later
G_CONST = 6.6743e-8
EPSILON_SOFT = 1e-8 # Softening parameter

class Simulation:
    def __init__(self, particles: ParticleData, G: float = G_CONST, epsilon: float = EPSILON_SOFT):
        self.particles = particles
        self.G = G
        self.epsilon = epsilon
        self.time = 0.0 # Simulation time

    def run(self, dt: float, num_steps: int, plot_interval: int = 10):
        """
        Runs the N-body simulation.

        Args:
            dt: Time step size.
            num_steps: Number of steps to simulate.
            plot_interval: Save a plot every 'plot_interval' steps. Set to 0 or None to disable.
        """

        # NOTE [TODO]: dt has to be adaptive to prevent the 'slingshot' effect, to be refactored
        print(f"Starting simulation with N={self.particles.num_active_particles} active particles.")
        print(f"dt={dt}, num_steps={num_steps}, G={self.G}, epsilon={self.epsilon}")

        # Initial step: Compute initial accelerations & first half-kick
        compute_accelerations_cpu(self.particles, self.G, self.epsilon)
        kick(self.particles, dt / 2.0)
        self.time += dt / 2.0

        start_time = time.time()

        for step in range(num_steps):
            # 1. Drift particles
            drift(self.particles, dt)

            # 2. Compute new accelerations based on new positions
            compute_accelerations_cpu(self.particles, self.G, self.epsilon)

            # 3. Kick velocities
            kick(self.particles, dt)

            # 4. Update time
            self.time += dt

            # --- Intermediate Output/Visualization ---
            if plot_interval and (step + 1) % plot_interval == 0:
                step_end_time = time.time()
                steps_so_far = step + 1
                avg_time_per_step = (step_end_time - start_time) / steps_so_far
                print(f"Step {steps_so_far}/{num_steps}, Sim Time: {self.time:.3f}, "
                      f"Avg Step Time: {avg_time_per_step:.4f} s")
                # Add visualization call
                plot_particles(self.particles, step=steps_so_far, time=self.time, save=True)

            # --- TODO: Collision Detection & Handling would go here ---
            # colliding_pairs = detect_collisions(self.particles)
            # handle_mergers(self.particles, colliding_pairs)
            # If mergers happened, potentially recompute accelerations or compact array


        end_time = time.time()
        total_time = end_time - start_time
        print("\nSimulation finished.")
        print(f"Total steps: {num_steps}")
        print(f"Final simulation time: {self.time:.3f}")
        print(f"Total execution time: {total_time:.3f} s")
        if num_steps > 0:
             print(f"Average time per step: {total_time / num_steps:.4f} s")

        # Final plot
        if plot_interval:
             plot_particles(self.particles, step=num_steps, time=self.time, save=True, final=True)