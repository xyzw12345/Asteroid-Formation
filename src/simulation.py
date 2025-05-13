import time
import numpy as np
from .particle_data import ParticleData
from .physics import compute_accelerations_cpu
from .collision_detection import check_for_overlaps, get_min_pairwise_dist
from .integrator import kick, drift
from .visualization import plot_particles
from .interactive_visualizer import InteractiveVisualizerVisPy

# Define constants here or import from a config file later
G_CONST = 1
EPSILON_SOFT = 1e-8 # Softening parameter

DEFAULT_ETA = 0.05 # Accuracy parameter for adaptive timestep
MIN_DT_FACTOR = 1e-8 # Factor of user_dt to prevent overly small dt

class Simulation:
    def __init__(self, particles: ParticleData, G: float = G_CONST, epsilon: float = EPSILON_SOFT):
        self.particles = particles
        self.G = G
        self.epsilon = min(epsilon, np.min(self.particles.radius))
        self.time = 0.0

    def _calculate_adaptive_dt(self, user_dt: float, eta: float = DEFAULT_ETA) -> float:
        """
        Calculates an adaptive timestep based on particle accelerations and velocities.
        Returns the effective timestep to use.
        """
        active_idx = self.particles.active_indices
        if active_idx.size == 0:
            return user_dt

        accel_mag = np.linalg.norm(self.particles.acceleration[active_idx], axis=1)
        vel_mag = np.linalg.norm(self.particles.velocity[active_idx], axis=1)

        characteristic_length = get_min_pairwise_dist(self.particles)

        # Avoid division by zero if acc or vel is zero
        # dt based on acceleration: dt_a ~ eta * sqrt(characteristic_length / |a|)
        dt_accel = np.full_like(accel_mag, np.inf)
        # A small constant to prevent dt_accel from being huge if accel_mag is tiny
        min_accel_for_dt = 1e-9
        valid_accel_mask = accel_mag > min_accel_for_dt
        if np.any(valid_accel_mask):
            dt_accel[valid_accel_mask] = eta * np.sqrt(characteristic_length / accel_mag[valid_accel_mask])
        
        # dt based on velocity: dt_v ~ eta * characteristic_length / |v|
        dt_vel = np.full_like(vel_mag, np.inf)
        min_vel_for_dt = 1e-9 # A small constant
        valid_vel_mask = vel_mag > min_vel_for_dt
        if np.any(valid_vel_mask):
            # Using epsilon as the length scale. Could use particle.radius[active_idx]
            dt_vel[valid_vel_mask] = eta * characteristic_length / vel_mag[valid_vel_mask]

        min_dt_crit = np.inf
        if dt_accel.size > 0: min_dt_crit = min(min_dt_crit, np.min(dt_accel))
        if dt_vel.size > 0: min_dt_crit = min(min_dt_crit, np.min(dt_vel))

        # Ensure dt doesn't become excessively small or larger than user_dt
        effective_dt = np.clip(min_dt_crit, user_dt * MIN_DT_FACTOR, user_dt)

        return effective_dt
    
    def _simulation_single_step(self, dt_max, eta_adaptive_dt):
        """Performs a single effective KDK step of the simulation."""
        # This logic is extracted from the loop in the original run method

        # If this is the very first call to _simulation_single_step
        if self.time == 0:
            compute_accelerations_cpu(self.particles, self.G, self.epsilon)
            dt_eff_step = self._calculate_adaptive_dt(dt_max, eta=eta_adaptive_dt)
            kick(self.particles, dt_eff_step / 2.0)
            self.current_dt_eff = dt_eff_step # Store for use in drift
        else:
            # Accelerations are from the end of the previous call's compute_A
            # The first kick of the KDK uses previous step's accel.
            self.current_dt_eff = self._calculate_adaptive_dt(dt_max, eta=eta_adaptive_dt)
            kick(self.particles, self.current_dt_eff / 2.0)

        drift(self.particles, self.current_dt_eff)
        compute_accelerations_cpu(self.particles, self.G, self.epsilon) # New accelerations
        kick(self.particles, self.current_dt_eff / 2.0) # Second half kick

        self.time += self.current_dt_eff # Actual time advanced
        
        colliding_pairs = check_for_overlaps(self.particles, self.time)
        for i, j in colliding_pairs:
            self.particles.merge(i, j)
    
    def run(self, dt_max: float, num_steps: int, plot_interval: int = 10, eta_adaptive_dt: float = DEFAULT_ETA):
        """
        Runs the N-body simulation with adaptive timestepping.

        Args:
            dt_max: Maximum time step size for one "major" step.
            num_steps: Number of major steps to simulate.
            plot_interval: Save a plot every 'plot_interval' major steps.
            eta_adaptive_dt: Accuracy parameter for adaptive timestepping.
        """
        print(f"Starting simulation with N={self.particles.num_active_particles} active particles.")
        print(f"Max dt={dt_max}, num_steps={num_steps}, G={self.G}, epsilon={self.epsilon}, eta={eta_adaptive_dt}")

        # Initial step: Compute initial accelerations & first half-kick
        # Need to do this with an initial small dt or dt_max/2
        compute_accelerations_cpu(self.particles, self.G, self.epsilon)
        # For initial kick, we could use dt_max/2 or an adaptively calculated one.
        # Let's calculate an initial dt_eff for the first half kick
        initial_dt_eff_kick = self._calculate_adaptive_dt(dt_max, eta=eta_adaptive_dt) / 2.0
        kick(self.particles, initial_dt_eff_kick) # Time only advances by dt_eff in the loop for full steps

        start_time_sim = time.time()
        total_substeps = 0

        for step in range(num_steps):         
            self._simulation_single_step(dt_max, eta_adaptive_dt)
            total_substeps += 1 # In this scheme, one "substep" is one full adaptive step.

            # --- Intermediate Output/Visualization ---
            if plot_interval and (step + 1) % plot_interval == 0:
                step_end_time_sim = time.time()
                steps_so_far = step + 1
                avg_time_per_major_step = (step_end_time_sim - start_time_sim) / steps_so_far
                print(f"Step {steps_so_far}/{num_steps}, Sim Time: {self.time:.3e}, "
                      f"Avg Step Time: {avg_time_per_major_step:.4f} s")
                plot_particles(self.particles, step=steps_so_far, time=self.time, save=True)


        end_time_sim = time.time()
        total_time_sim = end_time_sim - start_time_sim
        print("\nSimulation finished.")
        print(f"Total major steps: {num_steps}, Total effective substeps: {total_substeps}")
        print(f"Final simulation time: {self.time:.3e}")
        print(f"Total execution time: {total_time_sim:.3f} s")
        if num_steps > 0:
             print(f"Average time per major step: {total_time_sim / num_steps:.4f} s")

        if plot_interval:
             plot_particles(self.particles, step=num_steps, time=self.time, save=True, final=True)

    def run_interactive(self, dt_max_vis: float, eta_adaptive_dt: float = DEFAULT_ETA):
        """
        Runs the simulation with an interactive VisPy visualizer.
        Each call to the visualizer's timer will advance the simulation one effective step.
        """
        print(f"Starting INTERACTIVE simulation with N={self.particles.num_active_particles} active particles.")
        print(f"Max dt for visualization steps={dt_max_vis}, G={self.G}, epsilon={self.epsilon}, eta={eta_adaptive_dt}")

        self.vis = InteractiveVisualizerVisPy(self.particles)
        
        # Define the callback for the visualizer
        def advance_sim_and_log():
            step_start_sim_time_ns = time.perf_counter_ns()
            
            # Perform one simulation step
            self._simulation_single_step(dt_max_vis, eta_adaptive_dt)
            
            step_end_sim_time_ns = time.perf_counter_ns()
            sim_duration_ms = (step_end_sim_time_ns - step_start_sim_time_ns) / 1e6
            
            # Rudimentary logging every N calls or M seconds to avoid spamming console
            if not hasattr(self, '_vis_step_count'): self._vis_step_count = 0
            self._vis_step_count += 1
            if self._vis_step_count % 60 == 0: # Log every ~60 frames
                 print(f"Sim Time: {self.time:.3e}, Last dt_eff: {self.current_dt_eff:.2e}, Sim Step Calc Time: {sim_duration_ms:.2f} ms")

        self.vis.set_simulation_step_callback(advance_sim_and_log)
        self.vis.update_particle_data() # Initial data load
        self.vis.start_animation() # This will block until the window is closed

        print("\nInteractive simulation finished or window closed.")
        # Potentially save final state or do other cleanup
        final_pos = self.particles.get_active_positions()
        if final_pos.size > 0:
            # Example: save final state using the static plotter
            # plot_particles(self.particles, step=-1, time=self.time, save=True, final=True)
            pass
    