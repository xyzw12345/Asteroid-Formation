import numpy as np
from .particle_data import ParticleData

# Leapfrog Integrator (Kick-Drift-Kick variant)

def kick(particles: ParticleData, dt: float):
    """Velocity update step (Kick)."""
    active_idx = particles.active_indices
    if active_idx.size > 0: # Check if there are any active particles
        particles.velocity[active_idx] += particles.acceleration[active_idx] * dt

def drift(particles: ParticleData, dt: float):
    """Position update step (Drift)."""
    active_idx = particles.active_indices
    if active_idx.size > 0:
        particles.position[active_idx] += particles.velocity[active_idx] * dt
