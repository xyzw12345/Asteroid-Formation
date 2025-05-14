import numpy as np
from .particle_data import ParticleData

# Throughout this project, we set G = SUN_MASS = AU = 1, which makes the unit of time to be [year / (2 * pi)]

G = 1  # Gravitational constant
SUN_MASS = 1 # Mass of the central star
DENSITY = 9.280e6 # Density of Earth

def generate_test_disk(n_asteroids: int, max_particles: int, min_orbit_radius: float, max_orbit_radius: float,
                       min_mass: float, max_mass: float, perturbation_scale: float) -> ParticleData:
    """
    Generates a central star and a disk of asteroids in the XY plane
    with roughly circular Keplerian velocities.
    Args:
        n_asteroids: Number of asteroids to generate.
        max_particles: Total capacity for the ParticleData structure.
    Returns:
        ParticleData instance populated with the sun and asteroids.
    """
    if max_particles < n_asteroids + 1:
        raise ValueError("max_particles must be >= n_asteroids + 1")

    particles = ParticleData(capacity=max_particles)

    # Add the Sun (stationary at the center)
    particles.add_particle(pos=[0, 0, 0], vel=[0, 0, 0], mass=SUN_MASS, radius=1e-8) # Small radius for vis

    for _ in range(n_asteroids):
        # Position
        r = np.random.uniform(min_orbit_radius, max_orbit_radius)
        theta = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.random.normal(0, 0.01 * r) # Small vertical perturbation

        # Velocity (Keplerian + small random perturbation)
        speed_circ = np.sqrt(G * SUN_MASS / r)
        vx = -speed_circ * np.sin(theta)
        vy = speed_circ * np.cos(theta)
        vz = 0
        # Add small random velocity component (e.g., 10% of circular speed)
        vx += np.random.normal(0, perturbation_scale * speed_circ)
        vy += np.random.normal(0, perturbation_scale * speed_circ)
        vz += np.random.normal(0, perturbation_scale * speed_circ * 0.1) # Smaller z velocity perturbation

        # Mass and Radius
        mass = np.random.uniform(min_mass, max_mass)
        radius = (3 * mass / (4 * np.pi * DENSITY))**(1./3.) 

        particles.add_particle(pos=[x, y, z], vel=[vx, vy, vz], mass=mass, radius=radius)

    print(f"Generated {particles.num_active_particles} particles (1 Sun + {n_asteroids} asteroids).")
    return particles