import numpy as np
from .particle_data import ParticleData

# Throughout this project, we set G = SUN_MASS = AU = 1, which makes the unit of time to be [year / (2 * pi)]

G = 1  # Gravitational constant
SUN_MASS = 1 # Mass of the central star

def generate_test_disk(n_asteroids: int, max_particles: int) -> ParticleData:
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

    # Add asteroids
    min_radius = 0.95 # Inner edge of the disk (AU)
    max_radius = 1.05 # Outer edge of the disk (AU)
    asteroid_mass_min = 1e-12 * SUN_MASS # Very small mass relative to the sun
    asteroid_mass_max = 1e-9 * SUN_MASS
    density = 9.280e6 # taken to be the density of the earth

    for _ in range(n_asteroids):
        # Position
        r = np.random.uniform(min_radius, max_radius)
        theta = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.random.normal(0, 0.01 * r) # Small vertical perturbation

        # Velocity (Keplerian + small random perturbation)
        speed_circ = np.sqrt(G * SUN_MASS / r)
        vx = -speed_circ * np.sin(theta)
        vy = speed_circ * np.cos(theta)
        vz = 0
        # Add small random velocity component (e.g., 1% of circular speed)
        perturbation_scale = 0.01
        vx += np.random.normal(0, perturbation_scale * speed_circ)
        vy += np.random.normal(0, perturbation_scale * speed_circ)
        vz += np.random.normal(0, perturbation_scale * speed_circ * 0.1) # Smaller z velocity perturbation

        # Mass and Radius
        mass = np.random.uniform(asteroid_mass_min, asteroid_mass_max)
        # Radius = (3 * mass / (4 * pi * density))^(1/3) 
        radius = (3 * mass / (4 * np.pi * density))**(1./3.) 

        particles.add_particle(pos=[x, y, z], vel=[vx, vy, vz], mass=mass, radius=radius)

    print(f"Generated {particles.num_active_particles} particles (1 Sun + {n_asteroids} asteroids).")
    return particles