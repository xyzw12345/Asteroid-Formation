from .particle_data import ParticleData
import numpy as np
from .computation.cpu_numpy import compute_accelerations_cpu_numpy, check_for_overlaps_cpu_numpy, get_min_dist_cpu_numpy
from .computation.build.cpp_nbody_lib import compute_accelerations_cuda_n2 as compute_accelerations_cuda_n2
from .computation.build.cpp_nbody_lib import find_colliding_pairs_cuda_n2 as check_for_overlaps_cuda_n2
from .computation.build.cpp_nbody_lib import get_min_dist_sq_cuda_n2 as get_min_dist_sq_cuda_n2
from .computation.build.cpp_nbody_lib import find_colliding_pairs_cpu_sh as check_for_overlaps_cpu_sh
from .computation.build.cpp_nbody_lib import get_min_dist_cpu_sh as get_min_dist_cpu_sh
from .computation.build.cpp_nbody_lib import compute_accelerations_cpu_barnes_hut as compute_accelerations_cpu_barnes_hut


def compute_accelerations(particles: ParticleData, G: float = 1.0, epsilon: float = 0.0001, backend='cpu_numpy'):
    """
    Computes gravitational acceleration on all *active* particles.

    Args:
        particles: ParticleData object containing particle states.
        G: Gravitational constant.
        epsilon: Softening length to avoid singularities.
    """
    
    # Get indices of active particles
    active_indices = particles.active_indices
    positions = particles.position[active_indices]
    masses = particles.mass[active_indices]

    # Reset accelerations only for active particles
    particles.acceleration[active_indices] = 0.0

    accel = None
    if backend[0] == 'cpu_numpy':
        accel = compute_accelerations_cpu_numpy(positions, masses, G, epsilon)
    elif backend[0] == 'cuda_n2':
        accel = compute_accelerations_cuda_n2(positions, masses, G, epsilon)
    elif backend[0] == 'cpu_barnes_hut':
        accel = compute_accelerations_cpu_barnes_hut(positions, masses, G, epsilon, 0.1)
        # result1 = compute_accelerations_cuda_n2(positions, masses, G, epsilon)
        # sun_dist = positions - positions[0]
        # sun_dist[0] = np.full(3, np.inf)
        # sun_dist = np.sum(sun_dist**2, axis=1)
        # sun_dist = np.clip(sun_dist, epsilon * epsilon, None)
        # mag = G * masses[0] / ((sun_dist) ** (1.5))
        # sun_accel = mag[:, np.newaxis] * (positions - positions[0])
        # print(np.max(np.abs(result1 - sun_accel)))
        # print(np.max(np.abs(result1)))
        # print(np.max(np.abs((result1 - accel))))
    else:
        raise RuntimeError(f"Invalid backend selection, got {backend[0]}")
    
    for i, idx in enumerate(active_indices):
        particles.acceleration[idx] = accel[i]


def check_for_overlaps(particles: ParticleData, backend='cpu_numpy'):
    """
    Overlap detection for reporting. Checks if distance between centers < sum of radii.
    """
    active_idx = particles.active_indices
    if active_idx.size < 2:
        return [] # Not enough particles to collide

    positions = particles.position[active_idx]
    radii = particles.radius[active_idx]
    original_ids = particles.ids[active_idx] # For reporting

    collided_pairs = []
    if backend[1] == 'cpu_numpy':
        collided_pairs = check_for_overlaps_cpu_numpy(positions, radii)
    elif backend[1] == 'cuda_n2':
        collided_pairs = check_for_overlaps_cuda_n2(positions, radii)
    elif backend[1] == 'cpu_spatial_hash':
        collided_pairs = check_for_overlaps_cpu_sh(positions, radii, 1e-3)
    else:
        raise RuntimeError(f"Invalid backend selection, got {backend[1]}")

    return [(original_ids[i], original_ids[j]) for i, j in collided_pairs]

def get_min_dist(particles: ParticleData, backend='cpu_numpy'):
    """
    Calculation for giving minimum pairwise distance.
    """
    active_idx = particles.active_indices
    positions = particles.position[active_idx]
    if backend[2] == 'cpu_numpy':
        return get_min_dist_cpu_numpy(positions)
    elif backend[2] == 'cuda_n2':
        return np.sqrt(get_min_dist_sq_cuda_n2(positions))
    elif backend[2] == 'cpu_spatial_hash':
        return get_min_dist_cpu_sh(positions, 1e-3)
    else:
        raise RuntimeError(f"Invalid backend selection, got {backend[2]}")
