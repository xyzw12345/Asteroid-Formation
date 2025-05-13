import numpy as np
from .particle_data import ParticleData

def check_for_overlaps(particles: ParticleData, step_num: int):
    """
    Basic O(N^2) overlap detection for reporting.
    Checks if distance between centers < sum of radii.
    """
    active_idx = particles.active_indices
    if active_idx.size < 2:
        return [] # Not enough particles to collide

    positions = particles.position[active_idx]
    radii = particles.radius[active_idx]
    original_ids = particles.ids[active_idx] # For reporting

    collided_pairs = [] # List to store (original_id1, original_id2)

    for i in range(len(active_idx)):
        for j in range(i + 1, len(active_idx)):
            diff = positions[i] - positions[j]
            dist_sq = np.sum(diff**2)
            sum_radii = radii[i] + radii[j]

            if dist_sq < sum_radii**2:
                # Overlap detected!
                dist = np.sqrt(dist_sq)
                print(f"Step {step_num}: OVERLAP! Particles {original_ids[i]} and {original_ids[j]} "
                      f"dist: {dist:.3e}, sum_radii: {sum_radii:.3e}")
                collided_pairs.append((original_ids[i], original_ids[j]))
    return collided_pairs

def get_min_pairwise_dist(particles: ParticleData):
    min_pairwise_dist = np.inf
    active_idx = particles.active_indices
    positions = particles.position[active_idx]
    for i in range(len(active_idx)):
        for j in range(i + 1, len(active_idx)):
            diff = positions[i] - positions[j]
            min_pairwise_dist = min(min_pairwise_dist, np.linalg.norm(diff))
    return min_pairwise_dist
