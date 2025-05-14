import numpy as np

def compute_accelerations_cpu_numpy(positions: np.ndarray, masses: np.ndarray, G: float = 1.0, epsilon: float = 0.0001):
    """
    Computes gravitational acceleration on all *active* particles using
    direct N^2 summation on the CPU (partially vectorized with numpy).
    """

    accel = np.zeros_like(positions)

    # Use a broadcasted approach within the loop for acceleration calculation
    # Outer loop iterates through active particles computing their acceleration
    for i in range(len(positions)):
        # Vectorized difference: pos_all - pos_i
        diff = positions - positions[i] # Shape (num_active, 3)

        # Squared distances (avoiding self-interaction implicitly)
        dist_sq = np.sum(diff**2, axis=1) # Shape (num_active,)

        # Add softening factor
        dist_sq_softened = np.clip(dist_sq, epsilon**2, None)

        # Calculate 1 / dist^3, handling potential division by zero carefully
        # Mask out the self-interaction (where dist_sq is zero before softening)
        # dist_sq_soft will be epsilon^2 for the i=i case, which is non-zero
        inv_dist_cubed = dist_sq_softened**(-1.5)

        # Mask for j != i (already handled by the structure of diff calculation)
        # The 'i_idx'-th element corresponds to the self-interaction.

        # Calculate acceleration contributions: G * mass[j] * diff_vec * inv_dist_cubed
        # masses[active_indices] gives masses of all active particles
        # We need shape (num_active, 1) for broadcasting with diff (num_active, 3)
        masses_j = masses[:, np.newaxis] # Shape (num_active, 1)
        inv_dist_cubed_j = inv_dist_cubed[:, np.newaxis] # Shape (num_active, 1)

        # Sum contributions G * m_j * (r_j - r_i) / |r_j - r_i|^3
        # Note: diff[k] = pos[active_indices[k]] - pos[i]
        #       masses_j[k] = mass[active_indices[k]]
        accel_contrib = G * masses_j * diff * inv_dist_cubed_j # Shape (num_active, 3)

        # Sum contributions from all other active particles (j != i)
        # We can achieve this by setting the contribution from i onto itself to zero
        accel_contrib[i] = np.zeros(3) # Zero out self-contribution explicitly
        accel[i] = np.sum(accel_contrib, axis=0)
    return accel

def check_for_overlaps_cpu_numpy(positions: np.ndarray, radii: np.ndarray):
    """
    Basic O(N^2) overlap detection for reporting.
    Checks if distance between centers < sum of radii.
    """
    diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    sum_of_radii = radii[np.newaxis, :] + radii[:, np.newaxis]  

    row_index, column_index = np.where(dist < sum_of_radii)

    return zip(row_index, column_index)

def get_min_dist_cpu_numpy(positions: np.ndarray):
    """
    Basic O(N^2) calculation for giving minimum pairwise distance.
    """
    diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    return np.min(dist, axis=1)
