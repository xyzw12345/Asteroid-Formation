#pragma once
#include <vector>

// Forward declaration to avoid including pybind11 in .cu files directly if not needed there
// Or, if parameters are just raw pointers, pybind11 might not be needed in this header.
// For simplicity here, we assume raw pointers are passed from the binding layer.

namespace NBodyCUDA {

void compute_accelerations_cuda(
    double* d_accel_out, // Output: device pointer for accelerations (N, 3)
    const double* d_pos,   // Input: device pointer for positions (N, 3)
    const double* d_mass,  // Input: device pointer for masses (N)
    int num_particles,
    double G,
    double epsilon_sq // Pass epsilon squared
);

// For collision detection, we need to return pairs. This is tricky.
// Option A: Return a large boolean matrix (N,N) to CPU, then process on CPU. (Memory intensive)
// Option B: Identify pairs on GPU, compact them into a list, transfer list. (More complex kernel)
// Option C: For O(N^2), CPU version might be comparable if GPU version needs large matrix transfer.
// Let's start with returning the number of collisions and a small sample for now,
// or focus on a GPU version that calculates the (N,N) dist_sq matrix and we process it from there.

// For check_overlap, let's have it populate a device array of colliding pair indices (flat)
// and return the count. The caller will then transfer this small array.
struct GpuCollisionPair {
    int idx1;
    int idx2;
};

int find_colliding_pairs_cuda(
    GpuCollisionPair* d_colliding_pairs_buffer, // Output: device buffer to store pairs
    int max_pairs_capacity, // Capacity of the buffer
    const double* d_pos,    // Input: device pointer for positions (N, 3)
    const double* d_radii,  // Input: device pointer for radii (N)
    int num_particles
);


double get_min_pairwise_dist_sq_cuda(
    const double* d_pos,   // Input: device pointer for positions (N, 3)
    int num_particles
);

} // namespace NBodyCUDA