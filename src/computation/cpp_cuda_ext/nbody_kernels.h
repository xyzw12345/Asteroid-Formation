#pragma once
#include <vector>

namespace NBodyCUDA {

void compute_accelerations_cuda_n2(
    double* d_accel_out, // Output: device pointer for accelerations (N, 3)
    const double* d_pos,   // Input: device pointer for positions (N, 3)
    const double* d_mass,  // Input: device pointer for masses (N)
    int num_particles,
    double G,
    double epsilon_sq // Pass epsilon squared
);

struct GpuCollisionPair {
    int idx1;
    int idx2;
};

int find_colliding_pairs_cuda_n2(
    GpuCollisionPair* d_colliding_pairs_buffer, // Output: device buffer to store pairs
    int max_pairs_capacity, // Capacity of the buffer
    const double* d_pos,    // Input: device pointer for positions (N, 3)
    const double* d_radii,  // Input: device pointer for radii (N)
    int num_particles
);


void get_min_dist_sq_cuda_n2(
    double* d_min_dist_out,
    const double* d_pos,   // Input: device pointer for positions (N, 3)
    int num_particles
);

} // namespace NBodyCUDA