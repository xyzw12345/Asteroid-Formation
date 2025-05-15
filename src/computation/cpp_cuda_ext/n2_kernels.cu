#include "nbody_kernels.h"
#include <cuda_runtime.h>
#include <cstdio> // For printf in kernels (debugging)

// Helper for error checking
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

namespace NBodyCUDA {

__global__ void gravity_kernel(
    double* accel_out, // (N,3)
    const double* pos,   // (N,3)
    const double* mass,  // (N)
    int num_particles,
    double G,
    double epsilon_sq) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_particles) {
        double acc_x = 0.0;
        double acc_y = 0.0;
        double acc_z = 0.0;

        double pos_ix = pos[i * 3 + 0];
        double pos_iy = pos[i * 3 + 1];
        double pos_iz = pos[i * 3 + 2];

        for (int j = 0; j < num_particles; ++j) {
            if (i == j) continue;

            double dx = pos[j * 3 + 0] - pos_ix;
            double dy = pos[j * 3 + 1] - pos_iy;
            double dz = pos[j * 3 + 2] - pos_iz;

            double dist_sq = dx * dx + dy * dy + dz * dz;
            double inv_dist_sq = 1.0 / max(dist_sq, epsilon_sq); // Softened distance
            double inv_dist = sqrt(inv_dist_sq);
            double inv_dist_cubed = inv_dist * inv_dist_sq;

            double force_mag_over_m_i = G * mass[j] * inv_dist_cubed;

            acc_x += dx * force_mag_over_m_i;
            acc_y += dy * force_mag_over_m_i;
            acc_z += dz * force_mag_over_m_i;
        }
        accel_out[i * 3 + 0] = acc_x;
        accel_out[i * 3 + 1] = acc_y;
        accel_out[i * 3 + 2] = acc_z;
    }
}

void compute_accelerations_cuda_n2(
    double* d_accel_out, // Output: device pointer for accelerations (N, 3)
    const double* d_pos,   // Input: device pointer for positions (N, 3)
    const double* d_mass,  // Input: device pointer for masses (N)
    int num_particles,
    double G,
    double epsilon_sq) {

    // Kernel launch configuration
    int threads_per_block = 256; // Common choice, tune based on GPU
    int blocks_per_grid = (num_particles + threads_per_block - 1) / threads_per_block;

    gravity_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_accel_out, d_pos, d_mass, num_particles, G, epsilon_sq
    );
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors
    // cudaDeviceSynchronize is not strictly needed here if subsequent operations
    // involve implicit synchronization (like cudaMemcpy from device to host),
    // but good for debugging kernel issues.
    // CUDA_CHECK(cudaDeviceSynchronize()); 
}


// --- Collision Detection Kernel ---
// This kernel calculates the (N,N) dist_sq matrix on the fly for a subset of pairs
// or can be adapted to write to a collision flag matrix if memory allows.
// For finding actual pairs, a more complex approach with atomics or parallel reduction is needed.
// A simpler GPU approach for N^2 check_overlap is to compute the (N,N) boolean collision matrix,
// then transfer it (if small enough) or use thrust/CUB to find true indices.
// Given the memory constraints for N=50k, we cannot easily form an (N,N) matrix.
//
// A more scalable GPU approach for collisions requires spatial subdivision.
// For now, let's implement the "get_min_pairwise_dist_sq_cuda" as it's more feasible
// with N^2 on GPU without huge intermediate matrices.
// find_colliding_pairs_cuda is very hard to make efficient for N^2 on GPU
// without returning a huge matrix or very complex reduction.
// We might defer to a CPU version for check_overlap if a GPU N^2 version is too slow/memory hungry.

__global__ void min_dist_sq_kernel_inter_block(
    double* d_min_dist_output,
    const double* pos, // (N,3)
    int num_particles) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_particles) {
        double pos_ix = pos[i * 3 + 0];
        double pos_iy = pos[i * 3 + 1];
        double pos_iz = pos[i * 3 + 2];
        
        double current_min_sq_for_i = 1.0e+38;

        for (int j = 0; j < num_particles; ++j) {
            if (i == j) continue;

            double dx = pos[j * 3 + 0] - pos_ix;
            double dy = pos[j * 3 + 1] - pos_iy;
            double dz = pos[j * 3 + 2] - pos_iz;
            double dist_sq = dx * dx + dy * dy + dz * dz;
            if (dist_sq < current_min_sq_for_i) {
                current_min_sq_for_i = dist_sq;
            }
        }
        d_min_dist_output[i] = current_min_sq_for_i;
    }
}


void get_min_dist_sq_cuda_n2(
    double* d_min_dist_output,
    const double* d_pos,
    int num_particles) {

    int threads_per_block = 256;
    int blocks_per_grid = (num_particles + threads_per_block - 1) / threads_per_block;

    min_dist_sq_kernel_inter_block<<<blocks_per_grid, threads_per_block>>>(
        d_min_dist_output, d_pos, num_particles
    );
    CUDA_CHECK(cudaGetLastError());
}

__global__ void find_colliding_pairs_kernel(
    GpuCollisionPair* d_colliding_pairs_buffer, // Output buffer for pairs
    int* d_collision_count,                     // Atomic counter for number of collisions found
    int max_pairs_capacity,                     // Max capacity of the output buffer
    const double* d_pos,                        // (N, 3)
    const double* d_radii,                      // (N)
    int num_particles) {

    // Grid-striding loop: assign more than one particle 'i' to each thread if needed
    // More commonly, each thread handles one 'i' and loops over 'j',
    // or a block of threads handles a tile of the (i,j) matrix.

    // For simplicity here, let each thread handle one 'i' and iterate j > i
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_particles) {
        double pos_ix = d_pos[i * 3 + 0];
        double pos_iy = d_pos[i * 3 + 1];
        double pos_iz = d_pos[i * 3 + 2];
        double radius_i = d_radii[i];

        // Iterate over j > i to check unique pairs
        for (int j = i + 1; j < num_particles; ++j) {
            double pos_jx = d_pos[j * 3 + 0];
            double pos_jy = d_pos[j * 3 + 1];
            double pos_jz = d_pos[j * 3 + 2];
            double radius_j = d_radii[j];

            double dx = pos_jx - pos_ix;
            double dy = pos_jy - pos_iy;
            double dz = pos_jz - pos_iz;

            double dist_sq = dx * dx + dy * dy + dz * dz;
            double sum_radii = radius_i + radius_j;
            double sum_radii_sq = sum_radii * sum_radii;

            if (dist_sq < sum_radii_sq) {
                // Collision detected!
                // Get a unique index atomically to write the pair
                int current_idx = atomicAdd(d_collision_count, 1);
                
                // Check if we have space in the buffer
                if (current_idx < max_pairs_capacity) {
                    d_colliding_pairs_buffer[current_idx].idx1 = i; // Store indices relative to the input arrays
                    d_colliding_pairs_buffer[current_idx].idx2 = j;
                }
                // If current_idx >= max_pairs_capacity, the collision is detected
                // but not stored. The final *d_collision_count might be > max_pairs_capacity.
            }
        }
    }
}


int find_colliding_pairs_cuda_n2(
    GpuCollisionPair* d_colliding_pairs_buffer_host_provided, // Buffer for pairs (pre-allocated by host)
    int max_pairs_capacity,                                  // Capacity of the buffer
    const double* d_pos,                                     // Device pointer for positions (N, 3)
    const double* d_radii,                                   // Device pointer for radii (N)
    int num_particles) {

    if (num_particles < 2 || max_pairs_capacity == 0) {
        return 0; // No pairs to check or no space to store them
    }

    // We need a device-side counter for atomic operations
    int* d_collision_count;
    CUDA_CHECK(cudaMalloc(&d_collision_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_collision_count, 0, sizeof(int))); // Initialize count to 0

    // Kernel launch configuration
    int threads_per_block = 256;
    // Each thread will handle one 'i' and loop 'j' from 'i+1'
    int blocks_per_grid = (num_particles + threads_per_block - 1) / threads_per_block;

    find_colliding_pairs_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_colliding_pairs_buffer_host_provided,
        d_collision_count,
        max_pairs_capacity,
        d_pos,
        d_radii,
        num_particles
    );
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors
    // It's good to synchronize here to ensure the kernel completes and d_collision_count is updated
    // before we copy it back and free memory.
    CUDA_CHECK(cudaDeviceSynchronize()); 

    // Copy the final collision count back to host
    int h_collision_count;
    CUDA_CHECK(cudaMemcpy(&h_collision_count, d_collision_count, sizeof(int), cudaMemcpyDeviceToHost));

    // Free the device-side counter
    CUDA_CHECK(cudaFree(d_collision_count));

    // The number of pairs actually written to the buffer is min(h_collision_count, max_pairs_capacity)
    // The function returns the total number of collisions detected, which might be > max_pairs_capacity
    return h_collision_count; 
}

} // namespace NBodyCUDA
