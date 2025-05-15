#pragma once
#include <unordered_map>

namespace SpatialHashCPU {

struct Point3D {
    double x, y, z;
};

struct ParticleInfo {
    Point3D pos;
    int current_idx; // Index within the 'active_particles' arrays passed to the functions
};

struct GridParamsCPU {
    Point3D min_coord;
    Point3D max_coord;
    double cell_size_inv; // 1.0 / cell_size for faster calculations
    int grid_dim_x;
    int grid_dim_y;
    int grid_dim_z;
};

struct CellHash {
    std::size_t operator()(const std::tuple<int, int, int>& t) const {
        int ix = std::get<0>(t);
        int iy = std::get<1>(t);
        int iz = std::get<2>(t);
        std::size_t seed = 0;
        seed ^= std::hash<int>{}(ix) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(iy) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(iz) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

// Type for the grid: maps a cell coordinate tuple to a list of particle indices (within the active set)
using SpatialGrid = std::unordered_map<std::tuple<int, int, int>, std::vector<int>, CellHash>;

// Function to calculate grid parameters based on particle positions and desired cell size
GridParamsCPU calculate_grid_params_cpu(
    const ParticleInfo* active_particles,
    int num_active_particles,
    double target_cell_size);

// Function to build the spatial grid
SpatialGrid build_spatial_grid_cpu(
    const ParticleInfo* active_particles,
    int num_active_particles,
    const GridParamsCPU& grid_params);

// Main function for collision detection using spatial hashing
std::vector<std::tuple<int, int>> find_colliding_pairs_spatial_hash_cpu(
    const ParticleInfo* active_particles, // Array of active particle info
    const double* particle_radii,
    int num_active_particles,
    double target_cell_size);             // Desired cell size L

// Main function for finding minimum distance for each particle
void get_min_dist_array_spatial_hash_cpu(
    double* out_min_dists_per_particle,   // Output array (size num_active_particles)
    const ParticleInfo* active_particles, // Array of active particle info
    int num_active_particles,
    double target_cell_size);

} // namespace SpatialHashCPU

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