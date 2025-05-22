#include "../nbody_kernels.h"
#include <iostream>

namespace SpatialHashCPU {

// --- Helper to get cell coordinates ---
inline std::tuple<int, int, int> get_cell_coords(const Point3D& pos, const GridParamsCPU& params) {
    int ix = static_cast<int>(std::floor((pos.x - params.min_coord.x) * params.cell_size_inv));
    int iy = static_cast<int>(std::floor((pos.y - params.min_coord.y) * params.cell_size_inv));
    int iz = static_cast<int>(std::floor((pos.z - params.min_coord.z) * params.cell_size_inv));

    // Clamp to grid dimensions to handle particles on the max boundary
    ix = std::max(0, std::min(ix, params.grid_dim_x - 1));
    iy = std::max(0, std::min(iy, params.grid_dim_y - 1));
    iz = std::max(0, std::min(iz, params.grid_dim_z - 1));
    return {ix, iy, iz};
}


GridParamsCPU calculate_grid_params_cpu(
    const ParticleInfo* active_particles,
    int num_active_particles,
    double target_cell_size) {
    
    GridParamsCPU params;
    if (num_active_particles == 0) {
        params.min_coord = {0,0,0};
        params.max_coord = {0,0,0};
        params.cell_size_inv = (target_cell_size > 0) ? 1.0 / target_cell_size : 1.0;
        params.grid_dim_x = params.grid_dim_y = params.grid_dim_z = 1;
        return params;
    }

    params.min_coord = active_particles[0].pos;
    params.max_coord = active_particles[0].pos;

    for (int i = 1; i < num_active_particles; ++i) {
        params.min_coord.x = std::min(params.min_coord.x, active_particles[i].pos.x);
        params.min_coord.y = std::min(params.min_coord.y, active_particles[i].pos.y);
        params.min_coord.z = std::min(params.min_coord.z, active_particles[i].pos.z);
        params.max_coord.x = std::max(params.max_coord.x, active_particles[i].pos.x);
        params.max_coord.y = std::max(params.max_coord.y, active_particles[i].pos.y);
        params.max_coord.z = std::max(params.max_coord.z, active_particles[i].pos.z);
    }
    
    // Add a small padding to max_coord to ensure particles on the boundary are included
    double padding = target_cell_size * 0.01; // Small padding
    params.max_coord.x += padding;
    params.max_coord.y += padding;
    params.max_coord.z += padding;
    params.min_coord.x -= padding;
    params.min_coord.y -= padding;
    params.min_coord.z -= padding;

    // if (target_cell_size <= 0) { // Should not happen, but as a fallback
    //     target_cell_size = 1.0; 
    //     std::cerr << "Warning: target_cell_size was <= 0, defaulted to 1.0" << std::endl;
    // }
    params.cell_size_inv = 1.0 / target_cell_size;

    params.grid_dim_x = std::max(1, static_cast<int>(std::ceil((params.max_coord.x - params.min_coord.x) * params.cell_size_inv)));
    params.grid_dim_y = std::max(1, static_cast<int>(std::ceil((params.max_coord.y - params.min_coord.y) * params.cell_size_inv)));
    params.grid_dim_z = std::max(1, static_cast<int>(std::ceil((params.max_coord.z - params.min_coord.z) * params.cell_size_inv)));
    
    return params;
}


SpatialGrid build_spatial_grid_cpu(
    const ParticleInfo* active_particles,
    int num_active_particles,
    const GridParamsCPU& grid_params) {
    
    SpatialGrid grid;
    for (int i = 0; i < num_active_particles; ++i) {
        std::tuple<int, int, int> cell_idx_tuple = get_cell_coords(active_particles[i].pos, grid_params);
        grid[cell_idx_tuple].push_back(i); // Store index `i` from the active_particles array
    }
    return grid;
}


// Common function to process neighbors for both collisions and min_dist
void process_particle_neighbors_colliding_pairs(
    int p_active_idx, // Index of the current particle P in active_particles array
    const ParticleInfo* active_particles,
    const double* particle_radii,
    int num_active_particles,
    const GridParamsCPU& grid_params,
    const SpatialGrid& grid,
    std::vector<std::tuple<int, int>>& colliding_pairs_out) {                                  

    const auto& particle_p = active_particles[p_active_idx];
    std::tuple<int, int, int> p_cell_coords = get_cell_coords(particle_p.pos, grid_params);

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                std::tuple<int, int, int> neighbor_cell_coords = {
                    std::get<0>(p_cell_coords) + dx,
                    std::get<1>(p_cell_coords) + dy,
                    std::get<2>(p_cell_coords) + dz
                };

                // Check if neighbor cell is within grid bounds (optional, get_cell_coords clamps)
                if (std::get<0>(neighbor_cell_coords) < 0 || std::get<0>(neighbor_cell_coords) >= grid_params.grid_dim_x ||
                    std::get<1>(neighbor_cell_coords) < 0 || std::get<1>(neighbor_cell_coords) >= grid_params.grid_dim_y ||
                    std::get<2>(neighbor_cell_coords) < 0 || std::get<2>(neighbor_cell_coords) >= grid_params.grid_dim_z) {
                    continue;
                }

                auto it = grid.find(neighbor_cell_coords);
                if (it != grid.end()) { // If neighbor cell is not empty
                    const std::vector<int>& particles_in_cell = it->second;
                    for (int q_active_idx : particles_in_cell) {
                        if (p_active_idx == q_active_idx) continue; // Don't compare with self

                        const auto& particle_q = active_particles[q_active_idx];
                        
                        double diff_x = particle_p.pos.x - particle_q.pos.x;
                        double diff_y = particle_p.pos.y - particle_q.pos.y;
                        double diff_z = particle_p.pos.z - particle_q.pos.z;
                        double dist_sq = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;

                        // To avoid duplicate pairs (p,q) and (q,p), only add if p_active_idx < q_active_idx
                        if (p_active_idx < q_active_idx) { 
                            double sum_radii = particle_radii[p_active_idx] + particle_radii[q_active_idx];
                            if (dist_sq < sum_radii * sum_radii) {
                                colliding_pairs_out.emplace_back(particle_p.current_idx, particle_q.current_idx);
                            }
                        }
                        
                    }
                }
            }
        }
    }
}


// Common function to process neighbors for both collisions and min_dist
void process_particle_neighbors_min_dist(
    int p_active_idx, // Index of the current particle P in active_particles array
    const ParticleInfo* active_particles,
    int num_active_particles,
    const GridParamsCPU& grid_params,
    const SpatialGrid& grid,
    double& min_dist_sq_for_p_out) {                                 

    const auto& particle_p = active_particles[p_active_idx];
    std::tuple<int, int, int> p_cell_coords = get_cell_coords(particle_p.pos, grid_params);

    min_dist_sq_for_p_out = std::numeric_limits<double>::max();

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                std::tuple<int, int, int> neighbor_cell_coords = {
                    std::get<0>(p_cell_coords) + dx,
                    std::get<1>(p_cell_coords) + dy,
                    std::get<2>(p_cell_coords) + dz
                };

                // Check if neighbor cell is within grid bounds (optional, get_cell_coords clamps)
                if (std::get<0>(neighbor_cell_coords) < 0 || std::get<0>(neighbor_cell_coords) >= grid_params.grid_dim_x ||
                    std::get<1>(neighbor_cell_coords) < 0 || std::get<1>(neighbor_cell_coords) >= grid_params.grid_dim_y ||
                    std::get<2>(neighbor_cell_coords) < 0 || std::get<2>(neighbor_cell_coords) >= grid_params.grid_dim_z) {
                    continue;
                }

                auto it = grid.find(neighbor_cell_coords);
                if (it != grid.end()) { // If neighbor cell is not empty
                    const std::vector<int>& particles_in_cell = it->second;
                    for (int q_active_idx : particles_in_cell) {
                        if (p_active_idx == q_active_idx) continue; // Don't compare with self

                        const auto& particle_q = active_particles[q_active_idx];
                        
                        double diff_x = particle_p.pos.x - particle_q.pos.x;
                        double diff_y = particle_p.pos.y - particle_q.pos.y;
                        double diff_z = particle_p.pos.z - particle_q.pos.z;
                        double dist_sq = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;

                        if (dist_sq < min_dist_sq_for_p_out) {
                            min_dist_sq_for_p_out = dist_sq;
                        }

                    }
                }
            }
        }
    }
}


std::vector<std::tuple<int, int>> find_colliding_pairs_spatial_hash_cpu(
    const ParticleInfo* active_particles,
    const double* particle_radii,
    int num_active_particles,
    double target_cell_size) {
    
    std::vector<std::tuple<int, int>> all_colliding_pairs = {};
    if (num_active_particles < 2) return all_colliding_pairs;

    GridParamsCPU grid_params = calculate_grid_params_cpu(active_particles, num_active_particles, target_cell_size);
    SpatialGrid grid = build_spatial_grid_cpu(active_particles, num_active_particles, grid_params);

    for (int i = 0; i < num_active_particles; ++i) {
        process_particle_neighbors_colliding_pairs(
            i, active_particles, particle_radii, num_active_particles, grid_params, grid, all_colliding_pairs
        );
    }
    return all_colliding_pairs;
}


void get_min_dist_array_spatial_hash_cpu(
    double* out_min_dists_per_particle,
    const ParticleInfo* active_particles,
    int num_active_particles,
    double target_cell_size) {

    if (num_active_particles == 0) return;

    GridParamsCPU grid_params = calculate_grid_params_cpu(active_particles, num_active_particles, target_cell_size);
    SpatialGrid grid = build_spatial_grid_cpu(active_particles, num_active_particles, grid_params);

    for (int i = 0; i < num_active_particles; ++i) {
        double min_dist_sq_for_p;
        process_particle_neighbors_min_dist(
            i, active_particles, num_active_particles, grid_params, grid, min_dist_sq_for_p
        );
        
        if (min_dist_sq_for_p == std::numeric_limits<double>::max()) { // Particle was isolated
            out_min_dists_per_particle[i] = std::numeric_limits<double>::infinity();
        } else {
            out_min_dists_per_particle[i] = std::sqrt(min_dist_sq_for_p);
        }
    }
}

}