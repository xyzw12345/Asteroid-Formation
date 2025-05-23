#include "../nbody_kernels.h"
#include <omp.h>

namespace N2CPU {
void compute_accelerations_n2_cpu(
    double* out_accel_ptr, const double* active_pos_ptr, const double* active_mass_ptr,     
    int num_active_particles, double G, double epsilon) {
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_active_particles; ++i) {

        out_accel_ptr[i * 3 + 0] = 0;
        out_accel_ptr[i * 3 + 1] = 0;
        out_accel_ptr[i * 3 + 2] = 0;
        
        for (int j = 0; j < num_active_particles; ++j) {
            if (j == i) continue;
            double dx = active_pos_ptr[j * 3 + 0] - active_pos_ptr[i * 3 + 0];
            double dy = active_pos_ptr[j * 3 + 1] - active_pos_ptr[i * 3 + 1];
            double dz = active_pos_ptr[j * 3 + 2] - active_pos_ptr[i * 3 + 2];
            double dist_sq = std::max(epsilon * epsilon, dx * dx + dy * dy + dz * dz);
            double dist = std::sqrt(dist_sq);
            double force_mag = G / (dist_sq * dist) * active_mass_ptr[j];

            out_accel_ptr[i * 3 + 0] += force_mag * dx;
            out_accel_ptr[i * 3 + 1] += force_mag * dy;
            out_accel_ptr[i * 3 + 2] += force_mag * dz;
        }
    }    

}

std::vector<std::tuple<int, int>> find_colliding_pairs_n2_cpu(
    const double* active_pos_ptr, const double* active_vel_ptr,
    const double* active_radii_ptr, int num_active_particles, double dt) {
    std::vector<std::tuple<int, int>> final_result;
    #pragma omp parallel
    {
        std::vector<std::tuple<int, int>> local_thread_result;
        // The schedule(dynamic) can be helpful here if work per 'i' varies
        #pragma omp for schedule(dynamic) nowait // nowait can be used if merging is also parallelized
        for (int i = 0; i < num_active_particles; ++i) {
            for (int j = i + 1; j < num_active_particles; ++j) {
                double dx = active_pos_ptr[j * 3 + 0] - active_pos_ptr[i * 3 + 0];
                double dy = active_pos_ptr[j * 3 + 1] - active_pos_ptr[i * 3 + 1];
                double dz = active_pos_ptr[j * 3 + 2] - active_pos_ptr[i * 3 + 2];
                double vx = active_vel_ptr[j * 3 + 0] - active_vel_ptr[i * 3 + 0];
                double vy = active_vel_ptr[j * 3 + 1] - active_vel_ptr[i * 3 + 1];
                double vz = active_vel_ptr[j * 3 + 2] - active_vel_ptr[i * 3 + 2];
                
                double v_sq = vx * vx + vy * vy + vz * vz;
                double v_dot_x = vx * dx + vy * dy + vz * dz;
                double t0 = - (v_dot_x / v_sq) * dt;
                double min_dist_sq = dx * dx + dy * dy + dz * dz;
                min_dist_sq = std::min(min_dist_sq, (dx + dt * vx) * (dx + dt * vx) + (dy + dt * vy) * (dy + dt * vy) + (dz + dt * vz) * (dz + dt * vz));
                if (v_sq > 1e-10 && (0 <= t0) && (t0 <= dt) ) {
                    min_dist_sq = std::min(min_dist_sq, (dx + t0 * vx) * (dx + t0 * vx) + (dy + t0 * vy) * (dy + t0 * vy) + (dz + t0 * vz) * (dz + t0 * vz));
                }
                double sum_radii = active_radii_ptr[i] + active_radii_ptr[j];
                double sum_radii_sq = sum_radii * sum_radii;

                if (min_dist_sq < sum_radii_sq) {
                    local_thread_result.emplace_back(i, j);
                }
            }
        }
        
        // Merge local_thread_result into final_result (needs synchronization)
        // This critical section protects the shared final_result vector.
        #pragma omp critical
        {
            final_result.insert(final_result.end(), 
                                local_thread_result.begin(), 
                                local_thread_result.end());
        }
    } // End of parallel region
    return final_result;
}

void get_min_dist_array_n2_cpu(
    double* out_min_dists_per_particle, const double* active_pos_ptr, int num_active_particles){
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_active_particles; ++i) {
        out_min_dists_per_particle[i] = std::numeric_limits<double>::max();
        for (int j = 0; j < num_active_particles; ++j) {
            if (j == i) continue;
            double dx = active_pos_ptr[j * 3 + 0] - active_pos_ptr[i * 3 + 0];
            double dy = active_pos_ptr[j * 3 + 1] - active_pos_ptr[i * 3 + 1];
            double dz = active_pos_ptr[j * 3 + 2] - active_pos_ptr[i * 3 + 2];
            double dist_sq = dx * dx + dy * dy + dz * dz;
            out_min_dists_per_particle[i] = std::min(out_min_dists_per_particle[i], std::sqrt(dist_sq));
        }
    } 
}
} // namespace N2CPU