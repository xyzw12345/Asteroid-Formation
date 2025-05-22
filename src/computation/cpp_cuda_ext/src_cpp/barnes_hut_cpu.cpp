#include "../nbody_kernels.h"
#include <iostream>     // For debug
#include <stdexcept>    // For std::runtime_error
#include <omp.h>

namespace BarnesHutCPU {

// --- BarnesHutTree Member Function Implementations ---

BarnesHutTree::BarnesHutTree(
    const double* active_pos_xyz_flat_array,
    const double* active_masses,
    int num_active_particles)
    : root(nullptr), 
      all_particle_pos_ptr(active_pos_xyz_flat_array),
      all_particle_mass_ptr(active_masses),
      num_total_active_particles(num_active_particles) {

    if (num_active_particles == 0) {
        return;
    }

    // 1. Determine bounding box for the root node
    Point3D min_b = {active_pos_xyz_flat_array[0], active_pos_xyz_flat_array[1], active_pos_xyz_flat_array[2]};
    Point3D max_b = min_b;

    for (int i = 1; i < num_active_particles; ++i) {
        min_b.x = std::min(min_b.x, active_pos_xyz_flat_array[i * 3 + 0]);
        min_b.y = std::min(min_b.y, active_pos_xyz_flat_array[i * 3 + 1]);
        min_b.z = std::min(min_b.z, active_pos_xyz_flat_array[i * 3 + 2]);
        max_b.x = std::max(max_b.x, active_pos_xyz_flat_array[i * 3 + 0]);
        max_b.y = std::max(max_b.y, active_pos_xyz_flat_array[i * 3 + 1]);
        max_b.z = std::max(max_b.z, active_pos_xyz_flat_array[i * 3 + 2]);
    }

    Point3D root_center = (min_b + max_b) * 0.5;
    double root_half_width = 0.5 * std::max(max_b.x - min_b.x, std::max(max_b.y - min_b.y, max_b.z - min_b.z));
    // Ensure half_width is not zero if all particles are at the same point
    if (root_half_width < 1e-9) { // A small threshold
        root_half_width = 1.0; // Or some default sensible size
    }
    // Make root node slightly larger to contain all points comfortably
    root_half_width *= 1.01;


    root = new OctreeNode(root_center, root_half_width);

    // 2. Insert all particles into the tree
    for (int i = 0; i < num_active_particles; ++i) {
        insert(root, i);
    }

    // 3. Compute mass distributions for internal nodes (bottom-up)
    compute_node_mass_distribution(root);
}

void BarnesHutTree::insert(OctreeNode* node, int p_idx) {
    if (node == nullptr) { // Should not happen if root is initialized
        // This could indicate an issue with child creation or node bounds.
        // For robustness, you might throw or log an error.
        std::cerr << "Error: Trying to insert into a null node." << std::endl;
        return;
    }
    
    Point3D p_pos = {
        all_particle_pos_ptr[p_idx * 3 + 0],
        all_particle_pos_ptr[p_idx * 3 + 1],
        all_particle_pos_ptr[p_idx * 3 + 2]
    };

    // Check if particle is outside node bounds (should ideally not happen if root is sized correctly)
    // This check is important if particles can move significantly.
    // For simplicity in initial build, we assume root encompasses all.
    // More robust: if outside, expand root or handle error.


    if (node->num_particles_in_subtree == 0 && node->is_leaf) { // Node is empty leaf
        node->particle_idx = p_idx;
        node->num_particles_in_subtree = 1;
        // CoM and total_mass will be set by compute_node_mass_distribution later,
        // or you can set them here for leaf nodes.
        // node->center_of_mass = p_pos;
        // node->total_mass = all_particle_mass_ptr[p_idx];
        return;
    }

    if (node->is_leaf) { // Node is a leaf but already contains a particle
        // Convert this leaf node into an internal node
        node->is_leaf = false;
        
        // Re-insert the particle that was already in this node
        int existing_p_idx = node->particle_idx;
        node->particle_idx = -1; // Mark as internal

        // Determine octant for the existing particle and create child
        int octant_existing = node->get_octant({all_particle_pos_ptr[existing_p_idx*3],
                                                all_particle_pos_ptr[existing_p_idx*3+1],
                                                all_particle_pos_ptr[existing_p_idx*3+2]});
        Point3D child_center_existing = node->center;
        double child_half_width = node->half_width * 0.5;
        child_center_existing.x += (octant_existing & 1 ? child_half_width : -child_half_width);
        child_center_existing.y += (octant_existing & 2 ? child_half_width : -child_half_width);
        child_center_existing.z += (octant_existing & 4 ? child_half_width : -child_half_width);
        
        if (node->children[octant_existing] == nullptr) {
             node->children[octant_existing] = new OctreeNode(child_center_existing, child_half_width);
        }
        insert(node->children[octant_existing], existing_p_idx);


        // Now insert the new particle into the appropriate child of this (now internal) node
        int octant_new = node->get_octant(p_pos);
        Point3D child_center_new = node->center;
        child_center_new.x += (octant_new & 1 ? child_half_width : -child_half_width);
        child_center_new.y += (octant_new & 2 ? child_half_width : -child_half_width);
        child_center_new.z += (octant_new & 4 ? child_half_width : -child_half_width);

        if (node->children[octant_new] == nullptr) {
            node->children[octant_new] = new OctreeNode(child_center_new, child_half_width);
        }
        insert(node->children[octant_new], p_idx);
    } else { // Node is already an internal node
        int octant = node->get_octant(p_pos);
        Point3D child_center = node->center;
        double child_half_width = node->half_width * 0.5;
        child_center.x += (octant & 1 ? child_half_width : -child_half_width);
        child_center.y += (octant & 2 ? child_half_width : -child_half_width);
        child_center.z += (octant & 4 ? child_half_width : -child_half_width);

        if (node->children[octant] == nullptr) {
            node->children[octant] = new OctreeNode(child_center, child_half_width);
        }
        insert(node->children[octant], p_idx);
    }
    node->num_particles_in_subtree++; // Increment count for internal nodes during insertion
}

// Recursive function to compute CoM and TotalMass for internal nodes (post-order traversal)
void BarnesHutTree::compute_node_mass_distribution(OctreeNode* node) {
    if (node == nullptr) return;

    if (node->is_leaf) {
        if (node->particle_idx != -1) { // Leaf with a particle
            int p_idx = node->particle_idx;
            node->center_of_mass = {all_particle_pos_ptr[p_idx * 3 + 0],
                                    all_particle_pos_ptr[p_idx * 3 + 1],
                                    all_particle_pos_ptr[p_idx * 3 + 2]};
            node->total_mass = all_particle_mass_ptr[p_idx];
            node->num_particles_in_subtree = 1; // Should have been set during insert for leaves
        } else { // Empty leaf
            node->total_mass = 0.0;
            node->center_of_mass = node->center; // Or some default
            node->num_particles_in_subtree = 0;
        }
    } else { // Internal node
        node->total_mass = 0.0;
        node->center_of_mass = {0,0,0}; // Reset for accumulation
        node->num_particles_in_subtree = 0; // Reset and recount from children

        for (int i = 0; i < 8; ++i) {
            if (node->children[i] != nullptr) {
                compute_node_mass_distribution(node->children[i]); // Recurse first
                
                node->total_mass += node->children[i]->total_mass;
                node->center_of_mass = node->center_of_mass + 
                                       (node->children[i]->center_of_mass * node->children[i]->total_mass);
                node->num_particles_in_subtree += node->children[i]->num_particles_in_subtree;
            }
        }
        if (node->total_mass > 1e-9) { // Avoid division by zero if node became effectively empty
            node->center_of_mass = node->center_of_mass / node->total_mass;
        } else {
             node->center_of_mass = node->center; // Default to geometric center if massless
        }
    }
}


Point3D BarnesHutTree::calculate_force_on_particle(
    int target_p_idx, 
    OctreeNode* current_node, 
    double theta_sq, 
    double epsilon_sq,
    double G) const {
    
    Point3D force_acc = {0,0,0};
    if (current_node == nullptr || current_node->total_mass < 1e-12 /* effectively massless */ ) {
        return force_acc;
    }

    Point3D target_pos = {all_particle_pos_ptr[target_p_idx * 3 + 0],
                          all_particle_pos_ptr[target_p_idx * 3 + 1],
                          all_particle_pos_ptr[target_p_idx * 3 + 2]};

    if (current_node->is_leaf) {
        if (current_node->particle_idx != -1 && current_node->particle_idx != target_p_idx) {
            // Direct interaction with the single particle in this leaf node
            Point3D diff = current_node->center_of_mass - target_pos; // CoM of leaf is particle's pos
            double dist_sq = diff.length_sq();
            double dist_sq_soft = std::max(dist_sq, epsilon_sq);
            if (dist_sq_soft < 1e-18) dist_sq_soft = 1e-18; // Avoid division by zero from exact overlap after softening
            
            double inv_dist_sq = 1.0 / dist_sq_soft;
            double inv_dist_cubed = inv_dist_sq * std::sqrt(inv_dist_sq);
            double F_mag_over_m_target = G * current_node->total_mass * inv_dist_cubed;
            force_acc = force_acc + (diff * F_mag_over_m_target);
        }
    } else { // Internal node
        Point3D diff_node_com = current_node->center_of_mass - target_pos;
        double dist_sq_node_com = diff_node_com.length_sq();
        
        // s = node width = 2 * half_width
        // s^2 / d^2 < theta^2  => (2*half_width)^2 / dist_sq_node_com < theta_sq
        if ( ( (4.0 * current_node->half_width * current_node->half_width) / (dist_sq_node_com + 1e-12) ) < theta_sq ) {
            // Node is far enough, treat as a macro-particle
            double dist_sq_soft = std::max(dist_sq_node_com, epsilon_sq);
            if (dist_sq_soft < 1e-18) dist_sq_soft = 1e-18;

            double inv_dist_sq = 1.0 / dist_sq_soft;
            double inv_dist_cubed = inv_dist_sq * std::sqrt(inv_dist_sq);
            double F_mag_over_m_target = G * current_node->total_mass * inv_dist_cubed;
            force_acc = force_acc + (diff_node_com * F_mag_over_m_target);
        } else {
            // Node is too close, recurse on children
            for (int i = 0; i < 8; ++i) {
                if (current_node->children[i] != nullptr) {
                    force_acc = force_acc + calculate_force_on_particle(
                        target_p_idx, current_node->children[i], theta_sq, epsilon_sq, G
                    );
                }
            }
        }
    }
    return force_acc;
}


// --- Main Public Function (Implementation) ---
void compute_accelerations_barnes_hut_cpu(
    double* out_accel_ptr,
    const double* active_pos_ptr,
    const double* active_mass_ptr,
    int num_active_particles,
    double G,
    double epsilon,
    double theta) {
    // The Sun (assumed to be at particle 0) is processed separately.

    if (num_active_particles == 0) return;

    // 1. Build the tree
    BarnesHutTree tree(active_pos_ptr + 3, active_mass_ptr + 1, num_active_particles - 1);
    if (tree.root == nullptr) { // Handle case where tree construction failed or no particles
        for(int i = 0; i < num_active_particles * 3; ++i) out_accel_ptr[i] = 0.0;
        return;
    }

    double theta_sq = theta * theta;
    double epsilon_sq = epsilon * epsilon;

    out_accel_ptr[0] = 0;
    out_accel_ptr[1] = 0;
    out_accel_ptr[2] = 0;

    // 2. For each particle, calculate force using the tree
    #pragma omp parallel for schedule(dynamic)
    for (int i = 1; i < num_active_particles; ++i) {
        Point3D total_force = tree.calculate_force_on_particle(i - 1, tree.root, theta_sq, epsilon_sq, G);
    
        out_accel_ptr[i * 3 + 0] = total_force.x;
        out_accel_ptr[i * 3 + 1] = total_force.y;
        out_accel_ptr[i * 3 + 2] = total_force.z;
        
        double dx = active_pos_ptr[0] - active_pos_ptr[i * 3 + 0];
        double dy = active_pos_ptr[1] - active_pos_ptr[i * 3 + 1];
        double dz = active_pos_ptr[2] - active_pos_ptr[i * 3 + 2];
        double dist_sq = std::max(epsilon * epsilon, dx * dx + dy * dy + dz * dz);
        double dist = std::sqrt(dist_sq);
        double force_mag = G / (dist_sq * dist);

        out_accel_ptr[i * 3 + 0] += force_mag * active_mass_ptr[0] * dx;
        out_accel_ptr[i * 3 + 1] += force_mag * active_mass_ptr[0] * dy;
        out_accel_ptr[i * 3 + 2] += force_mag * active_mass_ptr[0] * dz;

        out_accel_ptr[0] -= force_mag * active_mass_ptr[i] * dx;
        out_accel_ptr[1] -= force_mag * active_mass_ptr[i] * dy;
        out_accel_ptr[2] -= force_mag * active_mass_ptr[i] * dz;
    }
    // Tree (and all its nodes) is automatically cleaned up when `tree` goes out of scope
    // due to BarnesHutTree destructor calling delete on root, which recursively deletes children.
}


} // namespace BarnesHutCPU