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

namespace BarnesHutCPU {

// --- Basic Structures ---
struct Point3D {
    double x, y, z;

    Point3D(double x_ = 0.0, double y_ = 0.0, double z_ = 0.0) : x(x_), y(y_), z(z_) {}

    Point3D operator+(const Point3D& other) const {
        return {x + other.x, y + other.y, z + other.z};
    }
    Point3D operator-(const Point3D& other) const {
        return {x - other.x, y - other.y, z - other.z};
    }
    Point3D operator*(double scalar) const {
        return {x * scalar, y * scalar, z * scalar};
    }
    Point3D operator/(double scalar) const {
        // Add check for scalar != 0 if necessary
        return {x / scalar, y / scalar, z / scalar};
    }
    double length_sq() const {
        return x * x + y * y + z * z;
    }
    double length() const {
        return std::sqrt(length_sq());
    }
};

// Information about particles being processed.
// For Barnes-Hut, we primarily need position and mass.
// We'll pass pointers to arrays of these for active particles.

struct OctreeNode {
    // Geometric properties
    Point3D center;      // Geometric center of the node's cube
    double half_width;   // Half the width of the node's cube (size/2)
                         // The node covers center.x +/- half_width, etc.

    // Particle aggregate properties
    Point3D center_of_mass;
    double total_mass;
    int num_particles_in_subtree; // Count of particles in this node and its children

    // Tree structure
    OctreeNode* children[8]; // Pointers to 8 child nodes (octants)
    bool is_leaf;

    // If it's a leaf and not empty, it might store the index of the single particle it contains
    // or a list of indices if max_particles_per_leaf > 1.
    // For simplicity with max_particles_per_leaf = 1:
    int particle_idx; // Index into the original active particle arrays

    OctreeNode(Point3D c, double hw) : 
        center(c), half_width(hw), 
        center_of_mass(), total_mass(0.0), num_particles_in_subtree(0),
        is_leaf(true), particle_idx(-1) // -1 indicates empty or internal
    {
        for (int i = 0; i < 8; ++i) {
            children[i] = nullptr;
        }
    }

    // Destructor to recursively delete children
    ~OctreeNode() {
        for (int i = 0; i < 8; ++i) {
            delete children[i]; // This will call the destructor of child nodes
            children[i] = nullptr; // Good practice
        }
    }

    // Method to determine which octant a point falls into relative to this node's center
    int get_octant(const Point3D& p) const {
        int octant = 0;
        if (p.x >= center.x) octant |= 1; // Right half
        if (p.y >= center.y) octant |= 2; // Top half
        if (p.z >= center.z) octant |= 4; // Front half
        return octant;
    }

    // (Future FMM Extension): FMM would add fields here for:
    // std::vector<std::complex<double>> multipole_coeffs; // Or other coefficient type
    // std::vector<std::complex<double>> local_coeffs;
};


class BarnesHutTree {
public:
    OctreeNode* root;
    const double* all_particle_pos_ptr;  // Pointer to X, Y, Z of all active particles
    const double* all_particle_mass_ptr; // Pointer to masses of all active particles
    int num_total_active_particles;

    BarnesHutTree(
        const double* active_pos_xyz_flat_array, // Flat array [x0,y0,z0, x1,y1,z1, ...]
        const double* active_masses,
        int num_active_particles);

    ~BarnesHutTree() {
        delete root; // This will trigger recursive deletion
    }

    void insert(OctreeNode* node, int p_idx);
    void compute_node_mass_distribution(OctreeNode* node);

    Point3D calculate_force_on_particle(
        int target_p_idx, 
        OctreeNode* current_node, 
        double theta_sq, 
        double epsilon_sq,
        double G) const;
};


// Main public function to be called from Pybind11 wrapper
void compute_accelerations_barnes_hut_cpu(
    double* out_accel_ptr,             // Output: Flat array [ax0,ay0,az0, ...]
    const double* active_pos_ptr,      // Input: Flat array [x0,y0,z0, ...]
    const double* active_mass_ptr,     // Input: Array of masses [m0, m1, ...]
    int num_active_particles,
    double G,
    double epsilon,                    // Softening length
    double theta                       // Barnes-Hut opening angle
);

} // namespace BarnesHutCPU