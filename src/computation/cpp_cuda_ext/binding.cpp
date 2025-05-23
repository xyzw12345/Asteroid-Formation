#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> // For returning std::vector<std::tuple<int,int>> if needed
#include "nbody_kernels.h" // Your C++ wrappers for CUDA calls
#ifdef USE_CUDA
#include <cuda_runtime.h> // For cudaMalloc, cudaMemcpy, etc.
#endif

namespace py = pybind11;

#ifdef USE_CUDA
// Helper to check CUDA calls from binding code
#define CHECK_CUDA_PYBIND(err) \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA Error in Pybind: ") + cudaGetErrorString(err)); \
    }


// Wrapper for compute_accelerations_cuda
py::array_t<double> py_compute_accelerations_cuda_n2(
    py::array_t<double, py::array::c_style | py::array::forcecast> positions_np,
    py::array_t<double, py::array::c_style | py::array::forcecast> masses_np,
    double G,
    double epsilon) {

    py::buffer_info pos_buf = positions_np.request();
    py::buffer_info mass_buf = masses_np.request();
    int num_particles = static_cast<int>(pos_buf.shape[0]);

    // Allocate GPU memory
    double *d_pos, *d_mass, *d_accel;
    CHECK_CUDA_PYBIND(cudaMalloc(&d_pos, num_particles * 3 * sizeof(double)));
    CHECK_CUDA_PYBIND(cudaMalloc(&d_mass, num_particles * sizeof(double)));
    CHECK_CUDA_PYBIND(cudaMalloc(&d_accel, num_particles * 3 * sizeof(double)));

    // Copy data from NumPy arrays (host) to GPU (device)
    CHECK_CUDA_PYBIND(cudaMemcpy(d_pos, pos_buf.ptr, num_particles * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_PYBIND(cudaMemcpy(d_mass, mass_buf.ptr, num_particles * sizeof(double), cudaMemcpyHostToDevice));

    // Call the CUDA wrapper
    NBodyCUDA::compute_accelerations_cuda_n2(d_accel, d_pos, d_mass, num_particles, G, epsilon * epsilon);

    // Prepare result NumPy array (allocate on host)
    py::array_t<double> accelerations_np_out({pos_buf.shape[0], static_cast<py::ssize_t>(3)});
    py::buffer_info acc_out_buf = accelerations_np_out.request();

    // Copy results from GPU to host NumPy array
    CHECK_CUDA_PYBIND(cudaMemcpy(acc_out_buf.ptr, d_accel, num_particles * 3 * sizeof(double), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CHECK_CUDA_PYBIND(cudaFree(d_pos));
    CHECK_CUDA_PYBIND(cudaFree(d_mass));
    CHECK_CUDA_PYBIND(cudaFree(d_accel));

    return accelerations_np_out;
}

// Wrapper for get_min_pairwise_dist_sq_cuda
py::array_t<double> py_get_min_dist_sq_cuda_n2(
    py::array_t<double, py::array::c_style | py::array::forcecast> positions_np) {
    
    py::buffer_info pos_buf = positions_np.request();
    int num_particles = static_cast<int>(pos_buf.shape[0]);

    double *d_pos, *d_min_dist_out;
    CHECK_CUDA_PYBIND(cudaMalloc(&d_pos, num_particles * 3 * sizeof(double)));
    CHECK_CUDA_PYBIND(cudaMalloc(&d_min_dist_out, num_particles * sizeof(double)));
    CHECK_CUDA_PYBIND(cudaMemcpy(d_pos, pos_buf.ptr, num_particles * 3 * sizeof(double), cudaMemcpyHostToDevice));

    NBodyCUDA::get_min_dist_sq_cuda_n2(d_min_dist_out, d_pos, num_particles);

    py::array_t<double> min_dist_out({pos_buf.shape[0]});
    py::buffer_info min_dist_out_buf = min_dist_out.request();

    CHECK_CUDA_PYBIND(cudaMemcpy(min_dist_out_buf.ptr, d_min_dist_out, num_particles * sizeof(double), cudaMemcpyDeviceToHost));

    CHECK_CUDA_PYBIND(cudaFree(d_pos));
    CHECK_CUDA_PYBIND(cudaFree(d_min_dist_out));
    
    return min_dist_out;
}

std::vector<std::tuple<int, int>> py_find_colliding_pairs_cuda_n2(
    py::array_t<double, py::array::c_style | py::array::forcecast> positions_np,
    py::array_t<double, py::array::c_style | py::array::forcecast> radii_np) {
    
    py::buffer_info pos_buf = positions_np.request();
    py::buffer_info radii_buf = radii_np.request();
    int num_particles = static_cast<int>(pos_buf.shape[0]);

    std::vector<std::tuple<int, int>> cpu_colliding_pairs;
    if (num_particles < 2) return cpu_colliding_pairs;

    double *d_pos, *d_radii;
    NBodyCUDA::GpuCollisionPair* d_colliding_pairs_buffer;
    
    // --- Determine max_pairs_capacity ---
    // Let's assume we can only store a limited number, e.g., related to N.
    int max_pairs_capacity = num_particles * 10; 
    if (num_particles > 1000 && num_particles < 10000) max_pairs_capacity = num_particles * 2;
    else if (num_particles >= 10000) max_pairs_capacity = num_particles; 

    CHECK_CUDA_PYBIND(cudaMalloc(&d_pos, num_particles * 3 * sizeof(double)));
    CHECK_CUDA_PYBIND(cudaMalloc(&d_radii, num_particles * sizeof(double)));
    CHECK_CUDA_PYBIND(cudaMalloc(&d_colliding_pairs_buffer, max_pairs_capacity * sizeof(NBodyCUDA::GpuCollisionPair)));
    
    CHECK_CUDA_PYBIND(cudaMemcpy(d_pos, pos_buf.ptr, num_particles * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_PYBIND(cudaMemcpy(d_radii, radii_buf.ptr, num_particles * sizeof(double), cudaMemcpyHostToDevice));

    int total_collisions_detected = NBodyCUDA::find_colliding_pairs_cuda_n2(
        d_colliding_pairs_buffer, 
        max_pairs_capacity, 
        d_pos, 
        d_radii, 
        num_particles
    );

    if (total_collisions_detected > 0 && max_pairs_capacity > 0) {
        int num_to_copy = std::min(total_collisions_detected, max_pairs_capacity);
        std::vector<NBodyCUDA::GpuCollisionPair> h_pairs_buffer(num_to_copy);
        
        CHECK_CUDA_PYBIND(cudaMemcpy(h_pairs_buffer.data(), d_colliding_pairs_buffer, 
                                     num_to_copy * sizeof(NBodyCUDA::GpuCollisionPair), cudaMemcpyDeviceToHost));
        
        // The indices returned by the kernel are relative to the full particle list (0 to N-1).
        // The Python side receives these directly.
        cpu_colliding_pairs.reserve(num_to_copy);
        for (int k = 0; k < num_to_copy; ++k) {
            cpu_colliding_pairs.emplace_back(h_pairs_buffer[k].idx1, h_pairs_buffer[k].idx2);
        }
        if (total_collisions_detected > max_pairs_capacity) {
            py::print("Warning: Number of collisions (", total_collisions_detected, 
                      ") exceeded buffer capacity (", max_pairs_capacity, "). Some collision pairs were not stored.");
            // This print might be too verbose for pybind, consider logging on Python side based on return.
        }
    }
    
    CHECK_CUDA_PYBIND(cudaFree(d_pos));
    CHECK_CUDA_PYBIND(cudaFree(d_radii));
    CHECK_CUDA_PYBIND(cudaFree(d_colliding_pairs_buffer));

    return cpu_colliding_pairs; 
}
#endif

std::vector<SpatialHashCPU::ParticleInfo> prepare_particle_info_vector(
    py::array_t<double, py::array::c_style | py::array::forcecast> positions_active_np) {
    py::buffer_info pos_buf = positions_active_np.request();

    py::ssize_t num_active = pos_buf.shape[0];

    if (pos_buf.ndim != 2 || pos_buf.shape[1] != 3) {
        throw std::runtime_error("Positions must be an N x 3 array");
    }

    std::vector<SpatialHashCPU::ParticleInfo> p_info_vec(num_active);
    const double* pos_ptr = static_cast<const double*>(pos_buf.ptr);

    for (py::ssize_t i = 0; i < num_active; ++i) {
        p_info_vec[i].pos = {pos_ptr[i * 3 + 0], pos_ptr[i * 3 + 1], pos_ptr[i * 3 + 2]};
        p_info_vec[i].current_idx = static_cast<int>(i); // Index within this active set
    }
    return p_info_vec;
}


// --- Pybind11 Wrapper for CPU Spatial Hash Collision Detection ---
std::vector<std::tuple<int, int>> py_find_colliding_pairs_sh_cpu(
    py::array_t<double, py::array::c_style | py::array::forcecast> positions_active_np,
    py::array_t<double, py::array::c_style | py::array::forcecast> radii_active_np,
    double target_cell_size) {
    
    py::buffer_info radii_buf = radii_active_np.request();
    const double* radii_ptr = static_cast<const double*>(radii_buf.ptr);
    py::ssize_t num_active = positions_active_np.request().shape[0];
    
    if (num_active < 2) {
        return {}; // Return empty list
    }

    std::vector<SpatialHashCPU::ParticleInfo> p_info_vec = prepare_particle_info_vector(
        positions_active_np
    );

    // Call your C++ CPU implementation
    // Assuming your C++ functions are in NBodyCPU namespace, adjust if not.
    // If they are global, remove NBodyCPU::
    std::vector<std::tuple<int, int>> colliding_pairs = 
        SpatialHashCPU::find_colliding_pairs_spatial_hash_cpu( // Or just find_colliding_pairs_spatial_hash_cpu
            p_info_vec.data(), 
            radii_ptr,
            static_cast<int>(num_active), 
            target_cell_size
        );
    
    // Pybind11 automatically converts std::vector<std::tuple<int, int>> to Python list of tuples
    return colliding_pairs;
}

// --- Pybind11 Wrapper for CPU Spatial Hash Get Min Distance Array ---
py::array_t<double> py_get_min_dist_array_sh_cpu(
    py::array_t<double, py::array::c_style | py::array::forcecast> positions_active_np,
    double target_cell_size) {

    py::ssize_t num_active = positions_active_np.request().shape[0];
    
    // Create an empty NumPy array for the results first
    // Shape is (num_active,)
    py::array_t<double> min_dists_np_out({num_active});
    py::buffer_info min_dists_buf = min_dists_np_out.request();
    double* min_dists_ptr = static_cast<double*>(min_dists_buf.ptr);

    if (num_active == 0) {
        return min_dists_np_out; // Return empty array
    }
    if (num_active == 1) { // Handle single particle case specifically if C++ doesn't
        min_dists_ptr[0] = std::numeric_limits<double>::infinity();
        return min_dists_np_out;
    }
    
    std::vector<SpatialHashCPU::ParticleInfo> p_info_vec = prepare_particle_info_vector(positions_active_np);

    // Call your C++ CPU implementation
    SpatialHashCPU::get_min_dist_array_spatial_hash_cpu( // Or just get_min_dist_array_spatial_hash_cpu
        min_dists_ptr,       // Pass pointer to the NumPy array's buffer
        p_info_vec.data(),
        static_cast<int>(num_active),
        target_cell_size
    );

    return min_dists_np_out;
}

py::array_t<double> py_compute_accelerations_bh_cpu(
    py::array_t<double, py::array::c_style | py::array::forcecast> positions_active_np,
    py::array_t<double, py::array::c_style | py::array::forcecast> masses_active_np,
    double G,
    double epsilon,
    double theta) {

    // Request buffer information from input NumPy arrays
    py::buffer_info pos_buf = positions_active_np.request();
    py::buffer_info mass_buf = masses_active_np.request();

    // Validate inputs (dimensions, types - though forcecast helps with type)
    if (pos_buf.ndim != 2 || pos_buf.shape[1] != 3) {
        throw std::runtime_error("Input positions must be an N x 3 NumPy array.");
    }
    if (mass_buf.ndim != 1) {
        throw std::runtime_error("Input masses must be an N NumPy array.");
    }
    py::ssize_t num_active_particles_s = pos_buf.shape[0];
    if (mass_buf.shape[0] != num_active_particles_s) {
        throw std::runtime_error("Number of positions and masses must match.");
    }
    int num_active_particles = static_cast<int>(num_active_particles_s);


    // Create an output NumPy array for accelerations (N x 3)
    // Pybind11 will manage the memory for this array.
    py::array_t<double> accelerations_out_np({num_active_particles_s, static_cast<py::ssize_t>(3)});
    py::buffer_info accel_out_buf = accelerations_out_np.request();

    // Get raw pointers to the data buffers
    const double* pos_ptr = static_cast<const double*>(pos_buf.ptr);
    const double* mass_ptr = static_cast<const double*>(mass_buf.ptr);
    double* accel_out_ptr = static_cast<double*>(accel_out_buf.ptr);

    // Call your C++ CPU Barnes-Hut implementation
    // Ensure the namespace matches where your function is defined (e.g., NBodyCPU or BarnesHutCPU)
    BarnesHutCPU::compute_accelerations_barnes_hut_cpu(
        accel_out_ptr,
        pos_ptr,
        mass_ptr,
        num_active_particles,
        G,
        epsilon,
        theta
    );

    return accelerations_out_np; // Return the NumPy array (which was filled by the C++ function)
}

std::vector<std::tuple<int, int>> py_find_colliding_pairs_n2_cpu(
    py::array_t<double, py::array::c_style | py::array::forcecast> positions_active_np,
    py::array_t<double, py::array::c_style | py::array::forcecast> velocity_active_np,
    py::array_t<double, py::array::c_style | py::array::forcecast> radii_active_np,
    double dt) {
    
    py::buffer_info radii_buf = radii_active_np.request();
    py::buffer_info pos_buf = positions_active_np.request();
    py::buffer_info vel_buf = velocity_active_np.request();
    const double* radii_ptr = static_cast<const double*>(radii_buf.ptr);
    const double* pos_ptr = static_cast<const double*>(pos_buf.ptr);
    const double* vel_ptr = static_cast<const double*>(vel_buf.ptr);
    py::ssize_t num_active = pos_buf.shape[0];
    
    if (num_active < 2) {
        return {}; // Return empty list
    }

    // py::print(N2CPU::find_colliding_pairs_n2_cpu(pos_ptr, radii_ptr, static_cast<int>(num_active)));

    // Pybind11 automatically converts std::vector<std::tuple<int, int>> to Python list of tuples
    return N2CPU::find_colliding_pairs_n2_cpu(pos_ptr, vel_ptr, radii_ptr, static_cast<int>(num_active), dt);
}

py::array_t<double> py_get_min_dist_array_n2_cpu(
    py::array_t<double, py::array::c_style | py::array::forcecast> positions_active_np) {
    
    py::buffer_info pos_buf = positions_active_np.request();
    const double* pos_ptr = static_cast<const double*>(pos_buf.ptr);
    py::ssize_t num_active = pos_buf.shape[0];
    
    py::array_t<double> min_dists_np_out({num_active});
    py::buffer_info min_dists_buf = min_dists_np_out.request();
    double* min_dists_ptr = static_cast<double*>(min_dists_buf.ptr);
    
    N2CPU::get_min_dist_array_n2_cpu(min_dists_ptr, pos_ptr, static_cast<int>(num_active));

    return min_dists_np_out;
}

py::array_t<double> py_compute_accelerations_n2_cpu(
    py::array_t<double, py::array::c_style | py::array::forcecast> positions_active_np,
    py::array_t<double, py::array::c_style | py::array::forcecast> masses_active_np,
    double G,
    double epsilon) {

    // Request buffer information from input NumPy arrays
    py::buffer_info pos_buf = positions_active_np.request();
    py::buffer_info mass_buf = masses_active_np.request();

    // Validate inputs (dimensions, types - though forcecast helps with type)
    if (pos_buf.ndim != 2 || pos_buf.shape[1] != 3) {
        throw std::runtime_error("Input positions must be an N x 3 NumPy array.");
    }
    if (mass_buf.ndim != 1) {
        throw std::runtime_error("Input masses must be an N NumPy array.");
    }
    py::ssize_t num_active_particles_s = pos_buf.shape[0];
    if (mass_buf.shape[0] != num_active_particles_s) {
        throw std::runtime_error("Number of positions and masses must match.");
    }
    int num_active_particles = static_cast<int>(num_active_particles_s);

    // Create an output NumPy array for accelerations (N x 3)
    // Pybind11 will manage the memory for this array.
    py::array_t<double> accelerations_out_np({num_active_particles_s, static_cast<py::ssize_t>(3)});
    py::buffer_info accel_out_buf = accelerations_out_np.request();

    // Get raw pointers to the data buffers
    const double* pos_ptr = static_cast<const double*>(pos_buf.ptr);
    const double* mass_ptr = static_cast<const double*>(mass_buf.ptr);
    double* accel_out_ptr = static_cast<double*>(accel_out_buf.ptr);

    // Call your C++ CPU Barnes-Hut implementation
    // Ensure the namespace matches where your function is defined (e.g., NBodyCPU or BarnesHutCPU)
    N2CPU::compute_accelerations_n2_cpu(accel_out_ptr, pos_ptr, mass_ptr, num_active_particles, G, epsilon);

    return accelerations_out_np; 
}

PYBIND11_MODULE(cpp_nbody_lib, m) {
    m.doc() = "CUDA N-body kernels via Pybind11";
#ifdef USE_CUDA
    m.def("compute_accelerations_cuda_n2", &py_compute_accelerations_cuda_n2, 
          "Compute gravitational accelerations on GPU (N^2)",
          py::arg("positions"), py::arg("masses"), py::arg("G"), py::arg("epsilon"));
    
    m.def("get_min_dist_sq_cuda_n2", &py_get_min_dist_sq_cuda_n2,
          "Get minimum pairwise distance on GPU (N^2)",
          py::arg("positions"));

    m.def("find_colliding_pairs_cuda_n2", &py_find_colliding_pairs_cuda_n2,
            "Find colliding pairs on GPU (N^2)",
            py::arg("positions"), py::arg("radii"));
#endif
    m.def("find_colliding_pairs_cpu_sh", &py_find_colliding_pairs_sh_cpu,
          "Find colliding pairs using CPU spatial hashing",
          py::arg("positions_active"), 
          py::arg("radii_active"), 
          py::arg("target_cell_size"));

    m.def("get_min_dist_cpu_sh", &py_get_min_dist_array_sh_cpu,
          "Get minimum distance to another particle for each particle using CPU spatial hashing",
          py::arg("positions_active"),
          py::arg("target_cell_size"));

    m.def("compute_accelerations_cpu_barnes_hut", &py_compute_accelerations_bh_cpu,
          "Compute gravitational accelerations using CPU Barnes-Hut",
          py::arg("positions_active"), // NumPy array (N, 3) of active particle positions
          py::arg("masses_active"),    // NumPy array (N,) of active particle masses
          py::arg("G"),                // Gravitational constant (double)
          py::arg("epsilon"),          // Softening length (double)
          py::arg("theta"),            // Barnes-Hut opening angle (double)
          R"pbdoc(
            Computes gravitational accelerations for active particles using a CPU-based Barnes-Hut algorithm.
            
            Args:
                positions_active (numpy.ndarray[float64[N, 3]]): Positions of active particles.
                masses_active (numpy.ndarray[float64[N]]): Masses of active particles.
                G (float): Gravitational constant.
                epsilon (float): Softening length.
                theta (float): Barnes-Hut opening angle parameter.
            
            Returns:
                numpy.ndarray[float64[N, 3]]: Calculated accelerations for active particles.
          )pbdoc");
    
    m.def("find_colliding_pairs_cpu_n2", &py_find_colliding_pairs_n2_cpu,
            "Find colliding pairs using CPU N^2",
            py::arg("positions_active"), 
            py::arg("velocity_active"), 
            py::arg("radii_active"),
            py::arg("dt")
        );
  
    m.def("get_min_dist_cpu_n2", &py_get_min_dist_array_n2_cpu,
            "Get minimum distance to another particle for each particle using CPU N^2",
            py::arg("positions_active"));
  
    m.def("compute_accelerations_cpu_n2", &py_compute_accelerations_n2_cpu,
            "Compute gravitational accelerations using CPU N^2",
            py::arg("positions_active"), // NumPy array (N, 3) of active particle positions
            py::arg("masses_active"),    // NumPy array (N,) of active particle masses
            py::arg("G"),                // Gravitational constant (double)
            py::arg("epsilon"));
}