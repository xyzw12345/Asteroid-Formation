#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> // For returning std::vector<std::tuple<int,int>> if needed
#include "nbody_kernels.h" // Your C++ wrappers for CUDA calls
#include <cuda_runtime.h> // For cudaMalloc, cudaMemcpy, etc.

namespace py = pybind11;

// Helper to check CUDA calls from binding code
#define CHECK_CUDA_PYBIND(err) \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA Error in Pybind: ") + cudaGetErrorString(err)); \
    }


// Wrapper for compute_accelerations_cuda
py::array_t<double> py_compute_accelerations_cuda(
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
    NBodyCUDA::compute_accelerations_cuda(d_accel, d_pos, d_mass, num_particles, G, epsilon * epsilon);

    // Prepare result NumPy array (allocate on host)
    py::ssize_t num_particles_s = static_cast<py::ssize_t>(pos_buf.shape[0]);
    py::array_t<double> accelerations_np_out({num_particles_s, static_cast<py::ssize_t>(3)});
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
double py_get_min_pairwise_dist_cuda(
    py::array_t<double, py::array::c_style | py::array::forcecast> positions_np) {
    
    py::buffer_info pos_buf = positions_np.request();
    int num_particles = static_cast<int>(pos_buf.shape[0]);

    if (num_particles < 2) return std::numeric_limits<double>::infinity();

    double *d_pos;
    CHECK_CUDA_PYBIND(cudaMalloc(&d_pos, num_particles * 3 * sizeof(double)));
    CHECK_CUDA_PYBIND(cudaMemcpy(d_pos, pos_buf.ptr, num_particles * 3 * sizeof(double), cudaMemcpyHostToDevice));

    double min_dist_sq = NBodyCUDA::get_min_pairwise_dist_sq_cuda(d_pos, num_particles);

    CHECK_CUDA_PYBIND(cudaFree(d_pos));
    
    return std::sqrt(min_dist_sq);
}

std::vector<std::tuple<int, int>> py_find_colliding_pairs_cuda(
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

    int total_collisions_detected = NBodyCUDA::find_colliding_pairs_cuda(
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
            // py::print("Warning: Number of collisions (", total_collisions_detected, 
            //           ") exceeded buffer capacity (", max_pairs_capacity, "). Some collision pairs were not stored.");
            // This print might be too verbose for pybind, consider logging on Python side based on return.
        }
    }
    
    CHECK_CUDA_PYBIND(cudaFree(d_pos));
    CHECK_CUDA_PYBIND(cudaFree(d_radii));
    CHECK_CUDA_PYBIND(cudaFree(d_colliding_pairs_buffer));

    return cpu_colliding_pairs; 
}

PYBIND11_MODULE(cuda_nbody_lib, m) {
    m.doc() = "CUDA N-body kernels via Pybind11";
    m.def("compute_accelerations", &py_compute_accelerations_cuda, 
          "Compute gravitational accelerations on GPU (N^2)",
          py::arg("positions"), py::arg("masses"), py::arg("G"), py::arg("epsilon"));
    
    m.def("get_min_pairwise_dist", &py_get_min_pairwise_dist_cuda,
          "Get minimum pairwise distance on GPU (N^2)",
          py::arg("positions"));

    m.def("find_colliding_pairs", &py_find_colliding_pairs_cuda,
            "Find colliding pairs on GPU (N^2 placeholder)",
            py::arg("positions"), py::arg("radii"));
}