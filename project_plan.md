# N-Body Asteroid Formation Simulation - Project Overview

## 1. Project Goal

To simulate the formation of asteroids in a simplified solar system model. The model initially consists of a central star (Sun) and a large number of small objects (planetesimals/asteroids). The simulation will track the gravitational interactions of these objects, detect collisions, and merge colliding objects. The ultimate aim is to observe aggregation and the potential formation of larger bodies.

## 2. Core Requirements

*   **Language:** Python as the primary language for orchestration and high-level logic, with C++/CUDA for performance-critical components.
*   **Modularity:** Code structured for easy maintenance and modification of individual components (physics, collision, integration, visualization).
*   **Performance:** Capable of simulating a significant number of objects (e.g., 50,000+) efficiently, leveraging GPU acceleration (Nvidia CUDA).
*   **Physics:**
    *   N-body gravitational interactions.
    *   Spherical objects.
    *   Inelastic collisions resulting in mergers (conservation of mass and momentum).
*   **Visualization:** Interactive 3D visualization of the particle system, showing positions, and potentially coloring/sizing by mass or velocity.

## 3. Design Choices & Current Implementation (Phase 0+)

### 3.1. Overall Architecture

*   **Hybrid Python Approach:** Python for main control flow, setup, and user interface. Computationally intensive parts are designed to be offloaded to optimized kernels (initially CPU NumPy, targetting GPU).
*   **Structure of Arrays (SoA):** Particle data (position, velocity, mass, etc.) is stored in separate NumPy arrays. This is generally more performant for vectorized operations and GPU memory access patterns.
    *   **`ParticleData` Class:** Manages these arrays, including handling a dynamic number of active particles up to a pre-allocated capacity. Supports marking particles as inactive upon merging and an optional `compact()` method.

### 3.2. Physics Engine

*   **Gravitational Constant:** `G=1` (along with `M_sun=1`, `AU=1`) is used for simplified astronomical units. Time unit becomes `1 year / (2π)`.
*   **Gravity Calculation:**
    *   **Current:** Direct N^2 summation (`compute_accelerations_cpu`) implemented in Python using NumPy for vectorization.
    *   **Softening:** Gravitational softening (`epsilon`) is used to prevent singularities during close encounters.
*   **Time Integration:**
    *   **Scheme:** Leapfrog (Kick-Drift-Kick variant) for its good energy conservation and time-reversibility. Implemented in `integrator.py`.
    *   **Timestepping:**
        *   **Current:** Global adaptive timestepping. The `_calculate_adaptive_dt` method in `Simulation` class determines an effective timestep `dt_eff` based on maximum accelerations and velocities (criteria: `eta * sqrt(epsilon / |a|)` and `eta * L_char / |v|`).
        *   Each "major step" of the simulation advances time by this `dt_eff`, ensuring stability during strong interactions.

### 3.3. Initial Conditions

*   **`initial_conditions.py`:** Module for generating particle sets.
*   **Current Setup:** `generate_test_disk` creates a central star and a disk of asteroids in the XY plane with roughly circular Keplerian velocities, plus random perturbations. Customizable number of asteroids, disk radius, mass range.

### 3.4. Collision Handling (Placeholder Stage)

*   **Detection:**
    *   Currently, a basic `check_for_overlaps` function (O(N^2)) exists for reporting/debugging, comparing inter-particle distance with the sum of radii.
*   **Merging:**
    *   **Not yet implemented.** This is a critical next step. The plan is to merge particles that collide, conserving mass and momentum, and calculating a new radius (e.g., assuming constant density).

### 3.5. Visualization

*   **Static Plotting (`visualization.py`):**
    *   Uses Matplotlib to generate 2D scatter plots (XY plane) of particle positions.
    *   Particles can be sized by mass and colored by speed.
    *   Saves plots to an `output/` directory.
*   **Interactive 3D Animation (`interactive_visualizer.py`):**
    *   **Library:** VisPy, chosen for its high-performance OpenGL-based rendering suitable for large datasets.
    *   **Features:**
        *   Displays particles in 3D. Sun can be styled differently. Asteroids colored by speed, sized by mass.
        *   Interactive camera (turntable default: drag to rotate, scroll to zoom).
        *   Pause/resume simulation (`Spacebar`).
        *   Animation driven by a `vispy.app.Timer` that calls a simulation step callback.
    *   The simulation runs one adaptive KDK step per visual frame update.

### 3.6. Main Control

*   **`simulation.py`:**
    *   `Simulation` class orchestrates the simulation loop, particle data, physics calls, and integration.
    *   Supports both batch run mode (`run()`) for generating many steps with static plots, and interactive mode (`run_interactive()`) using VisPy.
*   **`main.py`:** Entry point script, sets up parameters, and starts the chosen simulation mode.

## 4. Future Development & Refactoring

### 4.1. Performance Optimization (High Priority)

1.  **GPU Acceleration for Gravity:**
    *   Port `compute_accelerations_cpu` to a GPU kernel.
    *   **Options:**
        *   **CuPy:** Implement using CuPy's array operations or `cupy.RawKernel`. Good for staying within Python ecosystem.
        *   **Numba CUDA:** Decorate Python/NumPy code with `@numba.cuda.jit`.
        *   **C++/CUDA Kernels with Python Bindings (Pybind11):** For maximum control and performance.
    *   Handle data transfer between CPU (NumPy) and GPU (CuPy/CUDA device memory) efficiently. Aim to keep data on GPU as much as possible.
2.  **GPU Acceleration for Integration:** The `kick` and `drift` steps are O(N) and easily parallelizable on the GPU.
3.  **Advanced Gravity Algorithms (for >100k-1M+ particles):**
    *   If N^2 on GPU becomes too slow for even larger N, consider:
        *   **Barnes-Hut (O(N log N)):** Tree-based method, approximates far-field forces.
        *   **Fast Multipole Method (FMM, O(N)):** More complex, highly accurate.

### 4.2. Collision Detection & Handling (High Priority)

1.  **Efficient Broad-Phase Collision Detection (GPU/CPU):**
    *   Replace O(N^2) overlap check with a spatial subdivision method:
        *   **Spatial Hashing:** Good for sparse distributions, relatively easy to parallelize.
        *   **Uniform Grid / Octree (for 3D):** More complex data structures, potentially better for clustered particles.
    *   Goal: Reduce the number of pairs needing precise narrow-phase checks.
2.  **Narrow-Phase Collision Check:**
    *   For candidate pairs from broad-phase, perform `distance < R1 + R2`.
3.  **Collision Resolution (Merging Logic):**
    *   Implement robust merging:
        *   Conserve mass and linear momentum.
        *   Calculate new radius (e.g., `r_new^3 = r1^3 + r2^3` for constant density).
        *   Handle multiple simultaneous collisions in a single timestep carefully (e.g., process sequentially based on a criterion, or more complex simultaneous resolution).
        *   Update `ParticleData`: mark one particle inactive, update the survivor. Call `compact()` periodically or when `num_inactive / capacity` is high.
4.  **Collision Time Prediction & Refinement (Optional Enhancement):**
    *   For pairs very close to collision, calculate the precise time-to-impact (`t_coll`).
    *   If `t_coll < dt_eff`, advance the system by `t_coll`, resolve collision, then advance by `dt_eff - t_coll`. This improves accuracy of collision timing.

### 4.3. Physics & Simulation Enhancements

1.  **Rotational Dynamics:** Consider particle spin and angular momentum (much more complex).
2.  **Fragmentation:** Allow particles to break apart during high-energy collisions instead of always merging.
3.  **Gas Dynamics:** If simulating protoplanetary disks, include gas drag or other gas interactions (major addition).
4.  **More Sophisticated Initial Conditions:** Implement more realistic distributions for planetesimals.
5.  **Boundary Conditions:** Define behavior if particles leave the simulation volume (e.g., remove, reflect).
6.  **Energy/Momentum Tracking:** Regularly calculate and log total system energy and momentum to monitor conservation.

### 4.4. Visualization & User Interface

1.  **GPU Data for VisPy:** If simulation data is on GPU (CuPy), update VisPy visuals directly from GPU memory to avoid CPU-GPU-CPU transfers (e.g., via `__cuda_array_interface__` or shared OpenGL contexts).
2.  **Advanced Visual Effects:** Shaders for glittering, trails, etc.
3.  **Data Export:** Options to save simulation snapshots in standard formats (e.g., HDF5, VTK) for analysis with tools like ParaView.
4.  **GUI Controls:** For changing simulation parameters on-the-fly (e.g., using PyQt/Dear PyGui alongside VisPy).

### 4.5. Code Refactoring & Modularity

*   **Configuration Files:** Move simulation parameters (G, epsilon, eta, output paths, etc.) to a configuration file (e.g., YAML, JSON, INI) instead of hardcoding.
*   **Testing:** Implement unit tests for core components (integrator, gravity for simple cases, collision resolution).
*   **Documentation:** Add more detailed docstrings, comments, and potentially Sphinx documentation.
*   **Logging:** Use Python's `logging` module for more structured output.

## 5. Collaboration Guidelines

*   **Version Control:** Use Git for all code changes. Adhere to a branching strategy (e.g., feature branches merged into `main` via Pull Requests).
*   **Issue Tracking:** Use GitHub Issues (or similar) to track bugs, features, and tasks.
*   **Code Style:** Follow a consistent code style (e.g., PEP 8 for Python). Use linters (Flake8, Pylint) and formatters (Black, autopep8).
*   **Communication:** Regular discussion on design choices, progress, and roadblocks.
*   **Modular Commits:** Make small, logical commits with clear messages.
*   **Code Reviews:** For significant changes, a review process by another collaborator is beneficial.

This document provides a snapshot of the project's current state and future direction. It should be updated as the project evolves.