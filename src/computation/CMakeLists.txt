cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(NBodyPythonExt LANGUAGES CXX CUDA) # CXX is primary, CUDA is optional

# --- Python Configuration ---
# Find Python. This is crucial for pybind11 and building extensions.
# COMPONENTS Interpreter Development.Module: Ensure we find headers, libs, and can make modules
# REQUIRED: The build will fail if Python isn't found.
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
message(STATUS "Python Interpreter: ${Python_EXECUTABLE}")
message(STATUS "Python Include Dirs: ${Python_INCLUDE_DIRS}")
message(STATUS "Python Libraries: ${Python_LIBRARIES}") # Might be empty on some systems if not needed for linking modules

# --- Pybind11 Configuration ---
# Find Pybind11. It provides its own CMake configuration files.
# If pybind11 was installed via pip in the active Python environment,
# Python_SITELIB (from find_package(Python)) might help CMake find it.
# Or, if pybind11 is a submodule or installed elsewhere, you might need to set pybind11_DIR.
# For pip install:
# --- Pybind11 Configuration ---
# Find Pybind11.

# If pybind11 was installed via pip in the active Python environment:
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE pybind11_cmakedir_result
)

if(NOT pybind11_cmakedir_result EQUAL 0 OR NOT EXISTS "${pybind11_CMAKE_DIR}/pybind11Config.cmake")
    message(STATUS "pybind11 --cmakedir failed or did not return a valid path. Trying generic find_package.")
    # Fallback to generic find_package if --cmakedir didn't work or isn't what we want
    # This might require pybind11_DIR to be set manually or CMAKE_PREFIX_PATH
    set(PYBIND11_FINDPYTHON ON) # Tell pybind11 to use the Python found by find_package(Python)
    find_package(pybind11 CONFIG QUIET) # Try quiet first
else()
    message(STATUS "Found pybind11 CMake directory via --cmakedir: ${pybind11_CMAKE_DIR}")
    # Add the found directory to CMake's search path for package configurations
    list(APPEND CMAKE_PREFIX_PATH "${pybind11_CMAKE_DIR}")
    # Now find_package should work using this path
    find_package(pybind11 CONFIG REQUIRED)
endif()

# Check if pybind11 was actually found by either method
if(NOT pybind11_FOUND)
    message(FATAL_ERROR "Pybind11 not found. Please install pybind11 (e.g., pip install pybind11) "
                        "or set pybind11_DIR to its installation path containing pybind11Config.cmake.")
endif()

message(STATUS "Pybind11 Include Dirs: ${pybind11_INCLUDE_DIRS}")
message(STATUS "Pybind11 CMake Module Path (if found by find_package directly): ${pybind11_CMAKE_DIR}")

# --- Standard C++ Configuration ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- CUDA Configuration (Optional) ---
option(PROJECT_ENABLE_CUDA "Build with CUDA support if toolkit is found" ON) # User can turn this OFF
set(CUDA_FOUND FALSE) # Our own flag

if(PROJECT_ENABLE_CUDA)
    # Try to find CUDA toolkit. QUIETLY to not error if not found.
    find_package(CUDAToolkit QUIET) # This sets CUDAToolkit_FOUND, CUDAToolkit_INCLUDE_DIRS, etc.

    if(CUDAToolkit_FOUND)
        set(CUDA_FOUND TRUE)
        message(STATUS "CUDA Toolkit Found: Version ${CUDAToolkit_VERSION}")
        message(STATUS "CUDA Include Dirs: ${CUDAToolkit_INCLUDE_DIRS}")
        message(STATUS "NVCC Executable: ${CUDAToolkit_NVCC_EXECUTABLE}")

        # Enable CUDA language for the project if found and enabled
        # project() command already did this if CUDAToolkit was found before it.
        # If project() didn't have CUDA initially, you might need:
        # enable_language(CUDA) 
        # But having it in project() is cleaner.

        set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
        set(CMAKE_CUDA_STANDARD_REQUIRED True)
        set(CMAKE_CUDA_EXTENSIONS OFF)

        add_compile_definitions(USE_CUDA) # Define this for conditional C++ code

        # Specify CUDA Architectures
        set(CMAKE_CUDA_ARCHITECTURES 70 75 86 89 NATIVE CACHE STRING "CUDA Architectures")
        message(STATUS "Compiling for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

        # If using CMAKE_CUDA_HOST_COMPILER, set it here if needed (e.g., for MSVC on Ninja)
        if(WIN32 AND CMAKE_GENERATOR STREQUAL "Ninja" AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
             set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "Host compiler for CUDA")
        endif()

    else()
        message(WARNING "CUDA Toolkit not found. Building CPU-only version of the extension.")
    endif()
else()
    message(STATUS "CUDA support explicitly disabled via PROJECT_ENABLE_CUDA=OFF.")
endif()

# --- Source Files for the Python Extension ---
set(EXTENSION_NAME "cpp_nbody_lib") # Name of the Python module to be generated

set(CPP_SOURCES
    cpp_cuda_ext/binding.cpp
    cpp_cuda_ext/src_cpp/spatial_hash_cpu.cpp
    cpp_cuda_ext/src_cpp/barnes_hut_cpu.cpp
)

set(CUDA_SOURCES ) # Initialize empty
if(CUDA_FOUND)
    list(APPEND CUDA_SOURCES
        cpp_cuda_ext/src_cuda/n2_kernels.cu
    )
    # If you have a cuda_utils.cu, add it here as well.
endif()

# --- Build the Python Extension Module ---
# pybind11_add_module creates a shared library target suitable for Python import.
# It handles Python-specific compiler/linker flags, output naming, etc.
pybind11_add_module(${EXTENSION_NAME} MODULE # MODULE type is for Python extensions
    ${CPP_SOURCES}
    ${CUDA_SOURCES} # Pass CUDA sources here; pybind11_add_module can handle them
)

# --- Target Specific Configurations ---

# Include directories for our extension target
target_include_directories(${EXTENSION_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp_cuda_ext"       # For headers within cpp_src
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp_cuda_ext/src_cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp_cuda_ext/src_cuda"
    # pybind11_INCLUDE_DIRS is automatically added by pybind11_add_module
    # Python_INCLUDE_DIRS is also automatically added by pybind11_add_module
)

if(CUDA_FOUND)
    # Add CUDA toolkit include directory for C++ files if they include CUDA headers
    # (e.g., cuda_runtime.h via a utility header). pybind11_add_module might already do this.
    target_include_directories(${EXTENSION_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()


# --- Compiler Flags for the Extension Target ---
# These are similar to your setup.py's cpp_args and nvcc_flags

# C++ Compiler Flags
if(MSVC)
    target_compile_options(${EXTENSION_NAME} PRIVATE /std:c++17 /O2 /EHsc /MD) # /MD for dynamic runtime
else() # GCC/Clang
    target_compile_options(${EXTENSION_NAME} PRIVATE -std=c++17 -O3 -fPIC -Wall -Wextra -fopenmp)
endif()

# CUDA Compiler Flags (if CUDA is enabled)
if(CUDA_FOUND)
    # Device code flags
    target_compile_options(${EXTENSION_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3> # Optimization level
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math> # Optional
        # Add other device flags like -ftz=true if needed
    )
    # Host compiler flags (passed via -Xcompiler by nvcc)
    if(MSVC)
        target_compile_options(${EXTENSION_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler="/EHsc,/W3,/MD">
        )
    else() # GCC/Clang host
        target_compile_options(${EXTENSION_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC,-Wall,-Wextra>
        )
    endif()
    # Note: pybind11_add_module and CMAKE_CUDA_ARCHITECTURES handle GPU arch flags.
    # Debug/Release specific flags are often handled by CMAKE_BUILD_TYPE and
    # CMake's default CMAKE_<LANG>_FLAGS_<CONFIG> variables, which pybind11_add_module respects.
    # You can add more specific debug/release flags here using generator expressions if needed.
endif()

# --- Linker Flags for the Extension Target ---
if(CUDA_FOUND)
    # pybind11_add_module often handles Python library linking.
    # We need to ensure CUDA runtime is linked if we have CUDA code.
    # CMake's CUDA language support usually links cudart automatically for targets with .cu files.
    # If not, explicitly:
    # target_link_libraries(${EXTENSION_NAME} PRIVATE CUDA::cudart_static) # Or CUDA::cudart for shared
    # Your setup.py used cudart_static.lib for Windows.
    if(WIN32)
        target_link_libraries(${EXTENSION_NAME} PRIVATE CUDA::cudart_static)
    else()
        target_link_libraries(${EXTENSION_NAME} PRIVATE CUDA::cudart)
    endif()
endif()
# Python libraries are typically handled by pybind11_add_module.


# --- Installation (Optional but good for Python packages) ---
# This will install the compiled .pyd/.so file into the Python environment's site-packages
# when `cmake --install .` or `pip install .` (if using scikit-build) is run.
# pybind11_install places it correctly for Python to find.
# pybind11_install(${EXTENSION_NAME}) # Requires pybind11_DIR to be set, usually is after find_package


# --- Output some info ---
message(STATUS "Building Python extension: ${EXTENSION_NAME}")
if(CUDA_FOUND)
    message(STATUS "CUDA support: ENABLED for extension")
else()
    message(STATUS "CUDA support: DISABLED for extension")
endif()

# This line is crucial if you want `python setup.py build` or `pip install .` to work
# with CMake as the backend build system (using scikit-build or similar).
# For a pure CMake build, you'd just build the ${EXTENSION_NAME} target.
# If you are using scikit-build, it will find and call this CMakeLists.txt.
# If you are building manually with CMake, the output will be in your build directory.
# e.g., build/cpp_nbody_lib.cpython-39-x86_64-linux-gnu.so
# The exact name and location of the output module are handled by pybind11_add_module.
# To make it easily importable from the build directory for testing:
set_target_properties(${EXTENSION_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set_target_properties(${EXTENSION_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
# For Windows, also set the PDB output directory if building with MSVC debug info
if(MSVC)
    set_target_properties(${EXTENSION_NAME} PROPERTIES PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()