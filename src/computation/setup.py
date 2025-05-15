print("DEBUG: setup.py started")
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import os
import subprocess
import sys # For platform checks
import json

print("DEBUG: Imports done")

try:
    from setuptools import setup, Extension
    from setuptools.command.build_ext import build_ext
    import pybind11
    print("DEBUG: setuptools and pybind11 imported successfully")
except ImportError as e:
    print(f"FATAL ERROR: Could not import basic build dependencies: {e}")
    sys.exit(1)

def find_msvc_paths():
    """
    Tries to find MSVC cl.exe, include, and lib paths using vswhere.exe.
    Returns a dictionary with 'bin_path', 'include_path', 'lib_path' or None if not found.
    """
    msvc_paths = {}
    try:
        # Path to vswhere.exe (usually in Program Files (x86)\Microsoft Visual Studio\Installer)
        vswhere_path = os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"),
                                   "Microsoft Visual Studio", "Installer", "vswhere.exe")
        if not os.path.exists(vswhere_path):
            print("DEBUG: vswhere.exe not found at default location.")
            return None

        # Query for the latest stable version of VS that has the C++ workload
        # We need installationPath and version for specific toolset paths
        cmd = [
            vswhere_path,
            "-latest",
            "-products", "*",
            "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", # C++ compiler tools
            "-property", "installationPath",
            "-format", "value" # Changed to value for simpler parsing for one property
        ]
        print(f"DEBUG: Running vswhere: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        installation_path = result.stdout.strip()

        if not installation_path:
            print("DEBUG: vswhere did not return an installation path.")
            return None
        
        print(f"DEBUG: Found VS Installation Path: {installation_path}")

        # Common relative paths from the installationPath
        # These might need adjustment based on VS version (e.g., 2019 vs 2022, Community vs BuildTools)
        # This is a heuristic; a specific VC version might be needed.
        # For cl.exe (Hostx64, target x64)
        # Example: C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.38.33130\bin\Hostx64\x64
        # We need to find the latest toolset version under VC\Tools\MSVC
        
        msvc_tools_path = os.path.join(installation_path, "VC", "Tools", "MSVC")
        if not os.path.isdir(msvc_tools_path):
            print(f"DEBUG: MSVC tools path not found: {msvc_tools_path}")
            return None

        # Get the latest versioned toolset directory
        latest_toolset_version = ""
        for item in os.listdir(msvc_tools_path):
            if os.path.isdir(os.path.join(msvc_tools_path, item)) and item[0].isdigit(): # Basic check for version folder
                if item > latest_toolset_version: # Simple string comparison works for versions like "14.38.33130"
                    latest_toolset_version = item
        
        if not latest_toolset_version:
            print(f"DEBUG: Could not determine latest MSVC toolset version in {msvc_tools_path}")
            return None
        print(f"DEBUG: Using MSVC Toolset Version: {latest_toolset_version}")

        msvc_version_path = os.path.join(msvc_tools_path, latest_toolset_version)

        msvc_paths['bin_path'] = os.path.join(msvc_version_path, "bin", "Hostx64", "x64")
        msvc_paths['include_path'] = os.path.join(msvc_version_path, "include")
        msvc_paths['lib_path'] = os.path.join(msvc_version_path, "lib", "x64")

        # Also need Windows SDK paths (UCRT etc.) - vswhere doesn't easily give these.
        # The Developer Command Prompt sets these via complex scripts.
        # For now, let's focus on cl.exe path. This might not be enough.
        print(f"DEBUG: Heuristic MSVC bin path: {msvc_paths['bin_path']}")
        if not os.path.exists(os.path.join(msvc_paths['bin_path'], "cl.exe")):
            print(f"DEBUG: cl.exe not found in determined path: {msvc_paths['bin_path']}")
            return None
            
        return msvc_paths

    except (subprocess.CalledProcessError, FileNotFoundError, Exception) as e:
        print(f"DEBUG: Error finding MSVC paths using vswhere: {e}")
        return None

# Define a custom build_ext to handle CUDA compilation
class CudaBuildExt(build_ext):
    def build_extensions(self):
        print("DEBUG: CudaBuildExt build_extensions started")
        
        # --- Attempt to set up MSVC environment if not in Developer Prompt ---
        # Check if cl.exe is already in PATH (might be if run from Dev Prompt)
        cl_exe_in_path = False
        try:
            subprocess.check_output(['cl.exe', '/?'], stderr=subprocess.STDOUT)
            cl_exe_in_path = True
            print("DEBUG: cl.exe seems to be in PATH already.")
        except (OSError, subprocess.CalledProcessError):
            print("DEBUG: cl.exe not found in PATH. Attempting to locate MSVC.")
            msvc_env_paths = find_msvc_paths()
            if msvc_env_paths and 'bin_path' in msvc_env_paths:
                print(f"DEBUG: Adding MSVC bin path to os.environ['PATH']: {msvc_env_paths['bin_path']}")
                os.environ["PATH"] = msvc_env_paths['bin_path'] + os.pathsep + os.environ["PATH"]
                # For a more complete environment, you'd also set INCLUDE and LIB based on msvc_env_paths
                # and Windows SDK paths, which is much harder to do robustly here.
                if msvc_env_paths.get('include_path'):
                     os.environ["INCLUDE"] = msvc_env_paths['include_path'] + os.pathsep + os.environ.get("INCLUDE", "")
                if msvc_env_paths.get('lib_path'):
                     os.environ["LIB"] = msvc_env_paths['lib_path'] + os.pathsep + os.environ.get("LIB", "")
            else:
                print("WARNING: Could not automatically configure MSVC environment. Compilation might fail.")

        # Ensure nvcc is available
        try:
            subprocess.check_output(['nvcc', '--version'])
        except OSError:
            raise RuntimeError(
                "NVCC (NVIDIA CUDA Compiler) not found. Make sure it's in your PATH.")

        cuda_path = os.environ.get("CUDA_PATH") # e.g., C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/vXX.Y
        if not cuda_path:
            print("Warning: CUDA_PATH environment variable not set. CUDA compilation/linking might use system defaults or fail.")
            # Attempt to find common default paths if CUDA_PATH is not set (more robust for users)
            if sys.platform == "win32":
                # Common default paths for Windows
                program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
                possible_paths = [os.path.join(program_files, f"NVIDIA GPU Computing Toolkit\\CUDA\\v{ver}") for ver in ["12.1", "12.0", "11.8", "11.7"]] # Add more versions
                for p in possible_paths:
                    if os.path.exists(p):
                        cuda_path = p
                        print(f"Found CUDA Toolkit at: {cuda_path}")
                        break
            elif sys.platform == "linux":
                if os.path.exists("/usr/local/cuda"):
                    cuda_path = "/usr/local/cuda" # Common symlink
                    print(f"Found CUDA Toolkit at: {cuda_path}")
            # Add more platform specific detections if needed

        for ext in self.extensions:
            if not (hasattr(ext, '_needs_cuda') and ext._needs_cuda):
                continue # Skip non-CUDA extensions

            # --- CUDA specific setup for the extension ---
            if cuda_path:
                ext.include_dirs.append(os.path.join(cuda_path, 'include'))
                lib_dir = 'lib/x64' if sys.platform == "win32" else 'lib64'
                ext.library_dirs.append(os.path.join(cuda_path, lib_dir))
            
            if sys.platform == "win32":
                ext.extra_link_args.append('cudart_static.lib') # Or cudart.lib if dynamic
            else: # Linux/macOS
                ext.extra_link_args.append('-lcudart')
                if cuda_path: # Add lib path for linker if found
                     ext.extra_link_args.append(f'-L{os.path.join(cuda_path, "lib64")}')


            # --- Compile .cu files to object files ---
            cu_sources = [s for s in ext.sources if s.endswith('.cu')]
            other_sources = [s for s in ext.sources if not s.endswith('.cu')]
            
            objects = []
            for cu_source in cu_sources:
                base_name = os.path.splitext(cu_source)[0]
                obj_name = base_name + ('.obj' if sys.platform == "win32" else '.o')
                
                # Ensure build directory exists
                os.makedirs(os.path.join(self.build_temp, os.path.dirname(cu_source)), exist_ok=True)
                target_obj_path = os.path.join(self.build_temp, obj_name)

                # Add include paths for nvcc
                nvcc_include_args = ['-I"{}"'.format(inc) for inc in ext.include_dirs]
                
                # Specify GPU architecture (important!)
                
                # For RTX 4060, sm_89 is Ada Lovelace. You might also want to include sm_86 (Ampere).
                # Check your GPU's compute capability. RTX 4060 is Ada Lovelace (sm_89).
                # For broader compatibility, you can target multiple architectures.
                # A common set for modern cards:
                gpu_arch_flags = [
                    '-gencode=arch=compute_70,code=sm_70', # Volta
                    '-gencode=arch=compute_75,code=sm_75', # Turing
                    '-gencode=arch=compute_80,code=sm_80', # Ampere (GA100)
                    '-gencode=arch=compute_86,code=sm_86', # Ampere (GA10x)
                    '-gencode=arch=compute_89,code=sm_89', # Ada Lovelace
                    '-gencode=arch=compute_90,code=sm_90', # Hopper
                ]

                # Other NVCC flags
                nvcc_flags = ['-O3', '--use_fast_math'] # Example
                if sys.platform == "win32":
                    nvcc_flags.extend(['-Xcompiler', '/EHsc,/W3,/MD']) # Pass flags to host MSVC compiler
                else: # Linux
                    nvcc_flags.extend(['-Xcompiler', '-fPIC,-Wall,-Wextra'])

                # Command to compile .cu to .o/.obj
                compile_command = ['nvcc'] + \
                                  nvcc_flags + \
                                  gpu_arch_flags + \
                                  nvcc_include_args + \
                                  ['-c', cu_source, '-o', target_obj_path]
                
                print(f"Compiling CUDA source: {' '.join(compile_command)}")
                try:
                    subprocess.check_call(compile_command)
                    objects.append(target_obj_path)
                except subprocess.CalledProcessError as e:
                    print(f"Error compiling {cu_source}: {e}")
                    raise

            # Replace .cu sources with their compiled object files for the C++ compiler/linker
            ext.sources = other_sources
            ext.extra_objects = objects # Pass pre-compiled .cu objects to the linker

        super().build_extensions()


# In setup.py
cpp_args = []
if sys.platform == "win32": # MSVC
    cpp_args.extend(['/std:c++17', '/O2', '/EHsc', '/MD'])
else: # GCC/Clang
    cpp_args.extend(['-std=c++17', '-O3', '-fPIC', '-Wall', '-Wextra'])

cuda_extension = Extension(
    'cpp_nbody_lib', 
    sources=[
        os.path.join('cpp_cuda_ext', 'binding.cpp'), 
        os.path.join('cpp_cuda_ext', 'n2_kernels.cu') # Keep .cu here for CudaBuildExt to find
    ], 
    include_dirs=[
        pybind11.get_include()
    ],
    language='c++',
    extra_compile_args=cpp_args # For C++ files
)
cuda_extension._needs_cuda = True


setup(
    name='cpp_nbody_lib',
    version='0.0.1',
    ext_modules=[cuda_extension],
    cmdclass={'build_ext': CudaBuildExt}
)