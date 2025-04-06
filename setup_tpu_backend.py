#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for building and installing the TPU backend for llama-cpp-python
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and return its output"""
    print(f"Running: {' '.join(cmd if isinstance(cmd, list) else [cmd])}")
    result = subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True, shell=not isinstance(cmd, list))
    return result.stdout

def build_tpu_backend():
    """Build the TPU backend"""
    # Create build directory
    build_dir = Path("build_tpu")
    build_dir.mkdir(exist_ok=True)

    # Check for CMakeLists_tpu.txt
    cmake_file = Path("CMakeLists_tpu.txt")
    if not cmake_file.exists():
        print(f"Error: {cmake_file} not found.")
        print("Creating a basic CMakeLists_tpu.txt file...")
        
        # Create a basic CMakeLists_tpu.txt file
        with open(cmake_file, "w") as f:
            f.write("""
cmake_minimum_required(VERSION 3.12)
project(llama_cpp_tpu_backend)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find PyTorch package
find_package(Torch REQUIRED)
message(STATUS "PyTorch libraries: ${TORCH_LIBRARIES}")

# Find PyTorch/XLA
set(TORCH_XLA_DIR "" CACHE PATH "Path to PyTorch/XLA installation")
if(NOT TORCH_XLA_DIR)
    message(FATAL_ERROR "TORCH_XLA_DIR must be set to the PyTorch/XLA installation directory")
endif()

include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_XLA_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp
)

# Find llama.cpp source files
set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp" CACHE PATH "Path to llama.cpp repository")
if(NOT EXISTS ${LLAMA_CPP_DIR})
    message(FATAL_ERROR "llama.cpp repository not found at ${LLAMA_CPP_DIR}")
endif()

# Source files for the TPU backend
set(SOURCE_FILES 
    src/tpu_backend.cpp
)

# Create the TPU backend library
add_library(llama_cpp_tpu_backend SHARED ${SOURCE_FILES})

# Link against PyTorch and PyTorch/XLA
target_link_libraries(llama_cpp_tpu_backend 
    ${TORCH_LIBRARIES}
    ${TORCH_XLA_DIR}/lib/libtorch_xla.so
)

# Install the library
install(TARGETS llama_cpp_tpu_backend
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
""")
        print(f"Created {cmake_file}")
    
    # Check for source file
    src_dir = Path("src")
    src_dir.mkdir(exist_ok=True)
    
    tpu_backend_src = src_dir / "tpu_backend.cpp"
    if not tpu_backend_src.exists():
        print(f"Error: {tpu_backend_src} not found.")
        print("Creating a basic tpu_backend.cpp file...")
        
        # Create a basic tpu_backend.cpp file
        with open(tpu_backend_src, "w") as f:
            f.write("""
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <torch/torch.h>
#include <torch_xla/csrc/tensor.h>
#include <torch_xla/csrc/ops/ops.h>
#include <torch_xla/csrc/device.h>
#include <torch_xla/csrc/runtime/computation_client.h>

#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>

// Forward declarations
static void ggml_backend_tpu_free(void * ctx);
static void ggml_backend_tpu_set_tensor(ggml_backend_t backend, ggml_tensor * tensor);
static void ggml_backend_tpu_get_tensor(ggml_backend_t backend, ggml_tensor * tensor);
static void ggml_backend_tpu_mul_mat(ggml_backend_buffer_t buffer, const ggml_tensor * a, const ggml_tensor * b, ggml_tensor * c);
static bool ggml_backend_tpu_supports_op(const ggml_backend_t backend, const ggml_tensor * op);

// TPU backend context
struct ggml_backend_tpu_context {
    bool initialized;
    int device_count;
    std::vector<torch::Device> devices;
    std::unordered_map<ggml_tensor *, torch::Tensor> tensor_map;
    std::mutex mutex;
};

// TPU buffer context
struct ggml_backend_tpu_buffer_context {
    ggml_backend_tpu_context * tpu_ctx;
    void * data;
    size_t size;
};

// Initialize TPU backend
static ggml_backend_tpu_context * ggml_backend_tpu_init(void) {
    auto * ctx = new ggml_backend_tpu_context();
    ctx->initialized = false;
    
    try {
        // Set XLA environment variables programmatically if needed
        // setenv("XLA_USE_TPU", "1", 1);
        
        // Initialize PyTorch/XLA
        auto client = torch_xla::runtime::GetComputationClient();
        if (!client) {
            std::cerr << "Failed to get XLA computation client" << std::endl;
            return ctx;
        }
        
        auto device_count = client->GetNumDevices();
        ctx->device_count = device_count;
        
        if (device_count <= 0) {
            std::cerr << "No TPU devices found" << std::endl;
            return ctx;
        }
        
        // Get available TPU devices
        for (int i = 0; i < device_count; i++) {
            auto device = torch_xla::bridge::GetDeviceOrCurrent(i);
            ctx->devices.push_back(device);
        }
        
        ctx->initialized = true;
        std::cout << "TPU backend initialized with " << device_count << " devices" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize TPU backend: " << e.what() << std::endl;
    }
    
    return ctx;
}

// Cleanup TPU backend
static void ggml_backend_tpu_free(void * ctx_ptr) {
    auto * ctx = (ggml_backend_tpu_context*)ctx_ptr;
    if (ctx) {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        ctx->tensor_map.clear();
        delete ctx;
    }
}

// Convert ggml tensor to PyTorch tensor
static torch::Tensor ggml_tensor_to_torch(ggml_tensor * tensor) {
    if (!tensor) {
        throw std::runtime_error("Null tensor");
    }
    
    // Get tensor dimensions
    std::vector<int64_t> dims;
    for (int i = GGML_MAX_DIMS - 1; i >= 0; i--) {
        if (tensor->ne[i] > 1) {
            dims.push_back(tensor->ne[i]);
        }
    }
    
    // If dimensions are empty, add at least one dimension
    if (dims.empty()) {
        dims.push_back(1);
    }
    
    // Handle tensor data type
    torch::ScalarType dtype;
    switch (tensor->type) {
        case GGML_TYPE_F32:
            dtype = torch::kFloat32;
            break;
        case GGML_TYPE_F16:
            dtype = torch::kFloat16;
            break;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_K:
            // For quantized types, first dequantize to FP32
            {
                size_t nelements = ggml_nelements(tensor);
                float* dequantized = new float[nelements];
                
                // Assuming ggml has a dequantize function we can use
                // This is a placeholder - actual implementation needed
                ggml_tensor* temp = ggml_dup_tensor(NULL, tensor);
                temp->type = GGML_TYPE_F32;
                temp->data = dequantized;
                ggml_compute_forward_dequantize(NULL, tensor, temp);
                
                // Create tensor from dequantized data
                auto options = torch::TensorOptions().dtype(torch::kFloat32);
                torch::Tensor result = torch::from_blob(dequantized, dims, 
                    [dequantized](void*) { delete[] dequantized; }, options);
                
                return result;
            }
        default:
            throw std::runtime_error("Unsupported tensor type in TPU backend");
    }
    
    // Create PyTorch tensor from ggml tensor data
    auto options = torch::TensorOptions().dtype(dtype);
    torch::Tensor torch_tensor = torch::from_blob(
        tensor->data,
        dims,
        options
    ).clone(); // Clone to own the data
    
    return torch_tensor;
}

// Convert PyTorch tensor to ggml tensor
static void torch_tensor_to_ggml(torch::Tensor torch_tensor, ggml_tensor * tensor) {
    if (!tensor) {
        throw std::runtime_error("Null tensor");
    }
    
    // Check if tensor shapes match
    std::vector<int64_t> ggml_dims;
    for (int i = GGML_MAX_DIMS - 1; i >= 0; i--) {
        if (tensor->ne[i] > 1) {
            ggml_dims.push_back(tensor->ne[i]);
        }
    }
    if (ggml_dims.empty()) {
        ggml_dims.push_back(1);
    }
    
    auto torch_sizes = torch_tensor.sizes().vec();
    if (torch_sizes != ggml_dims) {
        // Try to reshape if total element count matches
        size_t ggml_size = 1;
        for (auto d : ggml_dims) {
            ggml_size *= d;
        }
        
        size_t torch_size = 1;
        for (auto d : torch_sizes) {
            torch_size *= d;
        }
        
        if (ggml_size != torch_size) {
            throw std::runtime_error("Tensor shape mismatch");
        }
        
        torch_tensor = torch_tensor.reshape(ggml_dims);
    }
    
    // Handle tensor data type
    torch::ScalarType dtype = torch_tensor.scalar_type();
    switch (tensor->type) {
        case GGML_TYPE_F32:
            if (dtype != torch::kFloat32) {
                torch_tensor = torch_tensor.to(torch::kFloat32);
            }
            break;
        case GGML_TYPE_F16:
            if (dtype != torch::kFloat16) {
                torch_tensor = torch_tensor.to(torch::kFloat16);
            }
            break;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_K:
            // Convert to FP32 first, then we'll need to quantize
            torch_tensor = torch_tensor.to(torch::kFloat32);
            
            // For quantized types, we need to quantize back
            // This requires proper implementation
            // For now, just try to use a simple approach
            {
                torch_tensor = torch_tensor.contiguous().cpu();
                float* fp32_data = static_cast<float*>(torch_tensor.data_ptr());
                
                // Use ggml's quantization if available
                ggml_quantize_from_float(fp32_data, tensor->data, ggml_nelements(tensor), tensor->type);
                return;
            }
        default:
            throw std::runtime_error("Unsupported tensor type for TPU backend");
    }
    
    // Copy data from PyTorch tensor to ggml tensor
    torch_tensor = torch_tensor.contiguous().cpu();
    std::memcpy(tensor->data, torch_tensor.data_ptr(), ggml_nbytes(tensor));
}

// TPU backend buffer functions
static ggml_backend_buffer_t ggml_backend_tpu_buffer_alloc(ggml_backend_t backend, size_t size) {
    ggml_backend_tpu_context* tpu_ctx = (ggml_backend_tpu_context*)ggml_backend_get_context(backend);
    
    auto* buft = new ggml_backend_buffer;
    auto* ctx = new ggml_backend_tpu_buffer_context;
    
    ctx->tpu_ctx = tpu_ctx;
    ctx->size = size;
    ctx->data = malloc(size);
    
    if (ctx->data == nullptr) {
        delete ctx;
        delete buft;
        return nullptr;
    }
    
    buft->context = ctx;
    buft->size = size;
    buft->data = ctx->data;
    buft->usage = GGML_BACKEND_BUFFER_USAGE_ANY;
    
    return buft;
}

static void ggml_backend_tpu_buffer_free(ggml_backend_buffer_t buffer) {
    auto* ctx = (ggml_backend_tpu_buffer_context*)buffer->context;
    free(ctx->data);
    delete ctx;
    delete buffer;
}

// Set tensor data to TPU
static void ggml_backend_tpu_set_tensor(ggml_backend_t backend, ggml_tensor * tensor) {
    auto* ctx = (ggml_backend_tpu_context*)ggml_backend_get_context(backend);
    if (!ctx->initialized) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(ctx->mutex);
    
    try {
        // Convert ggml tensor to PyTorch tensor
        torch::Tensor torch_tensor = ggml_tensor_to_torch(tensor);
        
        // Move tensor to TPU
        if (!ctx->devices.empty()) {
            torch_tensor = torch_tensor.to(ctx->devices[0]);
        }
        
        // Store in tensor map
        ctx->tensor_map[tensor] = torch_tensor;
    } catch (const std::exception& e) {
        std::cerr << "Failed to set tensor to TPU: " << e.what() << std::endl;
    }
}

// Get tensor data from TPU
static void ggml_backend_tpu_get_tensor(ggml_backend_t backend, ggml_tensor * tensor) {
    auto* ctx = (ggml_backend_tpu_context*)ggml_backend_get_context(backend);
    if (!ctx->initialized) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(ctx->mutex);
    
    try {
        auto it = ctx->tensor_map.find(tensor);
        if (it != ctx->tensor_map.end()) {
            // Move tensor back to CPU and copy data
            torch::Tensor cpu_tensor = it->second.cpu();
            torch_tensor_to_ggml(cpu_tensor, tensor);
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to get tensor from TPU: " << e.what() << std::endl;
    }
}

// Compute matrix multiplication on TPU
static void ggml_backend_tpu_mul_mat(ggml_backend_buffer_t buffer, const ggml_tensor * a, const ggml_tensor * b, ggml_tensor * c) {
    auto* buf_ctx = (ggml_backend_tpu_buffer_context*)buffer->context;
    auto* ctx = buf_ctx->tpu_ctx;
    
    if (!ctx->initialized) {
        std::cerr << "TPU backend not initialized" << std::endl;
        return;
    }
    
    std::lock_guard<std::mutex> lock(ctx->mutex);
    
    try {
        // Convert ggml tensors to PyTorch tensors
        torch::Tensor a_torch = ggml_tensor_to_torch((ggml_tensor*)a);
        torch::Tensor b_torch = ggml_tensor_to_torch((ggml_tensor*)b);
        
        // Move tensors to TPU
        if (!ctx->devices.empty()) {
            a_torch = a_torch.to(ctx->devices[0]);
            b_torch = b_torch.to(ctx->devices[0]);
        }
        
        // Perform matrix multiplication
        torch::Tensor c_torch = torch::matmul(a_torch, b_torch);
        
        // Move result back to CPU
        c_torch = c_torch.cpu();
        
        // Copy result to ggml tensor
        torch_tensor_to_ggml(c_torch, c);
    } catch (const std::exception& e) {
        std::cerr << "TPU matrix multiplication failed: " << e.what() << std::endl;
    }
}

// Check if operation is supported
static bool ggml_backend_tpu_supports_op(const ggml_backend_t backend, const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            // Currently only matrix multiplication is supported
            return true;
        default:
            return false;
    }
}

// TPU backend implementation
static ggml_backend_i ggml_backend_tpu_interface = {
    /* .get_name        = */ [](ggml_backend_t backend) { return "TPU"; },
    /* .free           = */ ggml_backend_tpu_free,
    /* .get_mem_buffer = */ nullptr, // Not needed for our implementation
    /* .set_tensor     = */ ggml_backend_tpu_set_tensor,
    /* .get_tensor     = */ ggml_backend_tpu_get_tensor,
    /* .cpy_tensor     = */ nullptr, // Use default implementation
    /* .compute_tensor = */ nullptr, // Will be handled by the scheduler based on supported_ops
    /* .supports_op    = */ ggml_backend_tpu_supports_op,
};

// Buffer implementation
static ggml_backend_buffer_i ggml_backend_tpu_buffer_interface = {
    /* .get_name        = */ [](ggml_backend_buffer_t buffer) { return "TPU"; },
    /* .free           = */ ggml_backend_tpu_buffer_free,
    /* .get_base       = */ [](ggml_backend_buffer_t buffer) -> void* { return ((ggml_backend_tpu_buffer_context*)buffer->context)->data; },
    /* .init_tensor    = */ nullptr, // Use default implementation 
    /* .set_tensor     = */ nullptr, // Use default implementation
    /* .get_tensor     = */ nullptr, // Use default implementation
    /* .cpy_tensor     = */ nullptr, // Use default implementation
};

// Create TPU backend
ggml_backend_t ggml_backend_tpu_init(void) {
    // Initialize TPU context
    auto* ctx = ggml_backend_tpu_init();
    if (!ctx->initialized) {
        delete ctx;
        return nullptr;
    }
    
    // Create backend
    ggml_backend_t backend = new ggml_backend{
        /* .interface = */ ggml_backend_tpu_interface,
        /* .context   = */ ctx,
    };
    
    return backend;
}

// Register TPU backend buffer
static ggml_backend_buffer_type_i ggml_backend_tpu_buffer_type_interface = {
    /* .get_name        = */ [](ggml_backend_buffer_type_t buft) { return "TPU"; },
    /* .alloc_buffer    = */ [](ggml_backend_buffer_type_t buft, size_t size) -> ggml_backend_buffer_t {
        return ggml_backend_tpu_buffer_alloc((ggml_backend_t)buft->context, size);
    },
    /* .get_alignment   = */ [](ggml_backend_buffer_type_t buft) -> size_t { return 16; },
    /* .get_alloc_size  = */ nullptr, // Use default implementation
    /* .get_max_size    = */ nullptr, // No specific limit
    /* .get_device      = */ nullptr, // No specific device
    /* .is_host         = */ [](ggml_backend_buffer_type_t buft) -> bool { return false; },
};

// Create and register TPU backend buffer type
ggml_backend_buffer_type_t ggml_backend_tpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_tpu_buffer_type_obj = {
        /* .iface    = */ ggml_backend_tpu_buffer_type_interface,
        /* .context  = */ nullptr,
    };
    
    static bool initialized = false;
    if (!initialized) {
        // Initialize our TPU backend
        ggml_backend_t backend = ggml_backend_tpu_init();
        if (backend == nullptr) {
            return nullptr;
        }
        
        // Store the backend as context
        ggml_backend_tpu_buffer_type_obj.context = backend;
        initialized = true;
    }
    
    return &ggml_backend_tpu_buffer_type_obj;
}

// Register the backend with ggml
extern "C" void ggml_backend_tpu_register(void) {
    ggml_backend_register(ggml_backend_tpu_buffer_type(), ggml_backend_tpu_init());
}
""")
        print(f"Created {tpu_backend_src}")
    
    # Check for llama.cpp repository
    llama_cpp_dir = Path("vendor/llama.cpp")
    if not llama_cpp_dir.exists():
        print(f"Error: {llama_cpp_dir} not found.")
        print("Cloning llama.cpp repository...")
        
        # Create vendor directory
        Path("vendor").mkdir(exist_ok=True)
        
        # Clone llama.cpp repository
        try:
            run_command(["git", "clone", "https://github.com/ggml-org/llama.cpp.git", str(llama_cpp_dir)])
        except Exception as e:
            print(f"Error cloning llama.cpp repository: {e}")
            return False

    # Run CMake
    cmake_cmd = [
        "cmake",
        "-S", ".",
        "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
    ]

    # Add PyTorch and PyTorch/XLA paths
    try:
        import torch
        import torch_xla # type: ignore

        # Get PyTorch installation path
        torch_path = Path(torch.__file__).parent
        print(f"Found PyTorch at {torch_path}")

        # Get PyTorch/XLA installation path
        torch_xla_path = Path(torch_xla.__file__).parent
        print(f"Found PyTorch/XLA at {torch_xla_path}")

        cmake_cmd.extend([
            f"-DCMAKE_PREFIX_PATH={torch_path}",
            f"-DTORCH_DIR={torch_path}",
            f"-DTORCH_XLA_DIR={torch_xla_path}"
        ])
    except (ImportError, AttributeError) as e:
        print(f"Warning: Could not find PyTorch or PyTorch/XLA. Error: {e}")
        print("Make sure both PyTorch and PyTorch/XLA are installed.")
        return False

    # Use the CMakeLists_tpu.txt file
    cmake_cmd.append("-C")
    cmake_cmd.append(str(cmake_file.absolute()))
    
    try:
        run_command(cmake_cmd)
    except subprocess.CalledProcessError as e:
        print(f"CMake configuration failed: {e}")
        print(f"CMake output: {e.stdout}")
        print(f"CMake error: {e.stderr}")
        return False

    # Build the library
    try:
        build_cmd = ["cmake", "--build", str(build_dir), "--config", "Release", "-j"]
        run_command(build_cmd)
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        print(f"Build output: {e.stdout}")
        print(f"Build error: {e.stderr}")
        return False

    # Install the library 
    try:
        install_dir = Path("install")
        install_dir.mkdir(exist_ok=True)
        
        install_cmd = ["cmake", "--install", str(build_dir), "--prefix", str(install_dir.absolute())]
        run_command(install_cmd)
    except subprocess.CalledProcessError as e:
        print(f"Install failed: {e}")
        print(f"Install output: {e.stdout}")
        print(f"Install error: {e.stderr}")
        return False

    return True

def install_tpu_backend():
    """Install the TPU backend"""
    # Find the llama-cpp-python package
    try:
        import llama_cpp
        llama_cpp_dir = Path(llama_cpp.__file__).parent
        print(f"Found llama-cpp-python at {llama_cpp_dir}")
    except ImportError:
        print("Error: llama-cpp-python not found. Please install it first.")
        return False

    # Determine library name based on platform
    lib_name = "libllama_cpp_tpu_backend.so"
    if platform.system() == "Windows":
        lib_name = "llama_cpp_tpu_backend.dll"
    elif platform.system() == "Darwin":
        lib_name = "libllama_cpp_tpu_backend.dylib"

    # Check for built library in common locations
    possible_lib_paths = [
        Path("install/lib") / lib_name,
        Path("install") / lib_name,
        Path("build_tpu/lib") / lib_name,
        Path("build_tpu") / lib_name,
        Path("build_tpu/Release") / lib_name,
        Path("build_tpu/Debug") / lib_name
    ]

    lib_path = None
    for path in possible_lib_paths:
        if path.exists():
            lib_path = path
            break

    if not lib_path:
        print(f"Error: Could not find TPU backend library. Searched in:")
        for path in possible_lib_paths:
            print(f"  - {path}")
        return False

    # Create backends directory if it doesn't exist
    backends_dir = llama_cpp_dir / "backends"
    backends_dir.mkdir(exist_ok=True)

    # Copy the library
    shutil.copy2(lib_path, backends_dir / lib_name)
    print(f"Installed {lib_name} to {backends_dir}")

    # Create an __init__.py file in the backends directory if it doesn't exist
    init_file = backends_dir / "__init__.py"
    if not init_file.exists():
        with open(init_file, "w") as f:
            f.write("# TPU backend for llama-cpp-python\n")

    # Create a tpu_backend.py file to register the backend
    tpu_backend_file = backends_dir / "tpu_backend.py"
    backend_code = """# TPU backend for llama-cpp-python

import os
import ctypes
from pathlib import Path

# Load the TPU backend library
def load_tpu_backend():
    # Set environment variables for TPU
    os.environ['PJRT_DEVICE'] = 'TPU'  # For Cloud TPUs 
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Allow dynamic memory allocation

    # Get the path to the TPU backend library
    backend_dir = Path(__file__).parent

    # Determine the library name based on the platform
    import platform
    if platform.system() == "Windows":
        lib_name = "llama_cpp_tpu_backend.dll"
    elif platform.system() == "Darwin":
        lib_name = "libllama_cpp_tpu_backend.dylib"
    else:  # Linux
        lib_name = "libllama_cpp_tpu_backend.so"

    lib_path = backend_dir / lib_name

    if not lib_path.exists():
        raise RuntimeError(f"TPU backend library not found at {lib_path}")

    # Load the library
    try:
        lib = ctypes.CDLL(str(lib_path))
        # Register the TPU backend
        if hasattr(lib, 'ggml_backend_tpu_register'):
            lib.ggml_backend_tpu_register()
            print("TPU backend registered successfully")
            return lib
        else:
            print("TPU backend library does not have the required function")
            return None
    except Exception as e:
        print(f"Failed to load TPU backend: {e}")
        return None

# Auto-load the TPU backend when this module is imported
try:
    _tpu_lib = load_tpu_backend()
except Exception as e:
    print(f"Warning: Failed to load TPU backend: {e}")
    _tpu_lib = None

def is_available():
    \"\"\"Check if TPU backend is available\"\"\"
    return _tpu_lib is not None

def get_tpu_device_count():
    \"\"\"Get the number of TPU devices available\"\"\"
    if not is_available():
        return 0
    
    try:
        import torch_xla
        import torch_xla.runtime as xrt
        devices = xrt.devices()
        return len([d for d in devices if d.startswith('TPU')])
    except (ImportError, Exception) as e:
        print(f"Error getting TPU device count: {e}")
        return 0
"""
    with open(tpu_backend_file, "w") as f:
        f.write(backend_code)

    print(f"Created {tpu_backend_file}")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    missing_deps = []
    
    try:
        import torch
        print(f"Found PyTorch {torch.__version__}")
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import torch_xla # type: ignore
        print(f"Found PyTorch/XLA")
    except ImportError:
        missing_deps.append("torch_xla")
    
    try:
        import llama_cpp
        print(f"Found llama-cpp-python {llama_cpp.__version__}")
    except ImportError:
        missing_deps.append("llama-cpp-python")
    
    if missing_deps:
        print(f"Missing dependencies: {', '.join(missing_deps)}")
        print("Please install the missing dependencies and try again.")
        return False
    
    return True

def main():
    """Main function"""
    print("TPU Backend Setup for llama-cpp-python")
    print("=====================================")
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Build the TPU backend
    print("\nBuilding TPU backend...")
    if not build_tpu_backend():
        print("Failed to build TPU backend")
        return 1
    
    # Install the TPU backend
    print("\nInstalling TPU backend...")
    if not install_tpu_backend():
        print("Failed to install TPU backend")
        return 1
    
    print("\nTPU backend setup completed successfully!")
    print("You can now use the TPU backend with llama-cpp-python")
    print("Example usage:")
    print("```python")
    print("from llama_cpp import Llama")
    print("from llama_cpp.backends import tpu_backend")
    print("")
    print("# Check if TPU backend is available")
    print("if tpu_backend.is_available():")
    print("    # Create a Llama model with TPU backend")
    print("    model = Llama(model_path='path/to/model.gguf', n_gpu_layers=-1, backend='tpu')")
    print("    # Use the model")
    print("    response = model.generate('Hello, world!')")
    print("    print(response)")
    print("```")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())