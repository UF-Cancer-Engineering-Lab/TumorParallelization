cmake_minimum_required(VERSION 3.12)

if (NOT EXISTS "${CMAKE_SOURCE_DIR}/pybind11/.gitignore")
    message( FATAL_ERROR "Please make sure pybind11 is cloned as a submodule. Run \"git submodule update --init\"")
endif()

set(CMAKE_CXX_STANDARD 11)
project(cuda_kernels LANGUAGES CXX CUDA)

# PyBind
set(PYBIND11_CPP_STANDARD -std=c++11)
add_subdirectory(pybind11)

# Cuda Code
file(GLOB_RECURSE SRC
    "src/*.cu"
    "src/*.cuh"
    "src/*.cpp"
    "src/*.h"
)
PYBIND11_ADD_MODULE(cuda_kernels ${SRC})

# Configure
set_target_properties(cuda_kernels PROPERTIES CUDA_ARCHITECTURES "72")

# Link
target_link_libraries(cuda_kernels PRIVATE pybind11::module)

# Post Build
add_custom_command(TARGET cuda_kernels POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:cuda_kernels> ${CMAKE_SOURCE_DIR})