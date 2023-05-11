cmake_minimum_required(VERSION 3.12)

if (NOT EXISTS "${CMAKE_SOURCE_DIR}/pybind11")
    execute_process(COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_SOURCE_DIR} git clone git@github.com:pybind/pybind11.git)
else()
    execute_process(COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_SOURCE_DIR}/pybind11 git pull)
endif()

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
set_target_properties(cuda_kernels PROPERTIES CUDA_ARCHITECTURES "50;72")

# Link
target_link_libraries(cuda_kernels PRIVATE pybind11::module)

# Post Build
add_custom_command(TARGET cuda_kernels POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:cuda_kernels> ${CMAKE_SOURCE_DIR}/../.venv/Lib/site-packages)