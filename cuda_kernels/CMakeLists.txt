cmake_minimum_required(VERSION 2.9)

if (NOT EXISTS "${CMAKE_SOURCE_DIR}/pybind11")
    execute_process(COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_SOURCE_DIR} git clone git@github.com:pybind/pybind11.git)
else()
    execute_process(COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_SOURCE_DIR}/pybind11 git pull)
endif()

project(cuda_kernels)
file(GLOB_RECURSE SRC
    "src/*.h"
    "src/*.cpp"
    "src/*.cu"
    "src/*.cuh"
)
add_subdirectory(pybind11)
PYBIND11_ADD_MODULE(cuda_kernels ${SRC})

add_custom_command(TARGET cuda_kernels POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:cuda_kernels> ${CMAKE_SOURCE_DIR}/../.venv/Lib/site-packages)