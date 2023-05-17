#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdio.h>

#include <iostream>
namespace py = pybind11;

// The buffer is organized as follows:
//       int1 is the ID of the particle
//       int2 is the x of the particle
//       int3 is the y of the particle
//       int4 is the z of the particle
//       int5 is the childNode of this particle
//           -1 if there is no child yet and no particle
//           -2 if there is a particle but no children
//           #  pointing to child index in buffer if a non-leaf node of tree
//       int6 is the lock state of the node
//           lock will be -1 if unlocked
//           lock == particleID if the node is locked
//       int7 is for cell type
//           0 is immovable cell
//           1 is movable cancer cell
//       int8 is reserved for now

const int NODE_SIZE_INT = 8;
const int NODE_SIZE_BYTES = NODE_SIZE_INT * 4;

const int TREE_ID_OFFSET = 0;
const int TREE_X_OFFSET = 1;
const int TREE_Y_OFFSET = 2;
const int TREE_Z_OFFSET = 3;
const int TREE_CHILD_OFFSET = 4;
const int TREE_LOCK_OFFSET = 5;
const int TREE_TYPE_OFFSET = 6;
const int TREE_RESERVED_OFFSET = 7;

const int NO_CHILD_NO_PARTICLE = -1;
const int PARTICLE_NO_CHILD = -2;
const int UNLOCKED = -1;
const int BARRIER_CELL = 0;
const int CANCER_CELL = 1;

const int PARTICLE_SIZE_INT = 3;  // X, Y, Z
const int PARTICLE_X_OFFSET = 0;
const int PARTICLE_Y_OFFSET = 1;
const int PARTICLE_Z_OFFSET = 2;

// Device Kernels
__global__ void clear_tree(int* tree_buffer, int* used_tree_buffer_size, unsigned int tree_buffer_size_nodes);
__global__ void read_tree(int* gpu_tree_buffer, int* gpu_particles_buffer, unsigned int tree_buffer_size_nodes);

// Host Functions
void print_gpu_tree_buffer(int* gpu_tree_buffer, unsigned int tree_buffer_size_nodes);
void h_clear_tree(int* gpu_tree_buffer, int* used_tree_buffer_size, unsigned int tree_buffer_size_nodes, bool async);
void h_read_tree(int* gpu_tree_buffer, int* gpu_particles_buffer, unsigned int tree_buffer_size_nodes, bool async);
py::array_t<int> walk_particles_gpu(py::array_t<int> initial_particles, py::array_t<int> boundary_particles, int number_of_timesteps, float bound_range, int max_tries, bool random_walk, bool return_gpu_tree_buffer, int tree_buffer_size_nodes);

int haroon_print() {
    std::cout << "Hello ffrom c++" << std::endl;
    return 0;
}

PYBIND11_MODULE(cuda_kernels, m) {
    m.def("walk_particles_gpu", &walk_particles_gpu, "Perform the walk particles gpu sim",
          py::arg("initial_particles"),
          py::arg("boundary_particles"),
          py::arg("number_of_timesteps"),
          py::arg("bound_range"),
          py::arg("max_tries") = 6,
          py::arg("random_walk") = true,
          py::arg("return_gpu_tree_buffer") = false,
          py::arg("tree_buffer_size_nodes") = 10000000);
    m.def("haroon_print", &haroon_print, "print aiogujn");
}