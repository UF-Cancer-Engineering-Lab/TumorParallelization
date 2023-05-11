#pragma once
#include <stdio.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
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

__global__ void clear_tree(int* buffer, int* buffer_size);
py::list walk_particles_gpu(py::array_t<int> initial_particles, py::array_t<int> boundary_particles, int number_of_timesteps, float bound_range, int max_tries=6);

int haroon_print() {
    std::cout << "Hello ffrom c++" << std::endl;
    return 0;
}

PYBIND11_MODULE(cuda_kernels, m) {
    m.def("walk_particles_gpu", &walk_particles_gpu, "Perform the walk particles gpu sim");
    m.def("haroon_print", &haroon_print, "print aiogujn");
}