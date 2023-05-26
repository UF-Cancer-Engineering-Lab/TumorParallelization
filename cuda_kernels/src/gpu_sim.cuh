#pragma once
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdio.h>
#include <stdlib.h>

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
//           lock will be -2 if non-leaf (better performance and less mem check than if lock only did locking)
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

const int NO_PARTICLE_NO_CHILD = -1;
const int PARTICLE_NO_CHILD = -2;
const int NON_LEAF = -2;
const int UNLOCKED = -1;
const int BARRIER_CELL = 0;
const int CANCER_CELL = 1;

const int PARTICLE_SIZE_INT = 3;  // X, Y, Z
const int PARTICLE_X_OFFSET = 0;
const int PARTICLE_Y_OFFSET = 1;
const int PARTICLE_Z_OFFSET = 2;

const int THREADS_PER_BLOCK = 1024;

// Device Kernels
__device__ void randomize_particle_position(int walked_particle_position[3], int original_particle_position[3], curandState* local_rnd_state, bool should_random_walk);
__device__ int get_next_octant(int particle_position[3], float bound_start[3], float bound_range);
__device__ void update_bound_start(float bound_start[3], float bound_range, int offset);
__global__ void rnd_setup_kernel(int seed, curandState* state);
__global__ void clear_tree(int* tree_buffer, int* used_tree_buffer_size, unsigned int tree_buffer_size_nodes, unsigned int default_used_size);
__global__ void build_tree(int* gpu_tree_buffer, int* used_tree_buffer_size, const int* gpu_particles_buffer, curandState* rnd_state, unsigned int tree_buffer_size_nodes, int number_of_particles, int particle_type, float bound_range, int max_tries, bool random_walk);
__global__ void read_tree(const int* gpu_tree_buffer, int* gpu_particles_buffer, int tree_buffer_size_nodes);
__global__ void init_mld(float* mld_buffer, int number_of_timesteps);
__global__ void sum_mld(float* mld_buffer, const int* gpu_particles_buffer, const int* gpu_init_particles_buffer, int timestep, int particle_count);
__global__ void divide_mld(float* mld_buffer, int number_of_timesteps, int particle_count);

// Host Functions
void print_gpu_tree_buffer(int* gpu_tree_buffer, unsigned int tree_buffer_size_nodes);
void h_clear_tree(int* gpu_tree_buffer, int* used_tree_buffer_size, unsigned int tree_buffer_size_nodes, unsigned int default_used_size, cudaStream_t stream = NULL);
void h_build_tree(int* gpu_tree_buffer, int* used_tree_buffer_size, int* gpu_particles_buffer, curandState* rnd_state, unsigned int tree_buffer_size_nodes, int number_of_particles, int particle_type, float bound_range, int max_tries, bool random_walk, cudaStream_t stream = NULL);
void h_read_tree(int* gpu_tree_buffer, int* gpu_particles_buffer, int used_tree_buffer_size, int tree_buffer_size_nodes, cudaStream_t stream = NULL);
void h_sum_mld(float* mld_buffer, int* gpu_particles_buffer, int* gpu_init_particles_buffer, int timestep, int particle_count, cudaStream_t stream = NULL);
py::tuple walk_particles_gpu(py::array_t<int> initial_particles, py::array_t<int> boundary_particles, int number_of_timesteps, float bound_range, int max_tries, bool random_walk, bool return_gpu_tree_buffer, int tree_buffer_size_nodes);

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
}