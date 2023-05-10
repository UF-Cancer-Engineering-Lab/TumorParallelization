#pragma once

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

__device__ const int NODE_SIZE = 8;

__global__ void vector_add(float *a, float *b, float *c);
__global__ void clear_tree(int* buffer, int* buffer_size);