#include "./gpu_octtree.cuh"

__global__ void clear_tree(int* buffer, int* buffer_size) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    printf("In clear tree!");
}