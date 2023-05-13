#include "gpu_sim.cuh"
#include <vector>

__global__ void clear_tree(int *buffer, int *used_buffer_size, unsigned int num_buffer_nodes)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < num_buffer_nodes)
    {
        buffer[tid * NODE_SIZE_INT + CHILD_OFFSET] = NO_CHILD_NO_PARTICLE;
        buffer[tid * NODE_SIZE_INT + LOCK_OFFSET] = UNLOCKED;
    }
}

void h_clear_tree(int *buffer, int *used_buffer_size, unsigned int num_buffer_nodes, bool async)
{
    dim3 block_dim(32, 1, 1);
    dim3 grid_dim((num_buffer_nodes / block_dim.x) + 1, 1, 1);

    clear_tree<<<grid_dim, block_dim>>>(buffer, nullptr, num_buffer_nodes);

    if (!async)
    {
        cudaDeviceSynchronize();
    }
}

py::array_t<int> walk_particles_gpu(py::array_t<int> initial_particles, py::array_t<int> boundary_particles, int number_of_timesteps, float bound_range, int max_tries)
{
    // Create gpu buffers
    const unsigned int desired_node_count = 9;
    int *gpu_buffer = nullptr;
    size_t gpu_buffer_size = desired_node_count * NODE_SIZE_BYTES;
    cudaMalloc(&gpu_buffer, gpu_buffer_size);

    // Run Kernels
    h_clear_tree(gpu_buffer, nullptr, desired_node_count, true);

    cudaDeviceSynchronize();

    // Move data from gpu to host
    std::vector<size_t> shape = {desired_node_count * NODE_SIZE_INT};
    py::array_t<int> result_array(shape);
    int *host_ptr = static_cast<int *>(result_array.request().ptr);
    cudaMemcpy(host_ptr, gpu_buffer, gpu_buffer_size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(gpu_buffer);

    printf("\n==============Finished Simulation==============\n");
    return result_array;
}