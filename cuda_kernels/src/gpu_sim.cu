#include <vector>

#include "gpu_sim.cuh"

void print_gpu_tree_buffer(int *gpu_tree_buffer, unsigned int num_buffer_nodes) {
    unsigned int buffer_size = num_buffer_nodes * NODE_SIZE_BYTES;
    int *host_ptr = (int *)malloc(buffer_size);
    cudaMemcpy(host_ptr, gpu_tree_buffer, buffer_size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < num_buffer_nodes; i++) {
        for (int j = 0; j < NODE_SIZE_INT; j++) {
            std::cout << host_ptr[i * NODE_SIZE_INT + j] << " ";
        }
        std::cout << buffer_size << std::endl;
    }
    delete host_ptr;
}

void h_clear_tree(int *gpu_tree_buffer, int *used_buffer_size, unsigned int num_buffer_nodes, bool async) {
    dim3 block_dim(32, 1, 1);
    dim3 grid_dim((num_buffer_nodes / block_dim.x) + 1, 1, 1);

    clear_tree<<<grid_dim, block_dim>>>(gpu_tree_buffer, nullptr, num_buffer_nodes);

    if (!async) {
        cudaDeviceSynchronize();
    }
}
__global__ void clear_tree(int *tree_buffer, int *used_buffer_size, unsigned int num_buffer_nodes) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < num_buffer_nodes) {
        tree_buffer[tid * NODE_SIZE_INT + CHILD_OFFSET] = NO_CHILD_NO_PARTICLE;
        tree_buffer[tid * NODE_SIZE_INT + LOCK_OFFSET] = UNLOCKED;
    }
}

py::array_t<int> walk_particles_gpu(py::array_t<int> initial_particles, py::array_t<int> boundary_particles, int number_of_timesteps, float bound_range, int max_tries) {
    // Create gpu buffers
    const unsigned int desired_node_count = 9;
    int *gpu_tree_buffer = nullptr;
    size_t gpu_buffer_size = desired_node_count * NODE_SIZE_BYTES;
    cudaMalloc(&gpu_tree_buffer, gpu_buffer_size);

    // Send particle data to the gpu
    size_t particle_count = initial_particles.shape(0);
    int *initial_particles_ptr = static_cast<int *>(initial_particles.request().ptr);
    int *gpu_particles_buffer;
    size_t gpu_particles_buffer_size = particle_count * 3 * sizeof(int);
    cudaMalloc(&gpu_particles_buffer, gpu_particles_buffer_size);
    cudaMemcpy(gpu_particles_buffer, initial_particles_ptr, gpu_particles_buffer_size, cudaMemcpyHostToDevice);

    std::vector<size_t> shape = {number_of_timesteps * particle_count * 3};
    py::array_t<int> result_array(shape);
    int *result_array_ptr = static_cast<int *>(result_array.request().ptr);

    // Run Kernels
    for (int timestep = 0; timestep < number_of_timesteps; timestep++) {
        h_clear_tree(gpu_tree_buffer, nullptr, desired_node_count, true);

        cudaDeviceSynchronize();

        // Move data from gpu to host
        int *offset_result_array_ptr = result_array_ptr + (timestep * particle_count * 3);
        cudaMemcpy(offset_result_array_ptr, gpu_particles_buffer, gpu_particles_buffer_size, cudaMemcpyDeviceToHost);
    }

    print_gpu_tree_buffer(gpu_tree_buffer, desired_node_count);
    result_array.resize({(size_t)number_of_timesteps, (size_t)particle_count, (size_t)3});

    // Cleanup
    cudaFree(gpu_tree_buffer);
    cudaFree(gpu_particles_buffer);

    printf("\n==============Finished Simulation==============\n");
    return result_array;
}