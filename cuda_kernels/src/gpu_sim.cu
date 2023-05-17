#include <vector>

#include "gpu_sim.cuh"

void print_gpu_tree_buffer(int *gpu_tree_buffer, unsigned int tree_buffer_size_nodes) {
    unsigned int buffer_size = tree_buffer_size_nodes * NODE_SIZE_BYTES;
    int *host_ptr = (int *)malloc(buffer_size);
    cudaMemcpy(host_ptr, gpu_tree_buffer, buffer_size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < tree_buffer_size_nodes; i++) {
        for (int j = 0; j < NODE_SIZE_INT; j++) {
            std::cout << host_ptr[i * NODE_SIZE_INT + j] << " ";
        }
        std::cout << buffer_size << std::endl;
    }
    delete host_ptr;
}

void h_clear_tree(int *gpu_tree_buffer, int *used_tree_buffer_size, unsigned int tree_buffer_size_nodes, bool async) {
    dim3 block_dim(32, 1, 1);
    dim3 grid_dim((tree_buffer_size_nodes / block_dim.x) + 1, 1, 1);

    clear_tree<<<grid_dim, block_dim>>>(gpu_tree_buffer, nullptr, tree_buffer_size_nodes);

    if (!async) {
        cudaDeviceSynchronize();
    }
}
__global__ void clear_tree(int *tree_buffer, int *used_tree_buffer_size, unsigned int tree_buffer_size_nodes) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < tree_buffer_size_nodes) {
        tree_buffer[tid * NODE_SIZE_INT + TREE_CHILD_OFFSET] = NO_CHILD_NO_PARTICLE;
        tree_buffer[tid * NODE_SIZE_INT + TREE_LOCK_OFFSET] = UNLOCKED;
    }
}

void h_read_tree(int *gpu_tree_buffer, int *gpu_particles_buffer, unsigned int tree_buffer_size_nodes, bool async) {
    dim3 block_dim(32, 1, 1);
    dim3 grid_dim((tree_buffer_size_nodes / block_dim.x) + 1, 1, 1);

    read_tree<<<grid_dim, block_dim>>>(gpu_tree_buffer, gpu_particles_buffer, tree_buffer_size_nodes);

    if (!async) {
        cudaDeviceSynchronize();
    }
}
__global__ void read_tree(int *gpu_tree_buffer, int *gpu_particles_buffer, unsigned int tree_buffer_size_nodes) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int tree_buffer_pos = tid * NODE_SIZE_INT;

    if (tid < tree_buffer_size_nodes) {
        // For each leaf node, write the updated particle position back to particle list
        if (gpu_tree_buffer[tree_buffer_pos + TREE_CHILD_OFFSET] == PARTICLE_NO_CHILD && gpu_tree_buffer[tree_buffer_pos + TREE_TYPE_OFFSET] == CANCER_CELL) {
            int particle_id = gpu_tree_buffer[tree_buffer_pos + TREE_ID_OFFSET];
            gpu_particles_buffer[particle_id * PARTICLE_SIZE_INT + PARTICLE_X_OFFSET] = gpu_tree_buffer[tree_buffer_pos + TREE_X_OFFSET];
            gpu_particles_buffer[particle_id * PARTICLE_SIZE_INT + PARTICLE_Y_OFFSET] = gpu_tree_buffer[tree_buffer_pos + TREE_Y_OFFSET];
            gpu_particles_buffer[particle_id * PARTICLE_SIZE_INT + PARTICLE_Z_OFFSET] = gpu_tree_buffer[tree_buffer_pos + TREE_Z_OFFSET];
        }
    }
}

py::array_t<int> walk_particles_gpu(py::array_t<int> initial_particles, py::array_t<int> boundary_particles, int number_of_timesteps, float bound_range, int max_tries, bool random_walk, bool return_gpu_tree_buffer, int tree_buffer_size_nodes) {
    // Create gpu tree buffer
    int *gpu_tree_buffer = nullptr;
    size_t gpu_tree_buffer_size = tree_buffer_size_nodes * NODE_SIZE_BYTES;
    cudaMalloc(&gpu_tree_buffer, gpu_tree_buffer_size);

    // Send particle data to the gpu
    size_t particle_count = initial_particles.shape(0);
    int *initial_particles_ptr = static_cast<int *>(initial_particles.request().ptr);
    int *gpu_particles_buffer;
    size_t gpu_particles_buffer_size = particle_count * 3 * sizeof(int);
    cudaMalloc(&gpu_particles_buffer, gpu_particles_buffer_size);
    cudaMemcpy(gpu_particles_buffer, initial_particles_ptr, gpu_particles_buffer_size, cudaMemcpyHostToDevice);

    // Create numpy list to hold result
    std::vector<size_t> shape = {number_of_timesteps * particle_count * 3};
    py::array_t<int> result_array(shape);
    int *result_array_ptr = static_cast<int *>(result_array.request().ptr);

    // Run Kernels for each timestep
    for (int timestep = 0; timestep < number_of_timesteps; timestep++) {
        h_clear_tree(gpu_tree_buffer, nullptr, tree_buffer_size_nodes, true);

        if (random_walk) {
            // Build tree
        }

        h_read_tree(gpu_tree_buffer, gpu_particles_buffer, tree_buffer_size_nodes, true);

        cudaDeviceSynchronize();

        // Move data from gpu to host
        int *offset_result_array_ptr = result_array_ptr + (timestep * particle_count * 3);
        cudaMemcpy(offset_result_array_ptr, gpu_particles_buffer, gpu_particles_buffer_size, cudaMemcpyDeviceToHost);
    }

    // Change windowing for python numpy array
    result_array.resize({(size_t)number_of_timesteps, (size_t)particle_count, (size_t)3});

    // Only used for testing
    if (return_gpu_tree_buffer) {
        std::vector<size_t> shape = {(size_t)tree_buffer_size_nodes * NODE_SIZE_INT};
        py::array_t<int> gpu_tree_result(shape);
        int *gpu_tree_result_ptr = static_cast<int *>(gpu_tree_result.request().ptr);
        cudaMemcpy(gpu_tree_result_ptr, gpu_tree_buffer, gpu_tree_buffer_size, cudaMemcpyDeviceToHost);
        return gpu_tree_result;
    }

    // Cleanup
    cudaFree(gpu_tree_buffer);
    cudaFree(gpu_particles_buffer);

    return result_array;
}