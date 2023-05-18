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

    clear_tree<<<grid_dim, block_dim>>>(gpu_tree_buffer, used_tree_buffer_size, tree_buffer_size_nodes);

    if (!async) {
        cudaDeviceSynchronize();
    }
}
__global__ void clear_tree(int *tree_buffer, int *used_tree_buffer_size, unsigned int tree_buffer_size_nodes) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid == 0) {
        used_tree_buffer_size[0] = NODE_SIZE_INT;
    }

    if (tid < tree_buffer_size_nodes) {
        tree_buffer[tid * NODE_SIZE_INT + TREE_CHILD_OFFSET] = NO_PARTICLE_NO_CHILD;
        tree_buffer[tid * NODE_SIZE_INT + TREE_LOCK_OFFSET] = UNLOCKED;
    }
}

void h_build_tree(int *gpu_tree_buffer, int *used_tree_buffer_size, int *gpu_particles_buffer, unsigned int tree_buffer_size_nodes, int number_of_particles, int particle_type, float bound_range, int max_tries, bool random_walk, bool async) {
    dim3 block_dim(32, 1, 1);
    dim3 grid_dim((number_of_particles / block_dim.x) + 1, 1, 1);

    build_tree<<<grid_dim, block_dim>>>(gpu_tree_buffer, used_tree_buffer_size, gpu_particles_buffer, tree_buffer_size_nodes, number_of_particles, particle_type, bound_range, max_tries, random_walk);

    if (!async) {
        cudaDeviceSynchronize();
    }
}
__global__ void build_tree(int *gpu_tree_buffer, int *used_tree_buffer_size, int *gpu_particles_buffer, unsigned int tree_buffer_size_nodes, int number_of_particles, int particle_type, float bound_range, int max_tries, bool random_walk) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < number_of_particles) {
        // State of particle
        bool inserted_particle = false;
        int particle_buffer_pos = tid * PARTICLE_SIZE_INT;
        int original_particle_position[PARTICLE_SIZE_INT];
        int walked_particle_position[PARTICLE_SIZE_INT];
        original_particle_position[PARTICLE_X_OFFSET] = gpu_particles_buffer[particle_buffer_pos + PARTICLE_X_OFFSET];
        original_particle_position[PARTICLE_Y_OFFSET] = gpu_particles_buffer[particle_buffer_pos + PARTICLE_Y_OFFSET];
        original_particle_position[PARTICLE_Z_OFFSET] = gpu_particles_buffer[particle_buffer_pos + PARTICLE_Z_OFFSET];
        int tries_left = max_tries;

        // TODO: Add random
        while (!inserted_particle && tries_left > 0) {
            // Walk the particle (randomize position) (for now write original)
            walked_particle_position[PARTICLE_X_OFFSET] = original_particle_position[PARTICLE_X_OFFSET];
            walked_particle_position[PARTICLE_Y_OFFSET] = original_particle_position[PARTICLE_Y_OFFSET];
            walked_particle_position[PARTICLE_Z_OFFSET] = original_particle_position[PARTICLE_Z_OFFSET];

            // Insert particle into tree
            {
                // Insertion State
                int curr_tree_pos = 0;
                float curr_bound_range = bound_range;
                float bound_start[3];
                bound_start[0] = bound_start[1] = bound_start[2] = -0.5f * bound_range;
                bool completed_insert_attempt = false;

                while (!completed_insert_attempt) {
                    // Travel deeper if non-leaf
                    if (gpu_tree_buffer[curr_tree_pos + TREE_LOCK_OFFSET] == NON_LEAF) {
                        int octant_offset = get_next_octant(walked_particle_position, bound_start, curr_bound_range);
                        curr_tree_pos = gpu_tree_buffer[curr_tree_pos + TREE_CHILD_OFFSET] + octant_offset * NODE_SIZE_INT;
                        update_bound_start(bound_start, curr_bound_range, octant_offset);
                        curr_bound_range /= 2.0;
                    }
                    // If leaf get lock and insert here
                    if (UNLOCKED == atomicCAS(&gpu_tree_buffer[curr_tree_pos + TREE_LOCK_OFFSET], UNLOCKED, tid)) {
                        int curr_node_child = gpu_tree_buffer[curr_tree_pos + TREE_CHILD_OFFSET];
                        // if (tid == 2) {
                        //     printf("\nInserting particle at %d", curr_tree_pos);
                        //     printf("\nNode information: %d %d %d %d %d %d %d", gpu_tree_buffer[curr_tree_pos + 0], gpu_tree_buffer[curr_tree_pos + 1], gpu_tree_buffer[curr_tree_pos + 2], gpu_tree_buffer[curr_tree_pos + 3], gpu_tree_buffer[curr_tree_pos + 4], gpu_tree_buffer[curr_tree_pos + 5], gpu_tree_buffer[curr_tree_pos + 6]);
                        // }

                        if (NO_PARTICLE_NO_CHILD == curr_node_child) {
                            gpu_tree_buffer[curr_tree_pos + TREE_ID_OFFSET] = tid;
                            gpu_tree_buffer[curr_tree_pos + TREE_X_OFFSET] = walked_particle_position[PARTICLE_X_OFFSET];
                            gpu_tree_buffer[curr_tree_pos + TREE_Y_OFFSET] = walked_particle_position[PARTICLE_Y_OFFSET];
                            gpu_tree_buffer[curr_tree_pos + TREE_Z_OFFSET] = walked_particle_position[PARTICLE_Z_OFFSET];
                            gpu_tree_buffer[curr_tree_pos + TREE_TYPE_OFFSET] = particle_type;
                            gpu_tree_buffer[curr_tree_pos + TREE_CHILD_OFFSET] = PARTICLE_NO_CHILD;
                            inserted_particle = true;

                            // Ensure writes are visible to other threads and free lock
                            __threadfence();
                            gpu_tree_buffer[curr_tree_pos + TREE_LOCK_OFFSET] = UNLOCKED;
                        }

                        // Need to move existing particle and new particle into subtree
                        else {
                            // Load relevant data from node
                            int existing_particle_position[3];
                            int existing_particle_id = gpu_tree_buffer[curr_tree_pos + TREE_ID_OFFSET];
                            existing_particle_position[0] = gpu_tree_buffer[curr_tree_pos + TREE_X_OFFSET];
                            existing_particle_position[1] = gpu_tree_buffer[curr_tree_pos + TREE_Y_OFFSET];
                            existing_particle_position[2] = gpu_tree_buffer[curr_tree_pos + TREE_Z_OFFSET];
                            int existing_particle_type = gpu_tree_buffer[curr_tree_pos + TREE_TYPE_OFFSET];

                            // Prevent infinite recursion if particles the same
                            if (walked_particle_position[0] == existing_particle_position[0] &&
                                walked_particle_position[1] == existing_particle_position[1] &&
                                walked_particle_position[2] == existing_particle_position[2]) {
                                gpu_tree_buffer[curr_tree_pos + TREE_LOCK_OFFSET] = UNLOCKED;
                            } else {
                                int offset_existing = 0;
                                int offset_new = 0;
                                int subtree_index = curr_tree_pos;

                                while (offset_existing == offset_new) {
                                    // Determine octant to place particles at next subdivision
                                    offset_existing = get_next_octant(existing_particle_position, bound_start, curr_bound_range);
                                    offset_new = get_next_octant(walked_particle_position, bound_start, curr_bound_range);
                                    // printf("bound_start: %f %f %f \nwith bound_range %f", bound_start[0], bound_start[1], bound_start[2], curr_bound_range);
                                    // printf("\noffset_existing %d \n offset_new %d", offset_existing, offset_new);

                                    // Get available spot in buffer to place particles
                                    int subdivision_space_needed = 8 * NODE_SIZE_INT;
                                    int child_level_pos = atomicAdd(&used_tree_buffer_size[0], subdivision_space_needed);

                                    // Lock child, then release parent as non-leaf
                                    // atomicCAS(&gpu_tree_buffer[child_level_pos + NODE_SIZE_INT * offset_existing + TREE_LOCK_OFFSET], UNLOCKED, tid);
                                    // atomicCAS(&gpu_tree_buffer[child_level_pos + NODE_SIZE_INT * offset_new + TREE_LOCK_OFFSET], UNLOCKED, tid);
                                    // gpu_tree_buffer[curr_tree_pos + TREE_CHILD_OFFSET] = child_level_pos;
                                    // gpu_tree_buffer[curr_tree_pos + TREE_LOCK_OFFSET] = NON_LEAF;

                                    gpu_tree_buffer[subtree_index + TREE_CHILD_OFFSET] = child_level_pos;
                                    if (subtree_index != curr_tree_pos) {
                                        gpu_tree_buffer[subtree_index + TREE_LOCK_OFFSET] = NON_LEAF;
                                    }

                                    // Subdivide bounds for next subdivision
                                    // curr_tree_pos = child_level_pos;
                                    // update_bound_start(bound_start, curr_bound_range, offset_existing);
                                    // curr_bound_range /= 2.0;

                                    subtree_index = child_level_pos;
                                    if (offset_existing == offset_new) {
                                        subtree_index += NODE_SIZE_INT * offset_existing;
                                    }
                                    update_bound_start(bound_start, curr_bound_range, offset_existing);
                                    curr_bound_range /= 2.0;
                                }

                                // DEBUG
                                // if (true) {
                                //     gpu_tree_buffer[curr_tree_pos + TREE_LOCK_OFFSET] = UNLOCKED;
                                //     completed_insert_attempt = true;
                                //     break;
                                // }

                                // Write particle positions into tree buffer and unlock

                                // Existing
                                // int existing_tree_pos = curr_tree_pos + offset_existing * NODE_SIZE_INT;
                                int existing_tree_pos = subtree_index + offset_existing * NODE_SIZE_INT;
                                gpu_tree_buffer[existing_tree_pos + TREE_ID_OFFSET] = existing_particle_id;
                                gpu_tree_buffer[existing_tree_pos + TREE_X_OFFSET] = existing_particle_position[0];
                                gpu_tree_buffer[existing_tree_pos + TREE_Y_OFFSET] = existing_particle_position[1];
                                gpu_tree_buffer[existing_tree_pos + TREE_Z_OFFSET] = existing_particle_position[2];
                                gpu_tree_buffer[existing_tree_pos + TREE_TYPE_OFFSET] = existing_particle_type;
                                gpu_tree_buffer[existing_tree_pos + TREE_CHILD_OFFSET] = PARTICLE_NO_CHILD;

                                // New
                                // int new_tree_pos = curr_tree_pos + offset_new * NODE_SIZE_INT;
                                int new_tree_pos = subtree_index + offset_new * NODE_SIZE_INT;
                                gpu_tree_buffer[new_tree_pos + TREE_ID_OFFSET] = tid;
                                gpu_tree_buffer[new_tree_pos + TREE_X_OFFSET] = walked_particle_position[0];
                                gpu_tree_buffer[new_tree_pos + TREE_Y_OFFSET] = walked_particle_position[1];
                                gpu_tree_buffer[new_tree_pos + TREE_Z_OFFSET] = walked_particle_position[2];
                                gpu_tree_buffer[new_tree_pos + TREE_TYPE_OFFSET] = particle_type;
                                gpu_tree_buffer[new_tree_pos + TREE_CHILD_OFFSET] = PARTICLE_NO_CHILD;

                                // Release locks
                                // __threadfence();
                                // gpu_tree_buffer[existing_tree_pos + TREE_LOCK_OFFSET] = UNLOCKED;
                                // gpu_tree_buffer[new_tree_pos + TREE_LOCK_OFFSET] = UNLOCKED;
                                // inserted_particle = true;

                                __threadfence();
                                gpu_tree_buffer[curr_tree_pos + TREE_LOCK_OFFSET] = NON_LEAF;
                                inserted_particle = true;
                            }
                        }
                        completed_insert_attempt = true;
                    }
                }
            }

            // Reset particle for the next iteration
            tries_left--;
        }
    }
}
__device__ int get_next_octant(int particle_position[3], float bound_start[3], float bound_range) {
    // Determine the center of boundary
    float center_X = bound_start[0] + (bound_range / 2.0f);
    float center_Y = bound_start[1] + (bound_range / 2.0f);
    float center_Z = bound_start[2] + (bound_range / 2.0f);

    // Convention followed found here: https://commons.wikimedia.org/wiki/Category:Octant_%28geometry%29
    if (particle_position[PARTICLE_X_OFFSET] >= center_X) {
        if (particle_position[PARTICLE_Y_OFFSET] >= center_Y) {
            if (particle_position[PARTICLE_Z_OFFSET] >= center_Z) {
                return 0;
            } else {
                return 4;
            }
        } else {
            if (particle_position[PARTICLE_Z_OFFSET] >= center_Z) {
                return 3;
            } else {
                return 7;
            }
        }
    } else {
        if (particle_position[PARTICLE_Y_OFFSET] >= center_Y) {
            if (particle_position[PARTICLE_Z_OFFSET] >= center_Z) {
                return 1;
            } else {
                return 5;
            }
        } else {
            if (particle_position[PARTICLE_Z_OFFSET] >= center_Z) {
                return 2;
            } else {
                return 6;
            }
        }
    }
}
__device__ void update_bound_start(float bound_start[3], float bound_range, int offset) {
    float center_X = bound_start[0] + (bound_range / 2.0f);
    float center_Y = bound_start[1] + (bound_range / 2.0f);
    float center_Z = bound_start[2] + (bound_range / 2.0f);

    // Convention followed found here: https://commons.wikimedia.org/wiki/Category:Octant_%28geometry%29
    if (offset == 0) {
        bound_start[0] = center_X;
        bound_start[1] = center_Y;
        bound_start[2] = center_Z;
    } else if (offset == 1) {
        bound_start[0] = bound_start[0];
        bound_start[1] = center_Y;
        bound_start[2] = center_Z;
    } else if (offset == 2) {
        bound_start[0] = bound_start[0];
        bound_start[1] = bound_start[1];
        bound_start[2] = center_Z;
    } else if (offset == 3) {
        bound_start[0] = center_X;
        bound_start[1] = bound_start[1];
        bound_start[2] = center_Z;
    } else if (offset == 4) {
        bound_start[0] = center_X;
        bound_start[1] = center_Y;
        bound_start[2] = bound_start[2];
    } else if (offset == 5) {
        bound_start[0] = bound_start[0];
        bound_start[1] = center_Y;
        bound_start[2] = bound_start[2];
    } else if (offset == 6) {
        bound_start[0] = bound_start[0];
        bound_start[1] = bound_start[1];
        bound_start[2] = bound_start[2];
    } else if (offset == 7) {
        bound_start[0] = center_X;
        bound_start[1] = bound_start[1];
        bound_start[2] = bound_start[2];
    }
}

void h_read_tree(int *gpu_tree_buffer, int *gpu_particles_buffer, int *used_tree_buffer_size, int tree_buffer_size_nodes, bool async) {
    int h_used_tree_buffer_size = 0;
    cudaMemcpy(&h_used_tree_buffer_size, used_tree_buffer_size, sizeof(int), cudaMemcpyDeviceToHost);
    int used_buffer_size_nodes = h_used_tree_buffer_size / NODE_SIZE_INT;

    dim3 block_dim(32, 1, 1);
    dim3 grid_dim((used_buffer_size_nodes / block_dim.x) + 1, 1, 1);

    read_tree<<<grid_dim, block_dim>>>(gpu_tree_buffer, gpu_particles_buffer, tree_buffer_size_nodes);

    if (!async) {
        cudaDeviceSynchronize();
    }
}
__global__ void read_tree(int *gpu_tree_buffer, int *gpu_particles_buffer, int tree_buffer_size_nodes) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int tree_buffer_pos = tid * NODE_SIZE_INT;

    if (tree_buffer_pos < tree_buffer_size_nodes) {
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

    // Create single size array to track used space in tree buffer
    int *used_tree_buffer_size = nullptr;
    cudaMalloc(&used_tree_buffer_size, sizeof(int));

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
        h_clear_tree(gpu_tree_buffer, used_tree_buffer_size, tree_buffer_size_nodes, true);
        h_build_tree(gpu_tree_buffer, used_tree_buffer_size, gpu_particles_buffer, tree_buffer_size_nodes, particle_count, CANCER_CELL, bound_range, max_tries, random_walk, true);
        h_read_tree(gpu_tree_buffer, gpu_particles_buffer, used_tree_buffer_size, tree_buffer_size_nodes, true);

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