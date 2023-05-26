#include <thread>
#include <vector>

#include "gpu_sim.cuh"

void print_gpu_tree_buffer(int *gpu_tree_buffer, unsigned int tree_buffer_size_nodes) {
    unsigned int buffer_size = tree_buffer_size_nodes * NODE_SIZE_BYTES;
    int *host_ptr = (int *)malloc(buffer_size);
    cudaMemcpy(host_ptr, gpu_tree_buffer, buffer_size, cudaMemcpyDeviceToHost);
    std::cout << buffer_size << std::endl;
    for (int i = 0; i < tree_buffer_size_nodes; i++) {
        std::cout << 8 * i << "     ";
        for (int j = 0; j < NODE_SIZE_INT; j++) {
            std::cout << host_ptr[i * NODE_SIZE_INT + j] << " ";
        }
        std::cout << "" << std::endl;
    }
    delete host_ptr;
}

void h_clear_tree(int *gpu_tree_buffer, int *used_tree_buffer_size, unsigned int tree_buffer_size_nodes, unsigned int default_used_size, cudaStream_t stream) {
    dim3 block_dim(THREADS_PER_BLOCK, 1, 1);
    dim3 grid_dim((tree_buffer_size_nodes / block_dim.x) + 1, 1, 1);

    clear_tree<<<grid_dim, block_dim, 0, stream>>>(gpu_tree_buffer, used_tree_buffer_size, tree_buffer_size_nodes, default_used_size);
}
__global__ void clear_tree(int *tree_buffer, int *used_tree_buffer_size, unsigned int tree_buffer_size_nodes, unsigned int default_used_size) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid == 0) {
        used_tree_buffer_size[0] = default_used_size;
    }

    if (tid < tree_buffer_size_nodes) {
        tree_buffer[tid * NODE_SIZE_INT + TREE_CHILD_OFFSET] = NO_PARTICLE_NO_CHILD;
        tree_buffer[tid * NODE_SIZE_INT + TREE_LOCK_OFFSET] = UNLOCKED;
    }
}

void h_build_tree(int *gpu_tree_buffer, int *used_tree_buffer_size, int *gpu_particles_buffer, curandState *rnd_state, unsigned int tree_buffer_size_nodes, int number_of_particles, int particle_type, float bound_range, int max_tries, bool random_walk, cudaStream_t stream) {
    dim3 block_dim(THREADS_PER_BLOCK, 1, 1);
    dim3 grid_dim((number_of_particles / block_dim.x) + 1, 1, 1);

    build_tree<<<grid_dim, block_dim, 0, stream>>>(gpu_tree_buffer, used_tree_buffer_size, gpu_particles_buffer, rnd_state, tree_buffer_size_nodes, number_of_particles, particle_type, bound_range, max_tries, random_walk);
}
__global__ void build_tree(int *gpu_tree_buffer, int *used_tree_buffer_size, const int *gpu_particles_buffer, curandState *rnd_state, unsigned int tree_buffer_size_nodes, int number_of_particles, int particle_type, float bound_range, int max_tries, bool random_walk) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < number_of_particles) {
        // State of particle
        int particle_buffer_pos = tid * PARTICLE_SIZE_INT;
        int original_particle_position[PARTICLE_SIZE_INT];
        int walked_particle_position[PARTICLE_SIZE_INT];
        original_particle_position[PARTICLE_X_OFFSET] = gpu_particles_buffer[particle_buffer_pos + PARTICLE_X_OFFSET];
        original_particle_position[PARTICLE_Y_OFFSET] = gpu_particles_buffer[particle_buffer_pos + PARTICLE_Y_OFFSET];
        original_particle_position[PARTICLE_Z_OFFSET] = gpu_particles_buffer[particle_buffer_pos + PARTICLE_Z_OFFSET];
        int tries_left = max_tries;
        curandState local_rnd_state = rnd_state[tid];

        while (tries_left > 0) {
            // Walk the particle (randomize position) (for now write original)
            randomize_particle_position(walked_particle_position, original_particle_position, &local_rnd_state, random_walk);

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
                        curr_bound_range *= 0.5;
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
                            tries_left = 0;

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

                                while (offset_existing == offset_new) {
                                    // Move curr_tree_pos to correct offset from prev iteration
                                    curr_tree_pos += offset_existing * NODE_SIZE_INT;

                                    // Determine octant to place particles at next subdivision
                                    offset_existing = get_next_octant(existing_particle_position, bound_start, curr_bound_range);
                                    offset_new = get_next_octant(walked_particle_position, bound_start, curr_bound_range);
                                    // printf("bound_start: %f %f %f \nwith bound_range %f", bound_start[0], bound_start[1], bound_start[2], curr_bound_range);
                                    // printf("\noffset_existing %d \n offset_new %d", offset_existing, offset_new);

                                    // Get available spot in buffer to place particles
                                    int subdivision_space_needed = 8 * NODE_SIZE_INT;
                                    int child_level_pos = atomicAdd(&used_tree_buffer_size[0], subdivision_space_needed);

                                    // Lock child, then release parent as non-leaf
                                    // atomicExch(&gpu_tree_buffer[child_level_pos + NODE_SIZE_INT * offset_existing + TREE_LOCK_OFFSET], tid);
                                    // atomicExch(&gpu_tree_buffer[child_level_pos + NODE_SIZE_INT * offset_new + TREE_LOCK_OFFSET], tid);
                                    // atomicExch(&gpu_tree_buffer[curr_tree_pos + TREE_CHILD_OFFSET], child_level_pos);
                                    // __threadfence();
                                    // atomicExch(&gpu_tree_buffer[curr_tree_pos + TREE_LOCK_OFFSET], NON_LEAF);
                                    gpu_tree_buffer[child_level_pos + NODE_SIZE_INT * offset_existing + TREE_LOCK_OFFSET] = tid;
                                    gpu_tree_buffer[child_level_pos + NODE_SIZE_INT * offset_new + TREE_LOCK_OFFSET] = tid;
                                    gpu_tree_buffer[curr_tree_pos + TREE_CHILD_OFFSET] = child_level_pos;
                                    __threadfence();
                                    gpu_tree_buffer[curr_tree_pos + TREE_LOCK_OFFSET] = NON_LEAF;

                                    // Subdivide bounds for next subdivision
                                    curr_tree_pos = child_level_pos;
                                    update_bound_start(bound_start, curr_bound_range, offset_existing);
                                    curr_bound_range *= 0.5;
                                }

                                // DEBUG
                                // if (true) {
                                //     gpu_tree_buffer[curr_tree_pos + TREE_LOCK_OFFSET] = UNLOCKED;
                                //     completed_insert_attempt = true;
                                //     break;
                                // }

                                // Write particle positions into tree buffer and unlock

                                // Existing
                                int existing_tree_pos = curr_tree_pos + offset_existing * NODE_SIZE_INT;
                                gpu_tree_buffer[existing_tree_pos + TREE_ID_OFFSET] = existing_particle_id;
                                gpu_tree_buffer[existing_tree_pos + TREE_X_OFFSET] = existing_particle_position[0];
                                gpu_tree_buffer[existing_tree_pos + TREE_Y_OFFSET] = existing_particle_position[1];
                                gpu_tree_buffer[existing_tree_pos + TREE_Z_OFFSET] = existing_particle_position[2];
                                gpu_tree_buffer[existing_tree_pos + TREE_TYPE_OFFSET] = existing_particle_type;
                                gpu_tree_buffer[existing_tree_pos + TREE_CHILD_OFFSET] = PARTICLE_NO_CHILD;

                                // New
                                int new_tree_pos = curr_tree_pos + offset_new * NODE_SIZE_INT;
                                gpu_tree_buffer[new_tree_pos + TREE_ID_OFFSET] = tid;
                                gpu_tree_buffer[new_tree_pos + TREE_X_OFFSET] = walked_particle_position[0];
                                gpu_tree_buffer[new_tree_pos + TREE_Y_OFFSET] = walked_particle_position[1];
                                gpu_tree_buffer[new_tree_pos + TREE_Z_OFFSET] = walked_particle_position[2];
                                gpu_tree_buffer[new_tree_pos + TREE_TYPE_OFFSET] = particle_type;
                                gpu_tree_buffer[new_tree_pos + TREE_CHILD_OFFSET] = PARTICLE_NO_CHILD;

                                // Release locks
                                __threadfence();
                                gpu_tree_buffer[existing_tree_pos + TREE_LOCK_OFFSET] = UNLOCKED;
                                gpu_tree_buffer[new_tree_pos + TREE_LOCK_OFFSET] = UNLOCKED;
                                tries_left = 0;
                            }
                        }
                        completed_insert_attempt = true;
                    }
                }
            }

            // Reset particle for the next iteration
            tries_left--;
        }

        rnd_state[tid] = local_rnd_state;
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
__device__ void randomize_particle_position(int walked_particle_position[3], int original_particle_position[3], curandState *local_rnd_state, bool should_random_walk) {
    walked_particle_position[PARTICLE_X_OFFSET] = original_particle_position[PARTICLE_X_OFFSET];
    walked_particle_position[PARTICLE_Y_OFFSET] = original_particle_position[PARTICLE_Y_OFFSET];
    walked_particle_position[PARTICLE_Z_OFFSET] = original_particle_position[PARTICLE_Z_OFFSET];
    if (should_random_walk) {
        int rnd_number = (int)ceilf(6.0f * curand_uniform(local_rnd_state));
        if (rnd_number == 1) {
            walked_particle_position[PARTICLE_X_OFFSET] += 1;
        } else if (rnd_number == 2) {
            walked_particle_position[PARTICLE_X_OFFSET] -= 1;
        } else if (rnd_number == 3) {
            walked_particle_position[PARTICLE_Y_OFFSET] += 1;
        } else if (rnd_number == 4) {
            walked_particle_position[PARTICLE_Y_OFFSET] -= 1;
        } else if (rnd_number == 5) {
            walked_particle_position[PARTICLE_Z_OFFSET] += 1;
        } else if (rnd_number == 6) {
            walked_particle_position[PARTICLE_Z_OFFSET] -= 1;
        }
    }
}
__global__ void rnd_setup_kernel(int seed, curandState *rnd_state) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    curand_init(seed, tid, 0, &rnd_state[tid]);
}

void h_read_tree(int *gpu_tree_buffer, int *gpu_particles_buffer, int used_tree_buffer_size, int tree_buffer_size_nodes, cudaStream_t stream) {
    int used_buffer_size_nodes = used_tree_buffer_size / NODE_SIZE_INT;

    dim3 block_dim(THREADS_PER_BLOCK, 1, 1);
    dim3 grid_dim((used_buffer_size_nodes / block_dim.x) + 1, 1, 1);

    read_tree<<<grid_dim, block_dim, 0, stream>>>(gpu_tree_buffer, gpu_particles_buffer, tree_buffer_size_nodes);
}
__global__ void read_tree(const int *gpu_tree_buffer, int *gpu_particles_buffer, int tree_buffer_size_nodes) {
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

__global__ void init_mld(float *mld_buffer, int number_of_timesteps) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < number_of_timesteps) {
        mld_buffer[tid] = 0.0f;
    }
}
__global__ void sum_mld(float *mld_buffer, const int *gpu_particles_buffer, const int *gpu_init_particles_buffer, int timestep, int particle_count) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float shared_mld[1];

    bool is_first_thread = threadIdx.x == 0;
    if (is_first_thread) {
        shared_mld[0] = 0.0f;
    }
    __syncthreads();

    if (tid < particle_count) {
        int buffer_pos = tid * PARTICLE_SIZE_INT;
        float delta_x = gpu_init_particles_buffer[buffer_pos + PARTICLE_X_OFFSET] - gpu_particles_buffer[buffer_pos + PARTICLE_X_OFFSET];
        float delta_y = gpu_init_particles_buffer[buffer_pos + PARTICLE_Y_OFFSET] - gpu_particles_buffer[buffer_pos + PARTICLE_Y_OFFSET];
        float delta_z = gpu_init_particles_buffer[buffer_pos + PARTICLE_Z_OFFSET] - gpu_particles_buffer[buffer_pos + PARTICLE_Z_OFFSET];
        float particle_mld = sqrtf((float)(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z));
        atomicAdd(&shared_mld[0], particle_mld);
    }

    // Write output to global mem
    __syncthreads();
    if (is_first_thread) {
        atomicAdd(&mld_buffer[timestep], shared_mld[0]);
    }
}
__global__ void divide_mld(float *mld_buffer, int number_of_timesteps, int particle_count) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < number_of_timesteps) {
        mld_buffer[tid] /= particle_count;
    }
}

void h_sum_mld(float *mld_buffer, int *gpu_particles_buffer, int *gpu_init_particles_buffer, int timestep, int particle_count, cudaStream_t stream) {
    dim3 block_dim(THREADS_PER_BLOCK, 1, 1);
    dim3 grid_dim((particle_count / block_dim.x) + 1, 1, 1);

    sum_mld<<<grid_dim, block_dim, 0, stream>>>(mld_buffer, gpu_particles_buffer, gpu_init_particles_buffer, timestep, particle_count);
}

py::tuple walk_particles_gpu(py::array_t<int> initial_particles, py::array_t<int> boundary_particles, int number_of_timesteps, float bound_range, int max_tries, bool random_walk, bool return_gpu_tree_buffer, int tree_buffer_size_nodes, int recording_interval) {
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
    int *gpu_init_particles_buffer;
    size_t gpu_particles_buffer_size = particle_count * 3 * sizeof(int);
    cudaMalloc(&gpu_particles_buffer, gpu_particles_buffer_size);
    cudaMalloc(&gpu_init_particles_buffer, gpu_particles_buffer_size);
    cudaMemcpy(gpu_particles_buffer, initial_particles_ptr, gpu_particles_buffer_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_init_particles_buffer, initial_particles_ptr, gpu_particles_buffer_size, cudaMemcpyHostToDevice);

    // Create numpy list to hold results
    std::vector<size_t> shape = {(number_of_timesteps / recording_interval) * particle_count * 3};
    py::array_t<int> result_array(shape);
    int *result_array_ptr = static_cast<int *>(result_array.request().ptr);

    std::vector<size_t> mld_shape = {(size_t)number_of_timesteps};
    py::array_t<float> mld_result_array(mld_shape);
    float *mld_result_array_ptr = static_cast<float *>(mld_result_array.request().ptr);

    // Create & init MLD buffer
    float *mld_buffer = nullptr;
    size_t mld_buffer_size = number_of_timesteps * sizeof(float);
    cudaMalloc(&mld_buffer, mld_buffer_size);
    dim3 mld_block_dim(THREADS_PER_BLOCK, 1, 1);
    dim3 mld_grid_dim((number_of_timesteps / mld_block_dim.x) + 1, 1, 1);
    init_mld<<<mld_grid_dim, mld_block_dim>>>(mld_buffer, number_of_timesteps);

    // Generate random state to sample from
    curandState *rnd_state;
    dim3 rnd_block_dim(THREADS_PER_BLOCK, 1, 1);
    dim3 rnd_grid_dim((particle_count / rnd_block_dim.x) + 1, 1, 1);

    cudaMalloc(&rnd_state, rnd_block_dim.x * rnd_grid_dim.x * sizeof(curandState));
    srand(time(0));
    int seed = rand();
    rnd_setup_kernel<<<rnd_grid_dim, rnd_block_dim>>>(seed, rnd_state);

    // Create static tree data
    size_t boundary_particle_count = boundary_particles.shape(0);
    int *boundary_particles_ptr = static_cast<int *>(boundary_particles.request().ptr);
    size_t boundary_particle_buffer_size = boundary_particle_count * 3 * sizeof(int);
    int *gpu_boundary_particles_buffer = nullptr;
    cudaMalloc(&gpu_boundary_particles_buffer, boundary_particle_buffer_size);
    cudaMemcpy(gpu_boundary_particles_buffer, boundary_particles_ptr, boundary_particle_buffer_size, cudaMemcpyHostToDevice);

    h_clear_tree(gpu_tree_buffer, used_tree_buffer_size, tree_buffer_size_nodes, NODE_SIZE_INT);
    h_build_tree(gpu_tree_buffer, used_tree_buffer_size, gpu_boundary_particles_buffer, rnd_state, tree_buffer_size_nodes, boundary_particle_count, BARRIER_CELL, bound_range, max_tries, false);

    // Copy the static data to its own buffer
    int *gpu_static_tree_buffer = nullptr;
    int static_tree_buffer_size = 0;
    cudaMemcpy(&static_tree_buffer_size, used_tree_buffer_size, sizeof(int), cudaMemcpyDeviceToHost);
    static_tree_buffer_size *= sizeof(int);
    cudaMalloc(&gpu_static_tree_buffer, gpu_tree_buffer_size);
    cudaMemcpy(gpu_static_tree_buffer, gpu_tree_buffer, gpu_tree_buffer_size, cudaMemcpyDeviceToDevice);
    cudaFree(gpu_boundary_particles_buffer);

    // Pinned memory for higher bandwidth/lower latency
    int *pinned_used_buffer_size = nullptr;
    int *pinned_static_tree_buffer_size_ints = nullptr;
    int *pinned_frame_result = nullptr;
    cudaError_t status = cudaHostAlloc(&pinned_used_buffer_size, sizeof(int), cudaHostAllocDefault);
    if (status != cudaSuccess)
        printf("Error allocating pinned host memory\n");

    status = cudaHostAlloc(&pinned_static_tree_buffer_size_ints, sizeof(int), cudaHostAllocDefault);
    if (status != cudaSuccess)
        printf("Error allocating pinned host memory\n");

    status = cudaHostAlloc(&pinned_frame_result, gpu_particles_buffer_size, cudaHostAllocDefault);
    if (status != cudaSuccess)
        printf("Error allocating pinned host memory\n");

    pinned_used_buffer_size[0] = static_tree_buffer_size;
    pinned_static_tree_buffer_size_ints[0] = static_tree_buffer_size / sizeof(int);

    // Streams to overlap data transfers with kernel executions
    cudaStream_t mem_stream;
    cudaStreamCreate(&mem_stream);
    cudaStream_t exec_stream;  // Also used for device to device to allow parallel copies
    cudaStreamCreate(&exec_stream);

    cudaDeviceSynchronize();

    // Flag and thread for multithreaded memcpy of pinned to paged memory
    int *offset_result_array_ptr = nullptr;
    std::thread copy_pinned_to_paged_thread;
    auto copy_pinned_to_paged = [&offset_result_array_ptr, &pinned_frame_result, &gpu_particles_buffer_size]() {
        memcpy(offset_result_array_ptr, pinned_frame_result, gpu_particles_buffer_size);
    };

    // Run Kernels for each timestep
    for (int timestep = 0; timestep < number_of_timesteps; timestep++) {
        // Setup static tree
        cudaMemcpyAsync(used_tree_buffer_size, pinned_static_tree_buffer_size_ints, sizeof(int), cudaMemcpyHostToDevice, mem_stream);
        cudaMemcpyAsync(gpu_tree_buffer, gpu_static_tree_buffer, pinned_used_buffer_size[0] * sizeof(int), cudaMemcpyDeviceToDevice, exec_stream);

        // Perform MLD while data is transferred
        h_sum_mld(mld_buffer, gpu_particles_buffer, gpu_init_particles_buffer, timestep, particle_count, exec_stream);

        // Wait for static data to be loaded before performing simulation
        cudaStreamSynchronize(mem_stream);

        // Perform walk and read data to particles buffer
        h_build_tree(gpu_tree_buffer, used_tree_buffer_size, gpu_particles_buffer, rnd_state, tree_buffer_size_nodes, particle_count, CANCER_CELL, bound_range, max_tries, random_walk, exec_stream);
        cudaMemcpyAsync(pinned_used_buffer_size, used_tree_buffer_size, sizeof(int), cudaMemcpyDeviceToHost, exec_stream);
        h_read_tree(gpu_tree_buffer, gpu_particles_buffer, pinned_used_buffer_size[0], tree_buffer_size_nodes, exec_stream);

        // Move data from gpu to host
        offset_result_array_ptr = result_array_ptr + ((timestep / recording_interval) * particle_count * 3);

        cudaStreamSynchronize(exec_stream);  // Wait for read_tree

        // Recording interval to reduce memory footprint of result
        if (timestep % recording_interval == 0) {
            if (timestep != 0) {
                copy_pinned_to_paged_thread.join();
            }

            cudaMemcpyAsync(pinned_frame_result, gpu_particles_buffer, gpu_particles_buffer_size, cudaMemcpyDeviceToHost, mem_stream);

            // Create thread to copy result
            copy_pinned_to_paged_thread = std::thread(copy_pinned_to_paged);
        }
    }

    // Make sure all streams/operations are done
    cudaDeviceSynchronize();
    copy_pinned_to_paged_thread.join();

    // Finalize mld calculations
    divide_mld<<<mld_grid_dim, mld_block_dim>>>(mld_buffer, number_of_timesteps, particle_count);
    cudaMemcpy(mld_result_array_ptr, mld_buffer, mld_buffer_size, cudaMemcpyDeviceToHost);

    std::cout << "Finished Running kernels;";

    // Change windowing for python numpy array
    result_array.resize({(size_t)(number_of_timesteps / recording_interval), (size_t)particle_count, (size_t)3});

    // Only used for testing
    if (return_gpu_tree_buffer) {
        std::vector<size_t> shape = {(size_t)tree_buffer_size_nodes * NODE_SIZE_INT};
        py::array_t<int> gpu_tree_result(shape);
        int *gpu_tree_result_ptr = static_cast<int *>(gpu_tree_result.request().ptr);
        cudaMemcpy(gpu_tree_result_ptr, gpu_tree_buffer, gpu_tree_buffer_size, cudaMemcpyDeviceToHost);
        return py::make_tuple(mld_result_array, gpu_tree_result);
    }

    // Cleanup
    cudaStreamDestroy(mem_stream);
    cudaStreamDestroy(exec_stream);
    cudaFree(gpu_tree_buffer);
    cudaFree(used_tree_buffer_size);
    cudaFree(gpu_particles_buffer);
    cudaFree(mld_buffer);
    cudaFree(rnd_state);
    cudaFree(gpu_static_tree_buffer);
    cudaFreeHost(pinned_used_buffer_size);
    cudaFreeHost(pinned_static_tree_buffer_size_ints);
    cudaFreeHost(pinned_frame_result);

    return py::make_tuple(mld_result_array, result_array);
}