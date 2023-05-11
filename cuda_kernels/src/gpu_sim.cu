#include "gpu_sim.cuh"

__global__ void clear_tree(int* buffer, int* buffer_size) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    printf("In clear tree!");
}

py::list walk_particles_gpu(py::array_t<int> initial_particles, py::array_t<int> boundary_particles, int number_of_timesteps, float bound_range, int max_tries) {

    // Create the list to hold the arrays
    py::list resultList;

    //// Create and append the NumPy arrays to the list
    //py::array_t<int> array1({ 2, 2 });
    //auto buffer1 = array1.request();
    //int* ptr1 = static_cast<int*>(buffer1.ptr);
    //ptr1[0] = 1;
    //ptr1[1] = 2;
    //ptr1[2] = 3;
    //ptr1[3] = 4;
    //resultList.append(array1);

    //py::array_t<float> array2({ 3, 3 });
    //auto buffer2 = array2.request();
    //float* ptr2 = static_cast<float*>(buffer2.ptr);
    //for (int i = 0; i < 9; i++) {
    //    ptr2[i] = static_cast<float>(i);
    //}
    //resultList.append(array2);

    // Return the list
    return resultList;

}