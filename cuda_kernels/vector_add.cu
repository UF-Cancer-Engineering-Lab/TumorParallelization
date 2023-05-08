extern "C" __global__ void vector_add(float *a, float *b, float *c)
{   
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    c[tid] = a[tid] + b[tid];
}