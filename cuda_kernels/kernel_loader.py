import numpy as np
import cupy as cp
import os

def get_cuda_functions():
    kernel_dir = "./cuda_kernels"
    cuda_functions = {}
    for filename in os.listdir(kernel_dir):
        if filename.endswith(".cu"):
            file = open("./cuda_kernels/" + filename)
            file_code = file.read()
            func_name = filename.removesuffix(".cu")
            cuda_functions[func_name] = cp.RawKernel(file_code, func_name)
    return cuda_functions

cuda_kernels = get_cuda_functions()

x1 = cp.arange(25, dtype=cp.float32).reshape(5, 5)
x2 = cp.arange(25, dtype=cp.float32).reshape(5, 5)
y = cp.zeros((5, 5), dtype=cp.float32)
cuda_kernels["vector_add"]((5,), (5,), (x1, x2, y))
print(y)