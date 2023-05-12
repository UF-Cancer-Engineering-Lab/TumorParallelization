Tumor Paralleization research Git

# SETUP INSTRUCTIONS
0. Clone repository<br/>
  `git clone https://github.com/UF-Cancer-Engineering-Lab/TumorParallelization/tree/replace-numba-with-cuda --recursive`<br/>
  `cd ./TumorParallelization`
  
1. Create virutal environment
  `python -m venv .venv`
  
2. Activate virtual environment <br/>
  Windows: `.\.venv\Scripts\activate.bat`<br/>
  Mac/Linux: `source ./.venv/bin/activate`
  
3. Install Requirements
`pip install -r requirements.txt`

4. Install NVIDIA's [CUDA TOOLKIT](https://developer.nvidia.com/cuda-downloads)

5. Build Cuda Kernels<br/>
`compile_cuda_kernels.bat` (Windows)<br/>
`chmod +x compile_cuda_kernels.sh && ./compile_cuda_kernels.sh` (Linux)

5. Go to config.py and change parameters as desired

6. Run program.py
