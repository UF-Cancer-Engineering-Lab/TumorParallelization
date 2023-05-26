Tumor Paralleization research Git

# SETUP INSTRUCTIONS

0. Clone repository<br/>
   `git clone git@github.com:UF-Cancer-Engineering-Lab/TumorParallelization.git --recursive`<br/>
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

6. Go to config.py and change parameters as desired

7. Run program.py

# Running Tests

Run tests.py `python tests.py`

# Making executable

(Windows only right now...)<br/>

- Run make_executable.bat
- Copy scene and config folder into `dist` folder
- Copy dist folder wherever desired and run program.exe

# Making changes to CUDA Code

- Go to cuda_kernels folder
- Go to src
- Make changes as desired
- Run `compile_cuda_kernels.bat` or `compile_cuda_kernels.sh`
- Run program.py
