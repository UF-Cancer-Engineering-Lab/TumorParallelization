import math
from numba import cuda
import numpy as np
from config import *


# @cuda.jit
# def INIT_MLD_BUFFER(MLD_buffer):
#     tid = cuda.grid(1)
#     if tid < len(MLD_buffer):
#         MLD_buffer[tid] = 0.0


# @cuda.jit
# def MLD_CUDA_SUM(MLD_buffer, initialSphere, latestParticles, timestep):

#     # Theoretically speed up by setting up shared memory
#     isFirstThread = cuda.threadIdx.x == 0
#     shared = cuda.shared.array(1, np.float32)
#     if isFirstThread:
#         shared[0] = np.float32(0.0)
#     cuda.syncthreads()

#     particleID = cuda.grid(1)
#     if particleID < latestParticles[0].shape[0]:
#         x = latestParticles[particleID][0] - initialSphere[particleID][0]
#         y = latestParticles[particleID][1] - initialSphere[particleID][1]
#         z = latestParticles[particleID][2] - initialSphere[particleID][2]
#         LD = math.sqrt((x**2 + y**2 + z**2))
#         cuda.atomic.add(shared, 0, LD)

#     cuda.syncthreads()
#     if isFirstThread:
#         cuda.atomic.add(MLD_buffer, timestep, shared[0])


# @cuda.jit
# def MLD_CUDA_DIVIDE_ALL(MLD_buffer, numParticles, numEpochs):
#     epoch = cuda.grid(1)
#     if epoch < numEpochs:
#         MLD_buffer[epoch] /= numParticles


# ----------------------------------------- Calculate LSD --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@cuda.jit
def INIT_MLD_BUFFER(MLD_buffer):
    tid = cuda.grid(1)
    if tid < len(MLD_buffer):
        MLD_buffer[tid] = 0.0


@cuda.jit
def MLD_CUDA_SUM(MLD_buffer, particles, timestep):

    # Theoretically speed up by setting up shared memory
    isFirstThread = cuda.threadIdx.x == 0
    shared = cuda.shared.array(1, np.float32)
    if isFirstThread:
        shared[0] = np.float32(0.0)
    cuda.syncthreads()

    particleID = cuda.grid(1)
    if particleID < particles[0].shape[0]:
        x = particles[timestep][particleID][0] - particles[0][particleID][0]
        y = particles[timestep][particleID][1] - particles[0][particleID][1]
        z = particles[timestep][particleID][2] - particles[0][particleID][2]
        LD = math.sqrt((x**2 + y**2 + z**2))
        cuda.atomic.add(shared, 0, LD)

    cuda.syncthreads()
    if isFirstThread:
        cuda.atomic.add(MLD_buffer, timestep, shared[0])


@cuda.jit
def MLD_CUDA_DIVIDE_ALL(MLD_buffer, numParticles, numEpochs):

    epoch = cuda.grid(1)
    if epoch < numEpochs:
        MLD_buffer[epoch] /= numParticles


def calculateLinearDistanceGPU(particles):
    numTimeSteps = len(particles)
    numParticles = np.shape(particles[0])[0]
    MLD_buffer = cuda.device_array(numTimeSteps, dtype=np.float32)

    nthreadsX = 32
    nblocksXSum = (numParticles // nthreadsX) + 1
    nblocksXInit = nblocksXDivide = (numTimeSteps // nthreadsX) + 1

    particlesGPU = cuda.to_device(np.array(particles, dtype=np.int32))

    INIT_MLD_BUFFER[nblocksXInit, nthreadsX](MLD_buffer)

    for timestep in range(0, len(particles)):
        MLD_CUDA_SUM[nblocksXSum, nthreadsX](MLD_buffer, particlesGPU, timestep)

    MLD_CUDA_DIVIDE_ALL[nblocksXDivide, nthreadsX](
        MLD_buffer, numParticles, numTimeSteps
    )
    return MLD_buffer.copy_to_host().tolist()
