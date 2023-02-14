from randomWalk import *
from randomWalkGPU import *
from postProcessing import *
import time

n_list = [1, 10, 100, 1000]
algorithms = [randomWalkCPU, randomWalkCPUOctTree, walkParticlesGPU]
timings = []
sphereRadius = 10

# Note, this disregards other parts of pipeline that were moved to the GPU, like calculateLinearDistanceGPU
for algorithmIndex, algorithm in enumerate(algorithms):
    timings.append([])
    for n in n_list:
        print("Running " + str(algorithm.__name__) + " with n = " + str(n))
        initialSphere = getInitialSphereNumpy(sphereRadius=sphereRadius)
        startTime = time.perf_counter()
        particles = algorithm(initialSphere, n=n, sphereRadius=sphereRadius)
        endTime = time.perf_counter()
        timings[algorithmIndex].append(endTime - startTime)

print(timings)
