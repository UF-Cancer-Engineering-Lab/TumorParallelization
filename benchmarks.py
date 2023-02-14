from randomWalk import *
from randomWalkGPU import *
from postProcessing import *
import time
import plotly.graph_objects as go

n = 5
sphereRadiusList = range(1, 15)
algorithms = [randomWalkCPU, randomWalkCPUOctTree, walkParticlesGPU]
timings = []

# Note, this disregards other parts of pipeline that were moved to the GPU, like calculateLinearDistanceGPU
for algorithmIndex, algorithm in enumerate(algorithms):
    timings.append([])
    for sphereRadius in sphereRadiusList:
        print(
            "Running "
            + str(algorithm.__name__)
            + " with n = "
            + str(n)
            + " and sphereRadius = "
            + str(sphereRadius)
        )
        initialSphere = getInitialSphereNumpy(sphereRadius=sphereRadius)
        startTime = time.perf_counter()
        particles = algorithm(initialSphere, n=n, sphereRadius=sphereRadius)
        endTime = time.perf_counter()
        timings[algorithmIndex].append(endTime - startTime)

print(timings)
fig = go.Figure()
fig.update_layout(
    title="Time to complete "
    + str(n)
    + " timesteps given different initial sphere radius.",
    xaxis_title="Sphere Radius (voxels)",
    yaxis_title="Time " + str(n) + " timesteps (s)",
)
for idx, algorithmData in enumerate(timings):
    fig.add_trace(
        go.Scatter(
            x=[*sphereRadiusList],
            y=algorithmData,
            mode="lines",
            name=str(algorithms[idx].__name__),
        )
    )
fig.show()
