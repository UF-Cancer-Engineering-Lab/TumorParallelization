from randomWalk import *
from randomWalkGPU import *
from postProcessing import *
from scene import *
import time


# ----------------------------------------- Program Start --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
scene = load_scene(scene_file_name)
initialSphere = getInitialSphereNumpy()
startTime = time.perf_counter()
# particles = randomWalkCPU(initialSphere)
particles = walkParticlesGPU(initialSphere, scene)
MLD = calculateLinearDistanceGPU(particles)

print("Time to complete simulation (s): " + str(time.perf_counter() - startTime))
show3DVisualization and plotCellData(particles)

print("Calculating mean linear displacement...")
plotMLD(MLD)

shouldSaveResults and saveResults(MLD, particles)
