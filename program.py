from randomWalk import *
from randomWalkGPU import *
from postProcessing import *
import time


# ----------------------------------------- Program Start --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
initialSphere = getInitialSphereNumpy()
startTime = time.perf_counter()
# particles = randomWalkCPU(initialSphere)
particles, MLD = walkParticlesGPU(initialSphere)
MLD = calculateLinearDistanceGPU(particles)

print("Time to complete simulation (s): " + str(time.perf_counter() - startTime))
show3DVisualization and plotCellData(particles)

print("Calculating mean linear displacement...")
plotMLD(MLD)

shouldSaveResults and saveResults(MLD, particles)
