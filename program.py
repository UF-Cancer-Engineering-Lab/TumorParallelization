from randomWalk import *
from randomWalkGPU import *
from postProcessing import *
import time


# ----------------------------------------- Program Start --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
squaredRadius = sphereRadius**2
squaredCapillaryRadius = capillaryRadius**2

initialSphere = getInitialSphereNumpy()
startTime = time.perf_counter()
# particlesCPU = randomWalkCPU(initialSphere)
particles, MLD = walkParticlesGPU(initialSphere)
MLD = calculateLinearDistanceGPU(particles)

print("Time to complete simulation (s): " + str(time.perf_counter() - startTime))
plotCellData(particles)

print("Calculating mean linear displacement...")
plotMLD(MLD)

saveResults and saveMLD(MLD)
