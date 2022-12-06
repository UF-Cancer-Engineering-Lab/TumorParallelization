from randomWalk import *
import time

# ----------------------------------------- Program Start --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
startTime = time.perf_counter()
squaredRadius = sphereRadius**2
squaredCapillaryRadius = capillaryRadius**2

particlesDF = randomWalkCPUOctTree()

print("Time to complete simulation (s): " + str(time.perf_counter() - startTime))
print("Simulation complete. Calculating mean squared displacement")

plotCellData(particlesDF)

# calculateMSD(particles)
