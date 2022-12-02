from randomWalk import *
import time

# ----------------------------------------- Program Start --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
startTime = time.perf_counter()
squaredRadius = sphereRadius**2
squaredCapillaryRadius = capillaryRadius**2

particles = randomWalkCPU()

print("Time to complete simulation (s): " + str(time.perf_counter() - startTime))
print("Simulation complete. Calculating mean squared displacement")

plotCellData(particles)

# calculateMSD(particles)
