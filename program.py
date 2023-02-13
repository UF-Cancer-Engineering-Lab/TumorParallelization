from randomWalk import *
from randomWalkGPU import *
from postProcessing import *
import time


def particlesToDF(particles):
    # Convert the numpy data back into a dataframe
    particlesDataFrames = []
    for timeStepData in particles:
        particlesDataFrames.append(
            pandas.DataFrame(
                {
                    "x": timeStepData[:, 0],
                    "y": timeStepData[:, 1],
                    "z": timeStepData[:, 2],
                },
                dtype=np.int32,
            )
        )
    return particlesDataFrames


# ----------------------------------------- Program Start --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
squaredRadius = sphereRadius**2
squaredCapillaryRadius = capillaryRadius**2

initialSphere = getInitialSphere()
startTime = time.perf_counter()
# particlesCPU = randomWalkCPU(initialSphere.to_numpy(dtype=np.int32))
particles, MLD = walkParticlesGPU(initialSphere.to_numpy(dtype=np.int32))
MLD = calculateLinearDistanceGPU(particles)
# particlesDataFrames = particlesToDF(particles)

print("Time to complete simulation (s): " + str(time.perf_counter() - startTime))
# plotCellData(particlesDataFrames)

print("Calculating mean linear displacement...")
plotMLD(MLD)

saveResults and saveMLD(MLD)
