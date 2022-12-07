from randomWalk import *
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
startTime = time.perf_counter()
squaredRadius = sphereRadius**2
squaredCapillaryRadius = capillaryRadius**2

initialSphere = getInitialSphere()
particles = randomWalkCPUOctTree(initialSphere.to_numpy(dtype=np.int32))
particlesDataFrames = particlesToDF(particles)


print("Time to complete simulation (s): " + str(time.perf_counter() - startTime))
print("Simulation complete. Calculating mean squared displacement")

plotCellData(particlesDataFrames)

# calculateMSD(particles)
