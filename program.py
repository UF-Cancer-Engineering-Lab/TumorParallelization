from randomWalk import *
from randomWalkGPU import *
from postProcessing import *
from scene import *
import time


# ----------------------------------------- Program Start --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
scene = load_scene(scene_file_name)
initialSphere = getInitialSphereNumpy()
print("Made initial sphere, starting simulation...")
startTime = time.perf_counter()
# particles = randomWalkCPU(initialSphere)
MLD, particles = walkParticlesGPU(initialSphere, scene)
print("Time to complete simulation (s): " + str(time.perf_counter() - startTime))
show3DVisualization and plotCellData(particles, scene)

print("Calculating mean linear displacement...")
showMLDVisualization and plotMLD(MLD)

shouldSaveResults and saveResults(MLD, particles)
