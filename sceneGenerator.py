import numpy as np

from octTreeCPU import buildTreeCPU
from randomWalk import getInitialSphereNumpy
from scene import write_particles_to_scene


# def getVolumeEdgeTree(volume_voxels):
#     # Throw volume into an octtree to accelerate performance
#     tree = buildTreeCPU(
#         volume_voxels,
#         boundRange=(
#             np.float32(np.max(np.absolute(volume_voxels)) + 1) * 2.0
#             if len(volume_voxels)
#             else np.float32(1.0)
#         ),
#     )
#     volume_boundary = np.array([], dtype=np.int32)

#     # Remove any particle that has particles next to it on all sides
#     for particle in volume_voxels:
#         # Do a volume search with rad 1 on this particle. Number of particles in it should be < (3*3*3)=27 to add to boundary
#         range = 3.0
#         particle_bound_position = particle - [1, 1, 1]
#         # print(particle, particle_bound_position, range)
#         neighbors = tree.query(particle_bound_position, range)
#         neighbors = neighbors.reshape(len(neighbors) // 3, 3)
#         # print(len(neighbors))
#         # print(neighbors)
#         if len(neighbors) < 27:
#             volume_boundary = np.append(volume_boundary, particle)

#     return volume_boundary.reshape(len(volume_boundary) // 3, 3)


def getVolumeEdge(volume_voxels):
    edge_particles = []

    tree = buildTreeCPU(
        volume_voxels,
        boundRange=(
            np.float32(np.max(np.absolute(volume_voxels)) + 1) * 2.0
            if len(volume_voxels)
            else np.float32(1.0)
        ),
    )
    for particle in volume_voxels:
        neighbor_count = 0
        neighbor_limit = 20
        for x in range(-1, 2):
            for y in range(-1, 2):
                for z in range(-1, 2):
                    adjustedPosition = particle + [x, y, z]
                    if not tree.contains(adjustedPosition):
                        # neighbor_count += 1
                        neighbor_limit = -1
                        edge_particles.append(adjustedPosition)
    print(len(edge_particles), len(volume_voxels))
    return np.array(edge_particles, dtype=np.int32)


volume = getInitialSphereNumpy(sphereRadius=10, porosityFraction=0.0)
volume = getVolumeEdge(volume)
write_particles_to_scene("sphere.json", volume)
