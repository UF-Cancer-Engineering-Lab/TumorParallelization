# Python code for 2D random walk.
# from cmath import sqrt
from cmath import sqrt

# from logging import _Level
import logging
import math
import numpy as np
import random
import pandas
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from config import *
from numba import jit
from octTreeCPU import *
from util import particlesToDF

# -----------------------------------------functions for walk algorithms: --------------------------------------------------------------------------------------------------------------------------------------------------------------------
def getInitialSphere(
    particlesNumber=particlesNumber,
    porosityFraction=porosityFraction,
    sphereRadius=sphereRadius,
):
    return particlesToDF(getInitialSphereNumpy())


def getInitialSphereNumpy(
    particlesNumber=particlesNumber,
    porosityFraction=porosityFraction,
    sphereRadius=sphereRadius,
):

    # child variables
    vacancies = round(particlesNumber * porosityFraction)
    totalPositions = particlesNumber + vacancies

    # creating full array
    x = 0
    y = 0
    z = 0
    initialSphere = []
    # From the panda library, a Data frame is just a c++ maps that are in the form of arrays

    # This while loop is getting all the x, y, z integer values insde of 3D sphere (in all 8 3D quadrants). Each of these combination of x,y,z values,
    # of which there are ~840, will be the vector components of a vector whose magnitude is an integer, conforming to a radius of a smaller sphere than the original one
    # set in the limit
    # initialSphere.index returns total rows in the initialSphere dataFrame
    while (
        len(initialSphere) < totalPositions
    ):  # len(initialSphere.index) is the number of total rows in the DataFrame
        # print("Loading initial sphere")
        sphereRadius = sphereRadius
        for z in range(-sphereRadius, sphereRadius):
            xMax = int(
                math.sqrt(sphereRadius**2 - z**2 - 0**2)
            )  # initially z = -6 #these xMax values appy for
            # print(z)
            for x in range(-xMax, xMax):
                yMax = int(math.sqrt(sphereRadius**2 - z**2 - x**2))
                # print(x,yMax)
                for y in range(-yMax, yMax):
                    # initialSphere = initialSphere.append(
                    #     , ignore_index=True
                    # )
                    initialSphere.append([np.int32(x), np.int32(y), np.int32(z)])

    totalPositions = len(initialSphere)
    vacancies = round(totalPositions * porosityFraction)
    # print("Initial sphere complete")

    # this randomizes the areas where the particles can and cannot go in the sphere
    for i in range(0, vacancies):
        val = random.randint(0, len(initialSphere) - 1)
        initialSphere.pop(val)

    return np.array(initialSphere, dtype=np.int32)


def randomWalkCPU(
    initialSphere,
    capillaryRadius,
    n=n,
    maxTries=maxTries,
    sphereRadius=sphereRadius,
):

    # Constraints for cell movement
    squaredRadius = sphereRadius**2
    squaredCapillaryRadius = capillaryRadius**2

    # creating two array for containing x and y coordinate
    # of size equals to the number of size and filled up with 0's
    particles = [
        initialSphere
    ]  # particles is now a list containing the first timestep result

    # random walking for n timesteps and add to each timestep result to particles
    for i in range(1, n + 1):
        particles.append(particles[-1].copy())

        particleN = -1
        for particle in particles[-1]:
            particleN += 1
            for j in range(maxTries):
                walkedParticle = particle.copy()
                val = random.randint(1, 6)

                if val == 1:
                    walkedParticle[0] += 1
                elif val == 2:
                    walkedParticle[0] -= 1
                elif val == 3:
                    walkedParticle[1] += 1
                elif val == 4:
                    walkedParticle[1] -= 1
                elif val == 5:
                    walkedParticle[2] += 1
                else:
                    walkedParticle[2] -= 1

                x_2 = walkedParticle[0] ** 2
                y_2 = walkedParticle[1] ** 2
                z_2 = walkedParticle[2] ** 2

                # comparing this values to the previous dataFrame (particles[i-1]) means we don't want the particle to move to a past position, nor do we want it to move to the x,y,z coordinate of a current position, last part is we want to squared distance to be within squared capillary radius
                if (
                    not any(
                        np.equal(
                            particles[i],
                            [
                                walkedParticle[0],
                                walkedParticle[1],
                                walkedParticle[2],
                            ],
                        ).all(1)
                    )
                ) and (  # No particle in this timestep shares the same position
                    (x_2 + y_2 + z_2) < squaredRadius
                    or (x_2 + z_2) < squaredCapillaryRadius
                    or (y_2 + z_2) < squaredCapillaryRadius
                ):  # If inside the capillary radius
                    particles[i][particleN] = walkedParticle
                    break

        (i % 100 == 0) and print("Time steps elapsed: " + str(i))
    return particles


def randomWalkCPUOctTree(
    initialSphere,
    capillaryRadius,
    n=n,
    maxTries=maxTries,
    sphereRadius=sphereRadius,
):

    # Constraints for cell movement
    squaredRadius = sphereRadius**2
    squaredCapillaryRadius = capillaryRadius**2

    # creating two array for containing x and y coordinate
    # of size equals to the number of size and filled up with 0's
    particles = [
        initialSphere
    ]  # particles is now a list containing the first timestep result

    # random walking for n timesteps and add to each timestep result to particles
    for i in range(1, n + 1):
        boundRange = (sphereRadius + 1 + i) * 2
        tree = buildTreeCPU(boundRange=boundRange)
        particles.append(particles[i - 1].copy())

        # Now walk the particles (the insert function returns if successful or not)
        particleN = -1
        for particle in particles[i]:
            particleN += 1
            for j in range(maxTries):
                walkedParticle = particle.copy()
                val = random.randint(1, 6)
                if val == 1:
                    walkedParticle[0] += 1
                elif val == 2:
                    walkedParticle[0] -= 1
                elif val == 3:
                    walkedParticle[1] += 1
                elif val == 4:
                    walkedParticle[1] -= 1
                elif val == 5:
                    walkedParticle[2] += 1
                else:
                    walkedParticle[2] -= 1
                x_2 = walkedParticle[0] ** 2
                y_2 = walkedParticle[1] ** 2
                z_2 = walkedParticle[2] ** 2

                # If able to insert into tree (unique value for this timestep) and inside the capillary
                if tree.insert(walkedParticle) and (
                    (x_2 + y_2 + z_2) < squaredRadius
                    or (x_2 + z_2) < squaredCapillaryRadius
                    or (y_2 + z_2) < squaredCapillaryRadius
                ):
                    particles[i][particleN] = walkedParticle
                    break

        (i % 100 == 0) and print("Time steps elapsed: " + str(i))
    return particles
