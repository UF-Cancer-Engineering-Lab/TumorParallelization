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

# -----------------------------------------functions for walk algorithms: --------------------------------------------------------------------------------------------------------------------------------------------------------------------
def getInitialSphere(
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
    initialSphere = pandas.DataFrame(columns=["x", "y", "z"])
    # From the panda library, a Data frame is just a c++ maps that are in the form of arrays

    # This while loop is getting all the x, y, z integer values insde of 3D sphere (in all 8 3D quadrants). Each of these combination of x,y,z values,
    # of which there are ~840, will be the vector components of a vector whose magnitude is an integer, conforming to a radius of a smaller sphere than the original one
    # set in the limit

    # initialSphere.index returns total rows in the initialSphere dataFrame
    while (
        len(initialSphere.index) < totalPositions
    ):  # len(initialSphere.index) is the number of total rows in the DataFrame
        print("Loading initial sphere")
        initialSphere = pandas.DataFrame(columns=["x", "y", "z"], dtype=np.int32)
        sphereRadius = sphereRadius + 1
        for z in range(-sphereRadius, sphereRadius + 1):  # initially (-6,6)
            xMax = int(
                math.sqrt(sphereRadius**2 - z**2 - 0**2)
            )  # initially z = -6 #these xMax values appy for
            # print(z)
            for x in range(-xMax, xMax + 1):
                yMax = int(math.sqrt(sphereRadius**2 - z**2 - x**2))
                # print(x,yMax)
                for y in range(-yMax, yMax + 1):
                    # initialSphere = initialSphere.append(
                    #     , ignore_index=True
                    # )
                    initialSphere.loc[len(initialSphere.index)] = [x, y, z]

    totalPositions = len(initialSphere.index)
    vacancies = round(totalPositions * porosityFraction)
    print("Initial sphere complete")

    # this randomizes the areas where the particles can and cannot go in the sphere
    for i in range(0, vacancies):
        val = random.randint(0, len(initialSphere.index) - 1)
        initialSphere = initialSphere.drop(labels=val, axis=0).reset_index(drop=True)
        # ^^^ .reset_index method just recalculates the indices so that the gap in indices isn't there anymore

    return initialSphere


def randomWalkCPU(
    initialSphere,
    n=n,
    maxTries=maxTries,
    sphereRadius=sphereRadius,
    capillaryRadius=capillaryRadius,
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

        print("Time steps elapsed: " + str(i))
    return particles


def randomWalkCPUOctTree(
    initialSphere,
    n=n,
    maxTries=maxTries,
    sphereRadius=sphereRadius,
    capillaryRadius=capillaryRadius,
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

        print("Time steps elapsed: " + str(i))
    return particles


# -----------------------------------------plotting stuff: --------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plotCellData(particlesDF, frame=-1):
    # pylab.title("Random Walk ($n = " + str(n) + "$ steps)")
    # Gather the position of each cell and distance to origin
    frame = -1
    x = particlesDF[frame]["x"]
    y = particlesDF[frame]["y"]
    z = particlesDF[frame]["z"]
    color = []

    for (xVal, yVal, zVal) in zip(x, y, z):
        color.append(
            abs(sqrt(xVal**2 + yVal**2 + zVal**2))
        )  # color based on distance to origin
    color = pandas.Series(color, copy=False)

    # Draw a scatter plot of cell positions
    # With color scale based on distance moved from origin for each cell
    fig = go.Figure(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            marker=go.scatter3d.Marker(
                size=3, color=color, colorscale="Viridis", opacity=0.8
            ),
            opacity=0.8,
            mode="markers",
        )
    )

    fig.update_scenes(aspectmode="data")

    fig.show()


# ----------------------------------------- Calculate MSD --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def calculateMSD(particles):
    initialSphere = particles[0]
    MSD = [0]
    sumMSD = 0
    for frame in particles:
        for particleN in frame.index:
            sumMSD = (
                sumMSD
                + (frame.iloc[particleN]["x"] - initialSphere.iloc[particleN]["x"]) ** 2
                + (frame.iloc[particleN]["y"] - initialSphere.iloc[particleN]["y"]) ** 2
                + (frame.iloc[particleN]["z"] - initialSphere.iloc[particleN]["z"]) ** 2
            )
        MSD.append(sumMSD / len(frame.index))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(0, len(MSD), 1), MSD, "k-")  # black lines, semitransparent alpha=0.1
    plt.show()

    dMSD = pandas.DataFrame(MSD, columns=["MSD (u.u)"])
    dMSD.to_csv("MSD.csv")

    # Save simulation results
    for particleN in range(0, len(particles)):
        particles[particleN].to_csv(str(particleN) + ".csv")

    print("Calculation complete. Printing PNGs")
    maxFrame = 50
    xmin = min(particles[maxFrame]["x"])
    xmax = max(particles[maxFrame]["x"])
    ymin = min(particles[maxFrame]["y"])
    ymax = max(particles[maxFrame]["y"])
    for frameN in range(0, maxFrame):
        maxIP = pandas.DataFrame(columns=["x", "y", "z"])
        for x in range(xmin, xmax, 1):
            for y in range(ymin, ymax, 1):
                maxIP = maxIP.append(
                    {
                        "x": x,
                        "y": y,
                        "z": (
                            (particles[frameN]["x"] == x)
                            & (particles[frameN]["y"] == y)
                        ).sum(),
                    },
                    ignore_index=True,
                )
        z = maxIP.pivot(columns="x", index="y", values="z")
        x = z.columns
        y = z.index
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.contourf(x, y, z, 16)  # , cmap='viridis');
        plt.savefig(str(frameN) + ".png")
        # pylab.show()

    print("PNGs printing complete")
    plt.close("all")
