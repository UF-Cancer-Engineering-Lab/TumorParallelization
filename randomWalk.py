# Python code for 2D random walk.
# from cmath import sqrt
from cmath import sqrt

# from logging import _Level
import logging
import math
import numpy as np
import pylab
import random
import pandas
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D





# ----------------------------------------- Voxel Framework --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Voxel Code: (max space = initial radius + number of time steps = 106, round to 105):

# but first, try it with a 6 by 6 by 6 space
voxel_length = 2  # meaning divide space up into 1 by 1 by 1
dimension = 40  


class Voxel:

    #length = voxelDivision  # static variable: all objects of the class cube will have a fixed length

    def __init__(self):
        self.list_particles = []

#if initialze voxel_array using np.ones(), then it'll be an array of integers. Instead,
#use the np.empty() method to initalize an array with uninitialized entries!
voxel_array = [[[Voxel() for i in range(dimension)] for j in range(dimension)] for k in range(dimension)]


#code to create space for voxels; will wrap in a method/class soon
space = np.indices((dimension+1, dimension+1, dimension+1), dtype = float)
space[0] = (voxel_length * (space[0] - (dimension/2))) - 0.5
space[1] = (voxel_length * (space[1] - (dimension/2))) - 0.5
space[2] = (voxel_length * (space[2] - (dimension/2))) - 0.5

#Define matrix for colors of each voxel:
data = np.zeros([dimension,dimension,dimension]) 
colors = np.empty((dimension,dimension,dimension), dtype=object)
colors[:][:][:] = 'blue'




# given a particles coordinates, find the nearest voxel_corner
def find_nearest_corner(x, y, z):
    #list of all corners:
    voxel_corners = space[2][0][0][0:dimension]
    while sum(voxel_corners == x) != 1:
        x -= 0.5
    while sum(voxel_corners == y) != 1:
        y -= 0.5
    while sum(voxel_corners == z) != 1:
        z -= 0.5
    
    corner_x = x
    corner_y = y
    corner_z = z
    
    return [corner_x, corner_y, corner_z]


# Hash Function: whatever a voxels corner is, it's position in the array is for a_i E {x, y, z}:
 # ((a_i + 0.5)/2) + 5 [<- inverse function], for all ai. This transformation, now called b_i, is the position of the voxel
 # in array "data", by index [b_1, b_2, b_3]

def hash(x):
    return ((x+0.5)/voxel_length) + (dimension/2)


# ----------------------------------------- Program Start --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# defining parameters of the simulation
n = 100  # number of timeSteps
maxTries = 6  # max tries for a particle to move
particlesNumber = 800  # initial particle count
porosityFraction = 0.05  # porosity fraction of particles,
# where porosity fraction is the ratio of void volume to total volume
# each "particle", or "cell" has some void space in it
capillaryRadius = 3  # radius of x and y axes capilarry freeways

# child variables
vacancies = round(particlesNumber * porosityFraction)
totalPositions = particlesNumber + vacancies

# creating full array
sphereRadius = 5
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
    initialSphere = pandas.DataFrame(columns=["x", "y", "z"])
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
                initialSphere = initialSphere.append(
                    {"x": x, "y": y, "z": z}, ignore_index=True
                )
                # print(len(initialSphere.index))


squaredRadius = sphereRadius**2
squaredCapillaryRadius = capillaryRadius**2
totalPositions = len(initialSphere.index)
vacancies = round(totalPositions * porosityFraction)
print("Initial sphere complete")

# this randomizes the areas where the particles can and cannot go in the sphere
for i in range(1, vacancies + 1):
    val = random.randint(0, len(initialSphere.index + 1))
    initialSphere = initialSphere.drop(labels=val, axis=0).reset_index(drop=True)
    # ^^^ .reset_index method just recalculates the indices so that the gap in indices isn't there anymore

# creating two array for containing x and y coordinate
# of size equals to the number of size and filled up with 0's
particles = [initialSphere]  # particles is now a list with single entry containing the x,y,z coordinates of the sphere










#------------------------- Assigning initial particle position to appropriate voxels: ------------------------------------------------
for particleN in initialSphere.index:
    x = initialSphere["x"].iloc[particleN]
    y = initialSphere["y"].iloc[particleN]
    z = initialSphere["z"].iloc[particleN]

    
    #Voxel stuff
    a_particle = [particleN, x, y, z]
    nearest_corner = find_nearest_corner(x, y, z)

    voxel_indice_x = int(hash(nearest_corner[0]))
    voxel_indice_y = int(hash(nearest_corner[1]))
    voxel_indice_z = int(hash(nearest_corner[2]))

    # print("Particle",particleN,":", x, y, z)
    # print("Nearest Corner:", nearest_corner[0], nearest_corner[1], nearest_corner[2])
    # print("Particle",particleN,"'s Voxel Indice:", voxel_indice_x, voxel_indice_y, voxel_indice_z)
    # print("\n")

    voxel_array[voxel_indice_x][voxel_indice_y][voxel_indice_z].list_particles += [a_particle]









# random walking
# for i in range(1,n+1):
for i in range(1, 15):
    particles.append(
        particles[i - 1].copy(deep=True)
    )  #  "deep copy = true"copies all the values of "initialSphere"
    for particleN in initialSphere.index:
        tries = 0
        while tries < maxTries:
            val = random.randint(1, 6)
            if val == 1:
                # doing iloc just means that we can index no by a number, instead of labels like "x"
                x = particles[i - 1]["x"].iloc[particleN] + 1
                y = particles[i - 1]["y"].iloc[particleN]
                z = particles[i - 1]["z"].iloc[particleN]
            elif val == 2:
                x = particles[i - 1]["x"].iloc[particleN] - 1
                y = particles[i - 1]["y"].iloc[particleN]
                z = particles[i - 1]["z"].iloc[particleN]
            elif val == 3:
                x = particles[i - 1]["x"].iloc[particleN]
                y = particles[i - 1]["y"].iloc[particleN] + 1
                z = particles[i - 1]["z"].iloc[particleN]
            elif val == 4:
                x = particles[i - 1]["x"].iloc[particleN]
                y = particles[i - 1]["y"].iloc[particleN] - 1
                z = particles[i - 1]["z"].iloc[particleN]
            elif val == 5:
                x = particles[i - 1]["x"].iloc[particleN]
                y = particles[i - 1]["y"].iloc[particleN]
                z = particles[i - 1]["z"].iloc[particleN] + 1
            else:
                x = particles[i - 1]["x"].iloc[particleN]
                y = particles[i - 1]["y"].iloc[particleN]
                z = particles[i - 1]["z"].iloc[particleN] - 1
            x_2 = x**2
            y_2 = y**2
            z_2 = z**2
            # comparing this values to the previous dataFrame (particles[i-1]) means we don't want the particle to move to a past position, nor do we want it to move to the x,y,z coordinate of a current position, last part is we want to squared distance to be within squared capillary radius
            if ~(
                (
                    (particles[i - 1]["x"] == x) & (particles[i - 1]["y"] == y) & (particles[i - 1]["z"] == z)
                ).any(axis=0)
                or (
                    (particles[i]["x"] == x) & (particles[i]["y"] == y) & (particles[i]["z"] == z)
                ).any(axis=0)
                or ~(
                    (x_2 + y_2 + z_2) < squaredRadius or (x_2 + z_2) < squaredCapillaryRadius or (y_2 + z_2) < squaredCapillaryRadius
                )
            ):
                particles[i].at[particleN, "x"] = x
                particles[i].at[particleN, "y"] = y
                particles[i].at[particleN, "z"] = z
                tries = maxTries


            #Voxel stuff: If particle updates its position in space, update it's voxel position:
                a_particle = [particleN, x, y, z]
                nearest_corner = find_nearest_corner(x, y, z)

                voxel_indice_x = int(hash(nearest_corner[0]))
                voxel_indice_y = int(hash(nearest_corner[1]))
                voxel_indice_z = int(hash(nearest_corner[2]))

                # if i >= 4:
                #     print("Inserting", particleN, "into voxel array: ", voxel_indice_x, voxel_indice_y, voxel_indice_z)

                voxel_array[voxel_indice_x][voxel_indice_y][voxel_indice_z].list_particles += [a_particle]


            #Extra: turn this particle's voxel's position to "1" to track it in the simulation:
            
                data[voxel_indice_x][voxel_indice_y][voxel_indice_z] = 1


                # print("Particle",particleN,":", x, y, z)
                # print("Nearest Corner:", nearest_corner[0], nearest_corner[1], nearest_corner[2])
                # print("Particle",particleN,"'s Voxel Indice:", voxel_indice_x, voxel_indice_y, voxel_indice_z)
                # print("\n")

            #Now, clear that particle's previous voxel position, using it;s i-1 Pandas dataFrame:

                #Retrieve old x,y,z coordinates of the particle
                old_x = particles[i-1].at[particleN, "x"]
                old_y = particles[i-1].at[particleN, "y"]
                old_z = particles[i-1].at[particleN, "z"]

                #Find identifying voxel it's in, by finding nearest_corner
                old_nearest_corner = find_nearest_corner(old_x, old_y, old_z)

                #Find indices of voxel that it was previously in using hash function:
                old_voxel_indice_x = int(hash(old_nearest_corner[0]))
                old_voxel_indice_y = int(hash(old_nearest_corner[1]))
                old_voxel_indice_z = int(hash(old_nearest_corner[2]))

                # print("Particle",particleN, i, "old pos:", old_x, old_y, old_z)
                # print("Nearest Corner:", nearest_corner[0], nearest_corner[1], nearest_corner[2])
                # print("Particle",particleN,"'s Voxel Indice:", voxel_indice_x, voxel_indice_y, voxel_indice_z)
                # print("\n")

                #clear that particle's entry from voxel's particle list using the .remove() function:
                voxel_array[old_voxel_indice_x][old_voxel_indice_y][old_voxel_indice_z].list_particles.remove([particleN, old_x, old_y, old_z])



            


            else:
                tries = tries + 1

    print("Time steps elapsed: " + str(i))


print("Simulation complete. Calculating mean squared displacement")


# -----------------------------------------plotting stuff: --------------------------------------------------------------------------------------------------------------------------------------------------------------------
#pylab.title("Random Walk ($n = " + str(n) + "$ steps)")
pylab.title("Random Walk ($n = " + str(3) + "$ steps)")
ax = pylab.axes(projection="3d")
frame = -1
x = particles[frame]["x"].tolist()
y = particles[frame]["y"].tolist()
z = particles[frame]["z"].tolist()

# Each matrix for x,y,z is a 11 x 11 x 11 matrix, for a voxel array of [10, 10, 10]
axis_x = space[0]
axis_y = space[1]
axis_z = space[2]

#This function plots the voxels:
#we have to create a 3D array for the first 3 parameters. The x-parameter will handle the first matrix of
#this 3D array, the y-parameter will handle to 2nd matrix of this 3d matrix, and the z-parameter will handle
#the 3rd matrix of this 3D-array
ax.voxels(axis_x, axis_y, axis_z, data, edgecolor="k", facecolors=colors, alpha = 0.5)
ax.scatter(x, y, z, c=z, cmap="viridis", linewidth=3)


pylab.show()


# ----------------------------------------- Calculate MSD --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# MSD = [0]
# sumMSD = 0
# for frame in particles:
#     for particleN in frame.index:
#         sumMSD = (
#             sumMSD
#             + (frame.iloc[particleN]["x"] - initialSphere.iloc[particleN]["x"]) ** 2
#             + (frame.iloc[particleN]["y"] - initialSphere.iloc[particleN]["y"]) ** 2
#             + (frame.iloc[particleN]["z"] - initialSphere.iloc[particleN]["z"]) ** 2
#         )
#     MSD.append(sumMSD / len(frame.index))

# fig, ax = plt.subplots(figsize=(12, 6))
# ax.plot(range(0, len(MSD), 1), MSD, "k-")  # black lines, semitransparent alpha=0.1
# plt.show()

# dMSD = pandas.DataFrame(MSD, columns=["MSD (u.u)"])
# dMSD.to_csv("MSD.csv")

# Save simulation results
# for particleN in range(0, len(particles)):
#     particles[particleN].to_csv(str(particleN) + ".csv")

# print("Calculation complete. Printing PNGs")
# maxFrame = 50
# xmin = min(particles[maxFrame]["x"])
# xmax = max(particles[maxFrame]["x"])
# ymin = min(particles[maxFrame]["y"])
# ymax = max(particles[maxFrame]["y"])
# for frameN in range(0, maxFrame):
#     maxIP = pandas.DataFrame(columns=["x", "y", "z"])
#     for x in range(xmin, xmax, 1):
#         for y in range(ymin, ymax, 1):
#             maxIP = maxIP.append(
#                 {
#                     "x": x,
#                     "y": y,
#                     "z": (
#                         (particles[frameN]["x"] == x) & (particles[frameN]["y"] == y)
#                     ).sum(),
#                 },
#                 ignore_index=True,
#             )
#     z = maxIP.pivot(columns="x", index="y", values="z")
#     x = z.columns
#     y = z.index
#     fig, ax = plt.subplots(figsize=(12, 12))
#     ax.contourf(x, y, z, 16)  # , cmap='viridis');
#     plt.savefig(str(frameN) + ".png")
#     # pylab.show()

# print("PNGs printing complete")
# plt.close("all")



def print_voxel(x,y,z):
    for i in range(x):
        for j in range(y):
            for k in range(z):
                print("At Voxel ", "[", i, "]", "[", j, "]","[", k, "]: \n",)
                particle_array = voxel_array[i][j][k].particles
                for q in range(len(particle_array)):
                    print("Particle ", q, ":")
                    print("X: ", particle_array[q][0])
                    print("Y: ", particle_array[q][1])
                    print("Z: ", particle_array[q][2])
