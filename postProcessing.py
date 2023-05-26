# Python code for 2D random walk.
# from cmath import sqrt
from cmath import sqrt

# from logging import _Level
import pandas
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from config import *
import numpy as np
import math
from util import particlesToDF
import os


# -----------------------------------------plotting stuff: --------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plotCellData(particles):
    fig = go.Figure(
        data=go.Scatter3d(
            x=particles[0][:, 0],
            y=particles[0][:, 1],
            z=particles[0][:, 2],
            marker=go.scatter3d.Marker(size=3, colorscale="Viridis", opacity=0.8),
            opacity=0.8,
            mode="markers",
        ),
    )

    frames = [
        go.Frame(
            data=[
                go.Scatter3d(
                    x=particles[frame][:, 0],
                    y=particles[frame][:, 1],
                    z=particles[frame][:, 2],
                )
            ],
            traces=[0],
            name=f"frame{frame}",
        )
        for frame in range(len(particles))
    ]
    fig.update(frames=frames)

    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    # Layout
    fig.update_layout(
        title="Cancer Simulation Animation",
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    # {
                    #     "args": [[None], frame_args(0)],
                    #     "label": "&#9724;",  # pause symbol
                    #     "method": "animate",
                    # },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders,
    )

    fig.update_scenes(aspectmode="data")

    fig.show()


def plotMLD(MLD):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(0, len(MLD), 1), MLD, "k-")  # black lines, semitransparent alpha=0.1
    plt.show()


# ----------------------------------------- MLD --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def calculateLinearDistanceNumpy(particles):
    initialSphere = particles[0]
    numParticles = np.shape(initialSphere)[0]
    MLD = [0]
    for frame in particles:
        sumLD = 0.0
        for particleN in range(0, numParticles):
            sumLD += math.sqrt(
                (frame[particleN][0] - initialSphere[particleN][0]) ** 2
                + (frame[particleN][1] - initialSphere[particleN][1]) ** 2
                + (frame[particleN][2] - initialSphere[particleN][2]) ** 2
            )
        MLD.append(sumLD / numParticles)
    return MLD


def saveResults(MLD, particles, outPath=outPath):
    if not os.path.exists(outPath):
        os.makedirs(outPath)

    print("Saving results...")
    dMLD = pandas.DataFrame(MLD, columns=["MLD (u.u)"])
    dMLD.to_csv(outPath + "MLD.csv")

    # Save simulation results
    particlesDF = particlesToDF(particles)
    for particleN in range(0, len(particlesDF)):
        particlesDF[particleN].to_csv(outPath + str(particleN) + ".csv")

    print("Calculation complete. Printing PNGs")
    maxFrame = len(particlesDF) - 1
    xmin = np.min(particles[maxFrame][:, 0])
    xmax = np.max(particles[maxFrame][:, 0])
    ymin = np.min(particles[maxFrame][:, 1])
    ymax = np.max(particles[maxFrame][:, 1])
    for frameN in range(0, maxFrame):
        maxIP = np.zeros(((xmax - xmin) * (ymax - ymin), 3), dtype=np.int32)
        for x in range(xmin, xmax, 1):
            for y in range(ymin, ymax, 1):
                idx = x * (ymax - ymin) + y
                maxIP[idx] = [
                    x,
                    y,
                    np.sum(
                        (particles[frameN][:, 0] == x) & (particles[frameN][:, 1] == y)
                    ),
                ]
        maxIP = pandas.DataFrame(
            {
                "x": maxIP[:, 0],
                "y": maxIP[:, 1],
                "z": maxIP[:, 2],
            },
            dtype=np.int32,
        )
        z = maxIP.pivot(columns="x", index="y", values="z")
        x = z.columns
        y = z.index
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.contourf(x, y, z, 16)  # , cmap='viridis');
        plt.savefig(outPath + str(frameN) + ".png")
        plt.close("all")

    print("PNGs printing complete")
