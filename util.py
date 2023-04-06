import pandas
import numpy as np

# -----------------------------------------util: --------------------------------------------------------------------------------------------------------------------------------------------------------------------
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