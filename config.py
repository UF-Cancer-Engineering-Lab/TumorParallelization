import json

config_path = "./config/config.json"
config_data = json.load(open(config_path))


# ----------------------------------------- Program Parameters --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# defining parameters of the simulation
n = config_data["n"]  # number of timeSteps
maxTries = config_data["maxTries"]  # max tries for a particle to move
particlesNumber = config_data["particlesNumber"]  # initial particle count
porosityFraction = config_data["porosityFraction"]  # porosity fraction of particles,
# where porosity fraction is the ratio of void volume to total volume
# each "particle", or "cell" has some void space in it
sphereRadius = config_data["sphereRadius"]
shouldSaveResults = config_data["shouldSaveResults"]
showMLDVisualization = config_data["showMLDVisualization"]
show3DVisualization = config_data["show3DVisualization"]
outPath = config_data["outPath"]
scene_file_name = config_data["scene_file_name"]
max_vram_allocation_gb = config_data["max_vram_allocation_gb"]
