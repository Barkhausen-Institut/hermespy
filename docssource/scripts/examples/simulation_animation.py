# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from hermespy.core import Transformation
from hermespy.simulation import Simulation, LinearTrajectory


# Create a new simulation featuring two devices
simulation = Simulation()
device_alpha = simulation.new_device()
device_beta = simulation.new_device()

# Assign each device a linear trajectory
device_alpha.trajectory = LinearTrajectory(
    initial_pose=Transformation.From_Translation(np.array([0, 0, 20])),
    final_pose=Transformation.From_Translation(np.array([0, 100, 5])),
    duration=60,
)
device_beta.trajectory = LinearTrajectory(
    initial_pose=Transformation.From_Translation(np.array([100, 100, 0])),
    final_pose=Transformation.From_Translation(np.array([0, 0, 0])),
    duration=60,
)

# Visualize the trajectories
simulation.scenario.visualize()
plt.show()
