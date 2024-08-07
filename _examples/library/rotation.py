# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from hermespy.core import Transformation
from hermespy.simulation import Simulation, LinearTrajectory, StaticTrajectory


# Setup flags
flag_traj_static_a = False  # Is alpha device trajectory static?
flag_traj_static_b = False  # Is beta device trajectory static?
flag_lookat = True  # Should the devices look at each other?

# Create a new simulation featuring two devices
simulation = Simulation()
device_alpha = simulation.new_device()
device_beta = simulation.new_device()
duration = 60

# Init positions and poses
init_pose_alpha = Transformation.From_Translation(np.array([10., 10., 0.]))
fina_pose_alpha = Transformation.From_Translation(np.array([50., 50., 0.]))
init_pose_beta = Transformation.From_Translation(np.array([30., 10., 20.]))
fina_pose_beta = Transformation.From_Translation(np.array([30., 20., 20.]))
if flag_lookat:
    init_pose_alpha = init_pose_alpha.lookat(init_pose_beta.translation)
    fina_pose_alpha = fina_pose_alpha.lookat(fina_pose_beta.translation)
    init_pose_beta = init_pose_beta.lookat(init_pose_alpha.translation)
    fina_pose_beta = fina_pose_beta.lookat(fina_pose_alpha.translation)

# Assign each device a trajectory
# alpha
if flag_traj_static_a:
    device_alpha.trajectory = StaticTrajectory(init_pose_alpha)
else:
    device_alpha.trajectory = LinearTrajectory(init_pose_alpha, fina_pose_alpha, duration)
# beta
if flag_traj_static_b:
    device_beta.trajectory = StaticTrajectory(init_pose_beta)
else:
    device_beta.trajectory = LinearTrajectory(init_pose_beta, fina_pose_beta, duration)

# Lock the devices onto each other
if flag_lookat:
    device_alpha.trajectory.lookat(device_beta.trajectory)
    device_beta.trajectory.lookat(device_alpha.trajectory)

visualization = simulation.scenario.visualize()
with plt.ion():
    for timestamp in np.linspace(0, duration, 200):
        simulation.scenario.visualize.update_visualization(visualization, time=timestamp)
        plt.pause(0.1)
    plt.show()
