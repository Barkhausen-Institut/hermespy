# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from hermespy.core import dB, Transformation
from hermespy.channel import MultiTargetRadarChannel, VirtualRadarTarget, FixedCrossSection
from hermespy.radar import Radar, FMCW, ReceiverOperatingCharacteristic
from hermespy.simulation import Simulation, StaticTrajectory

# Initialize a single device operating at 78.5 GHz
simulation = Simulation()
device = simulation.new_device(carrier_frequency=78.5e9)

# Create a radar channel modeling two targets at 10m and 50m distance
channel = MultiTargetRadarChannel(attenuate=False)
simulation.set_channel(device, device, channel)

first_target = VirtualRadarTarget(
    FixedCrossSection(1),
    StaticTrajectory(Transformation.From_Translation(np.array([10, 0, 0]))),
    static=False,
)
second_target = VirtualRadarTarget(
    FixedCrossSection(1),
    StaticTrajectory(Transformation.From_Translation(np.array([0, 50, 0]))),
    static=True,
)
channel.add_target(first_target)
channel.add_target(second_target)

# Configure an FMCW radar with 5 GHz bandwidth illuminating the target
radar = Radar(FMCW())
device.add_dsp(radar)

# Configure a simulation evluating the radar's operating characteristics
simulation.add_evaluator(ReceiverOperatingCharacteristic(radar, device, device, channel))
simulation.new_dimension('noise_level', dB(np.arange(0, -22, -2).tolist()), device)
simulation.num_samples = 1000

# Run simulation and plot resulting ROC curve
result = simulation.run()
result.plot()
plt.show()
