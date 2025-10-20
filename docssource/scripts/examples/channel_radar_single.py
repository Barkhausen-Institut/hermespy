# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from hermespy.core import dB
from hermespy.channel import SingleTargetRadarChannel
from hermespy.radar import Radar, FMCW, ReceiverOperatingCharacteristic
from hermespy.simulation import Simulation

# Initialize a single device operating at 78.5 GHz
simulation = Simulation(seed=42)
device = simulation.new_device(carrier_frequency=78.5e9)

# Create a radar channel modeling a single target at 10m distance
channel = SingleTargetRadarChannel(10, 1, attenuate=False)
simulation.set_channel(device, device, channel)

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
