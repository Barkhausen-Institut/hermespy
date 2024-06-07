# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from hermespy.core import dB
from hermespy.channel import MultipathFadingChannel
from hermespy.modem import BitErrorEvaluator, RRCWaveform, SimplexLink, SCLeastSquaresChannelEstimation, SCZeroForcingChannelEqualization
from hermespy.simulation import Simulation


# Initialize two devices to be linked by a channel
simulation = Simulation()
alpha_device = simulation.new_device(carrier_frequency=1e8)
beta_device = simulation.new_device(carrier_frequency=1e8)

# Create a channel between the two devices
delays = 1e-9 * np.array([0, 0.3819, 0.4025, 0.5868, 0.4610, 0.5375, 0.6708, 0.5750, 0.7618, 1.5375, 1.8978, 2.2242, 2.1717, 2.4942, 2.5119, 3.0582, 4.0810, 4.4579, 4.5695, 4.7966, 5.0066, 5.3043, 9.6586])
powers = 10 ** (np.array([-13.4, 0, -2.2, -4, -6, -8.2, -9.9, -10.5, -7.5, -15.9, -6.6, -16.7, -12.4, -15.2, -10.8, -11.3, -12.7, -16.2, -18.3, -18.9, -16.6, -19.9, -29.7]) / 10)
rice_factors = np.zeros_like(delays)
channel = MultipathFadingChannel(delays, powers, rice_factors)
simulation.set_channel(alpha_device, beta_device, channel)

# Configure communication link between the two devices
link = SimplexLink(alpha_device, beta_device)

# Specify the waveform and postprocessing to be used by the link
link.waveform = RRCWaveform(
    symbol_rate=1e8, oversampling_factor=2, num_data_symbols=1000, 
    num_preamble_symbols=10, pilot_rate=10)
link.waveform.channel_estimation = SCLeastSquaresChannelEstimation()
link.waveform.channel_equalization = SCZeroForcingChannelEqualization()

# Configure a simulation to evaluate the link's BER and sweep over the receive SNR
simulation.add_evaluator(BitErrorEvaluator(link, link))
simulation.new_dimension('noise_level', dB(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20), beta_device)

# Run simulation and plot resulting SNR curve
result = simulation.run()
result.plot()
plt.show()
