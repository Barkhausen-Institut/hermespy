# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from hermespy.core import dB
from hermespy.channel import MultipathFading5GTDL, TDLType
from hermespy.modem import BitErrorEvaluator, RRCWaveform, SimplexLink, SCLeastSquaresChannelEstimation, SCZeroForcingChannelEqualization
from hermespy.simulation import Simulation


# Initialize two devices to be linked by a channel
simulation = Simulation()
alpha_device = simulation.new_device(carrier_frequency=1e8)
beta_device = simulation.new_device(carrier_frequency=1e8)

# Create a channel between the two devices
channel = MultipathFading5GTDL(model_type=TDLType.B, rms_delay=1e-9)
simulation.set_channel(alpha_device, beta_device, channel)

# Configure communication link between the two devices
link = SimplexLink(alpha_device, beta_device)

# Specify the waveform and postprocessing to be used by the link
link.waveform = RRCWaveform(
    symbol_rate=1e8, oversampling_factor=2, num_data_symbols=1000, 
    num_preamble_symbols=10, pilot_rate=10
)
link.waveform.channel_estimation = SCLeastSquaresChannelEstimation()
link.waveform.channel_equalization = SCZeroForcingChannelEqualization()

# Configure a simulation to evaluate the link's BER and sweep over the receive SNR
simulation.add_evaluator(BitErrorEvaluator(link, link))
simulation.new_dimension('noise_level', dB(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20), beta_device)

# Run simulation and plot resulting SNR curve
result = simulation.run()
result.plot()
plt.show()
