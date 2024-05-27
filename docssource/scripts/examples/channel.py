# -*- coding: utf-8 -*-

from hermespy.channel import Channel
from hermespy.simulation import SimulatedDevice

# Hidden code block
import matplotlib.pyplot as plt

from hermespy.core import dB
from hermespy.channel import TDL as Channel
from hermespy.modem import BitErrorEvaluator, SimplexLink, RootRaisedCosineWaveform, SingleCarrierLeastSquaresChannelEstimation, SingleCarrierZeroForcingChannelEqualization

# Initialize two devices to be linked by a channel
alpha_device = SimulatedDevice()
beta_device = SimulatedDevice()

# Create a channel between the two devices
channel = Channel()

# Configure communication link between the two devices
link = SimplexLink(alpha_device, beta_device)

# Specify the waveform and postprocessing to be used by the link
link.waveform = RootRaisedCosineWaveform(symbol_rate=1e8, oversampling_factor=2,
                                         num_data_symbols=1000, num_preamble_symbols=10, pilot_rate=10)
link.waveform.channel_estimation = SingleCarrierLeastSquaresChannelEstimation()
link.waveform.channel_equalization = SingleCarrierZeroForcingChannelEqualization()

# Generate the device's transmissions
alpha_transmission = alpha_device.transmit()
beta_transmission = beta_device.transmit()

# Propagate the transmissions over the channel
channel_realization = channel.realize()
alpha_propagation = channel_realization.sample(alpha_device, beta_device).propagate(alpha_transmission)
beta_propagation = channel_realization.sample(beta_device, alpha_device).propagate(beta_transmission)

# Receive the transmissions at both devices
alpha_reception = alpha_device.receive(beta_propagation)
beta_reception = beta_device.receive(alpha_propagation)


from hermespy.simulation import Simulation

# Create a new simulation and add the two existing devices
simulation = Simulation()

simulation.add_device(alpha_device)
simulation.add_device(beta_device)

# Configure the simulation to use the appropriate channel
simulation.set_channel(alpha_device, beta_device, channel)

# Configure a bit error rate evaluation
simulation.add_evaluator(BitErrorEvaluator(link, link))

# Run the simulation
simulation.new_dimension('noise_level', dB(0, 2, 4, 8, 12, 16, 20), beta_device)
result = simulation.run()

# Plot the results
result.plot()
plt.show()
