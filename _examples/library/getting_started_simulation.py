import matplotlib.pyplot as plt

from hermespy.core import dB
from hermespy.channel import MultipathFading5GTDL
from hermespy.simulation import Simulation
from hermespy.modem import BitErrorEvaluator, SimplexLink, RootRaisedCosineWaveform, SingleCarrierLeastSquaresChannelEstimation, SingleCarrierZeroForcingChannelEqualization

# Create a new simulation
simulation = Simulation()

# Add two dedicated devices to the simulation
tx_device = simulation.new_device()
rx_device = simulation.new_device()

# Specifiy the channel instance linking the two devices
simulation.set_channel(tx_device, rx_device, MultipathFading5GTDL())

# Define a simplex communication link between the two devices
link = SimplexLink(tx_device, rx_device)

# Configure the waveform to be transmitted over the link
link.waveform = RootRaisedCosineWaveform(symbol_rate=1e6, oversampling_factor=8,
                                                   num_preamble_symbols=10, num_data_symbols=100,
                                                   roll_off=.9)
link.waveform.channel_estimation = SingleCarrierLeastSquaresChannelEstimation()
link.waveform.channel_equalization = SingleCarrierZeroForcingChannelEqualization()

# Generate and visualize a communication waveform transmitted over the link
transmission = link.transmit()
transmission.signal.plot()

# Receive the transmission at rx_device
reception = link.receive(transmission.signal)
reception.symbols.plot_constellation()

# Generate and plot a single simulation drop
drop = simulation.scenario.drop()
drop.device_transmissions[0].operator_transmissions[0].signal.plot()
drop.device_receptions[1].operator_receptions[0].equalized_symbols.plot_constellation()

# Add a bit error rate evaluation to the simulation
ber = BitErrorEvaluator(link, link)
ber.evaluate().visualize()

# Iterate over the receiving device's SNR and estimate the respective bit error rates
simulation.new_dimension('snr', dB(20, 16, 12, 8, 4, 0), rx_device)
simulation.add_evaluator(ber)

result = simulation.run()
result.plot()

# Display plots
plt.show()
