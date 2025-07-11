import matplotlib.pyplot as plt

from hermespy.core import dB
from hermespy.channel import TDL
from hermespy.simulation import Simulation, SNR
from hermespy.modem import BitErrorEvaluator, SimplexLink, RootRaisedCosineWaveform, SingleCarrierLeastSquaresChannelEstimation, SingleCarrierZeroForcingChannelEqualization

# Create a new simulation
from hermespy.core import ConsoleMode
simulation = Simulation(console_mode=ConsoleMode.INTERACTIVE, num_samples=100, debug=True)

# Add two dedicated devices to the simulation
tx_device = simulation.new_device()
rx_device = simulation.new_device()

# Specify the hardware noise model
tx_device.noise_level = SNR(dB(20), tx_device)
rx_device.noise_level = SNR(dB(20), tx_device)

# Specifiy the channel instance linking the two devices
simulation.set_channel(tx_device, rx_device, TDL())

# Define a simplex communication link between the two devices
link = SimplexLink()
tx_device.transmitters.add(link)
rx_device.receivers.add(link)

# Configure the waveform to be transmitted over the link
link.waveform = RootRaisedCosineWaveform(
    symbol_rate=1e6, oversampling_factor=8,
    num_preamble_symbols=10, num_data_symbols=100,
    roll_off=.9,
)
link.waveform.channel_estimation = SingleCarrierLeastSquaresChannelEstimation()
link.waveform.channel_equalization = SingleCarrierZeroForcingChannelEqualization()

# Generate and visualize a communication waveform transmitted over the link
transmission = tx_device.transmit()
transmission.mixed_signal.plot(title='Tx Signal')

# Receive the transmission at rx_device
reception = rx_device.receive(transmission)
reception.operator_receptions[0].equalized_symbols.plot_constellation(title='Rx Constellation')

# Generate and plot a single simulation drop
drop = simulation.scenario.drop()
drop.device_transmissions[0].mixed_signal.plot(title='Tx Signal')
drop.device_receptions[1].operator_receptions[0].equalized_symbols.plot_constellation(title='Rx Constellation')

# Add a bit error rate evaluation to the simulation
ber = BitErrorEvaluator(link, link)

# Iterate over the receiving device's SNR and estimate the respective bit error rates
import numpy as np
simulation.new_dimension('noise_level', dB(*np.linspace(32, -16, 13, endpoint=True)), rx_device)
simulation.add_evaluator(ber)

simulation.results_dir = simulation.default_results_dir()
simulation.plot_results = True
result = simulation.run()
plt.show()
