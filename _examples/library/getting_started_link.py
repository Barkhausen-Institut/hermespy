import matplotlib.pyplot as plt

# Import required HermesPy modules
from hermespy.channel import Channel
from hermespy.simulation import SimulatedDevice
from hermespy.modem import Modem, RootRaisedCosineWaveform, BitErrorEvaluator

# Create two simulated devices acting as source and sink
tx_device = SimulatedDevice()
rx_device = SimulatedDevice()

# Define a transmit operation on the first device
tx_operator = Modem()
tx_operator.waveform_generator = RootRaisedCosineWaveform(symbol_rate=1e6, num_preamble_symbols=0, num_data_symbols=40, oversampling_factor=8, roll_off=.9)
tx_operator.device = tx_device

# Define a receive operation on the second device
rx_operator = Modem()
rx_operator.waveform_generator = RootRaisedCosineWaveform(symbol_rate=1e6, num_preamble_symbols=0, num_data_symbols=40, oversampling_factor=8, roll_off=.9)
rx_operator.device = rx_device

# Simulate a channel between the two devices
channel = Channel(tx_operator.device, rx_operator.device)

# Simulate the signal transmission over the channel
tx_signal, _, tx_bits = tx_operator.transmit()
rx_signal, _, channel_state = channel.propagate(tx_signal)
rx_device.receive(rx_signal)
_, rx_symbols, rx_bits = rx_operator.receive()

# Evaluate bit errors during transmission and visualize the received symbol constellation
evaluator = BitErrorEvaluator(tx_operator, rx_operator)
evaluator.evaluate().plot()
rx_symbols.plot_constellation()
plt.show()
