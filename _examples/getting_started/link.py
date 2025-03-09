import matplotlib.pyplot as plt

# Import required HermesPy modules
from hermespy.channel import IdealChannel
from hermespy.simulation import SimulatedDevice
from hermespy.modem import TransmittingModem, ReceivingModem, RootRaisedCosineWaveform, BitErrorEvaluator

# Create two simulated devices acting as source and sink
tx_device = SimulatedDevice()
rx_device = SimulatedDevice()

# Define a transmit operation on the first device
tx_operator = TransmittingModem()
tx_operator.waveform = RootRaisedCosineWaveform(symbol_rate=1e6, num_preamble_symbols=0, num_data_symbols=40, oversampling_factor=8, roll_off=.9)
tx_device.transmitters.add(tx_operator)

# Define a receive operation on the second device
rx_operator = ReceivingModem()
rx_operator.waveform = RootRaisedCosineWaveform(symbol_rate=1e6, num_preamble_symbols=0, num_data_symbols=40, oversampling_factor=8, roll_off=.9)
rx_device.receivers.add(rx_operator)

# Evaluate bit errors during transmission and visualize the received symbol constellation
evaluator = BitErrorEvaluator(tx_operator, rx_operator)

# Simulate a channel between the two devices
channel = IdealChannel()

# Simulate the signal transmission over the channel
transmission = tx_device.transmit()
propagation = channel.propagate(transmission, tx_device, rx_device)
reception = rx_device.receive(propagation)

# Visualize communication performance
evaluator.evaluate().visualize()
reception.operator_receptions[0].equalized_symbols.plot_constellation()
plt.show()
