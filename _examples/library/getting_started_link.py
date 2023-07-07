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
tx_operator.waveform_generator = RootRaisedCosineWaveform(symbol_rate=1e6, num_preamble_symbols=0, num_data_symbols=40, oversampling_factor=8, roll_off=.9)
tx_device.transmitters.add(tx_operator)

# Define a receive operation on the second device
rx_operator = ReceivingModem()
rx_operator.waveform_generator = RootRaisedCosineWaveform(symbol_rate=1e6, num_preamble_symbols=0, num_data_symbols=40, oversampling_factor=8, roll_off=.9)
rx_device.receivers.add(rx_operator)

# Simulate a channel between the two devices
channel = IdealChannel(tx_device, rx_device)

# Simulate the signal transmission over the channel
transmission = tx_operator.transmit()
rx_signal, _, channel_state = channel.propagate(tx_device.transmit())
rx_device.process_input(rx_signal)
reception = rx_operator.receive()

# Evaluate bit errors during transmission and visualize the received symbol constellation
evaluator = BitErrorEvaluator(tx_operator, rx_operator)
evaluator.evaluate().plot()
reception.symbols.plot_constellation()
plt.show()
