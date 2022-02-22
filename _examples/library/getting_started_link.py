import matplotlib.pyplot as plt

from hermespy.channel import Channel
from hermespy.simulation import SimulatedDevice
from hermespy.modem import Modem, WaveformGeneratorPskQam, BitErrorEvaluator

tx_device = SimulatedDevice()
rx_device = SimulatedDevice()

tx_operator = Modem()
tx_operator.waveform_generator = WaveformGeneratorPskQam(oversampling_factor=8)
tx_operator.device = tx_device

rx_operator = Modem()
rx_operator.waveform_generator = WaveformGeneratorPskQam(oversampling_factor=8)
rx_operator.device = rx_device

channel = Channel(tx_operator.device, rx_operator.device)

evaluator = BitErrorEvaluator(tx_operator, rx_operator)

tx_signal, _, tx_bits = tx_operator.transmit()
rx_signal, _, channel_state = channel.propagate(tx_signal)
rx_device.receive(rx_signal)
_, rx_symbols, rx_bits = rx_operator.receive()

evaluator.evaluate().plot()
rx_symbols.plot_constellation()
plt.show()
