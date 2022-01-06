import matplotlib.pyplot as plt

from hermespy.hardware_loop.hardware_loop import HardwareLoop
from hermespy.hardware_loop.ni_mmwave import NiMmWaveDevice
from hermespy.modem.modem import Modem
from hermespy.modem.waveform_generator_psk_qam import WaveformGeneratorPskQam, PskQamCorrelationSynchronization
from hermespy.modem.evaluators import BitErrorEvaluator

hardware_loop = HardwareLoop()

device = NiMmWaveDevice('192.168.189.120')
device.tx_gain_db = 0
device.rx_gain_db = 50
device.carrier_frequency = 75e9
device.time_buffer = 1e-7

hardware_loop.add_device(device)

modem = Modem()
waveform_generator = WaveformGeneratorPskQam(oversampling_factor=4)
modem.waveform_generator = waveform_generator
waveform_generator.num_preamble_symbols = 10
waveform_generator.num_data_symbols = 50
waveform_generator.modulation_order = 64
waveform_generator.synchronization = PskQamCorrelationSynchronization()
device.sampling_rate = waveform_generator.sampling_rate

modem.device = device

bit_error = BitErrorEvaluator(modem, modem)

tx_signal, tx_symbols, tx_bits = modem.transmit()
device.trigger()
rx_signal, rx_symbols, rx_bits = modem.receive()

bit_error.evaluate().plot()

tx_signal.plot()
rx_signal.plot()
plt.show()
