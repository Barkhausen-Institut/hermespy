import matplotlib.pyplot as plt

from hermespy.hardware_loop.hardware_loop import HardwareLoop
from hermespy.hardware_loop.ni_mmwave import NiMmWaveDualDevice
from hermespy.modem.modem import Modem
from hermespy.modem.waveform_generator_chirp_fsk import WaveformGeneratorChirpFsk

hardware_loop = HardwareLoop()

device = NiMmWaveDualDevice('192.168.189.120', '192.168.189.121')
hardware_loop.add_device(device)

modem = Modem()
waveform_generator = WaveformGeneratorChirpFsk()
modem.waveform_generator = waveform_generator
waveform_generator.num_data_chirps = 1
waveform_generator.num_pilot_chirps = 3
waveform_generator.guard_interval = 0.
waveform_generator.chirp_duration = 100e-6
waveform_generator.chirp_bandwidth = 500e3
modem.device = device

tx_signal, tx_symbols, tx_bits = modem.transmit()
device.trigger()
rx_signal, rx_symbols, rx_bits = modem.receive()

tx_signal.plot()
rx_signal.plot()
plt.show()
