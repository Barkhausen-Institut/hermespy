import matplotlib.pyplot as plt

from hermespy.hardware_loop.hardware_loop import HardwareLoop
from hermespy.hardware_loop.ni_mmwave import NiMmWaveDevice
from hermespy.modem.modem import Modem
from hermespy.modem.waveform_generator_chirp_fsk import WaveformGeneratorChirpFsk

hardware_loop = HardwareLoop()

device = NiMmWaveDevice('192.168.189.120', 5558)
hardware_loop.add_device(device)

modem = Modem()
modem.waveform_generator = WaveformGeneratorChirpFsk()
modem.device = device

tx_signal, tx_symbols, tx_bits = modem.transmit()
device.trigger()
rx_signal, rx_symbols, rx_bits = modem.receive()

tx_signal.plot()
rx_signal.plot()
plt.show()
