import os.path as path

import matplotlib.pyplot as plt

from hermespy.hardware_loop.ni_mmwave import NiMmWaveDevice
from hermespy.modem.modem import Modem
from hermespy.modem.waveform_generator_chirp_fsk import WaveformGeneratorChirpFsk
from hermespy.modem.waveform_generator_psk_qam import WaveformGeneratorPskQam, PskQamCorrelationSynchronization
from hermespy.modem.evaluators import BitErrorEvaluator

# Initialize device binding
device = NiMmWaveDevice('192.168.189.120')
device.tx_gain_db = 0
device.rx_gain_db = 50
device.carrier_frequency = 75e9
device.time_buffer = 1e-7

# Configure communication operator and waveform
modem = Modem()
waveform_generator = WaveformGeneratorChirpFsk()
modem.waveform_generator = waveform_generator
#waveform_generator.num_preamble_symbols = 10
#waveform_generator.num_data_symbols = 50
#waveform_generator.modulation_order = 64
waveform_generator.num_pilot_chirps = 0
waveform_generator.num_data_chirps = 10
# waveform_generator.synchronization = PskQamCorrelationSynchronization()
device.sampling_rate = waveform_generator.sampling_rate

modem.device = device

# Transmit and receive waveforms over device
tx_signal, tx_symbols, tx_bits = modem.transmit()
device.trigger()
rx_signal, rx_symbols, rx_bits = modem.receive()

# Compute bit errors
bit_error = BitErrorEvaluator(modem, modem)

# Visualize results
plt.style.use(path.join(path.dirname(__file__), '..', '..', 'hermespy', 'resources', 'styles', 'dark.mplstyle'))
bit_error.evaluate().plot()
tx_signal.plot(title="Tx Signal")
rx_signal.plot(title="Rx Signal")
plt.show()
