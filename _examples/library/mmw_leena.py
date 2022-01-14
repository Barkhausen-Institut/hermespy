import os
import matplotlib.pyplot as plt
import numpy as np

from hermespy.modem import Modem
from hermespy.modem.waveform_generator_psk_qam import WaveformGeneratorPskQam, PskQamCorrelationSynchronization, \
    PskQamLeastSquaresChannelEstimation, PskQamZeroForcingChannelEqualization
from hermespy.hardware_loop.ni_mmwave import NiMmWaveDevice
from hermespy.modem.bits_source import StreamBitsSource


# Create a new simulated device
device = NiMmWaveDevice('192.168.189.120')

waveform = WaveformGeneratorPskQam(oversampling_factor=4)
waveform.modulation_order = 16
waveform.num_data_symbols = 1024
waveform.num_preamble_symbols = 100
waveform.guard_interval = 0.
waveform.synchronization = PskQamCorrelationSynchronization()
waveform.channel_estimation = PskQamLeastSquaresChannelEstimation()
waveform.channel_equalization = PskQamZeroForcingChannelEqualization()
device.sampling_rate = waveform.sampling_rate

source = StreamBitsSource(os.path.join(os.path.dirname(__file__), '../resources/leena.raw'))
leena_num_bits = 512 * 512 * 8
image_buffer = np.ones((512, 512), dtype=np.uint8)
image_buffer[0, 0] = 255

# Add a modem at the simulated device
modem = Modem()
modem.device = device
modem.bits_source = source
modem.waveform_generator = waveform


# Compute number of required frames
bits_per_frame = modem.num_data_bits_per_frame
byte_per_frame = int(bits_per_frame / 8)
num_frames = int(leena_num_bits / bits_per_frame)

plt.ion()
fig, axes = plt.subplots()
image = axes.imshow(image_buffer)

for f in range(num_frames):

    signal, _, tx_bits = modem.transmit()
    device.trigger()
    _, _, data_bits = modem.receive()

    if len(data_bits) > 0:
        image_buffer.flat[f*byte_per_frame:(f+1)*byte_per_frame] = np.packbits(data_bits)
    image.set_data(image_buffer)
    fig.canvas.flush_events()

plt.show()
