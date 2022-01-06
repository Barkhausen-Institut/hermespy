import os
import matplotlib.pyplot as plt
import numpy as np

from hermespy.modem import Modem, WaveformGeneratorPskQam
from hermespy.hardware_loop.ni_mmwave import NiMmWaveDualDevice
from hermespy.modem.bits_source import StreamBitsSource


# Create a new simulated device
device = NiMmWaveDualDevice('192.168.189.120', '192.168.189.121')
device.sampling_rate = 100e6

waveform = WaveformGeneratorPskQam()
waveform.modulation_order = 64
waveform.num_data_symbols = 2 ** 14
waveform.num_preamble_symbols = 8

source = StreamBitsSource(os.path.join(os.path.dirname(__file__), '../resources/leena.raw'))
leena_num_bits = 512 * 512 * 8
image_buffer = np.zeros((512, 512), dtype=np.uint8)
image_buffer[0, 0] = 255

bits_per_frame = waveform.bits_per_frame
byte_per_frame = int(waveform.bits_per_frame / 8)
num_frames = int(leena_num_bits / bits_per_frame)

# Add a modem at the simulated device
modem = Modem()
modem.device = device
modem.bits_source = source
modem.waveform_generator = waveform

plt.ion()
figure, axes = plt.subplots()
image = axes.imshow(image_buffer)

for f in range(num_frames):

    print(f"Transmitting frame {f+1} of {num_frames}")

    signal, _, _ = modem.transmit()
    device.trigger()
    _, _, data_bits = modem.receive()

    image_buffer.flat[f*byte_per_frame:(f+1)*byte_per_frame] = np.packbits(data_bits)
    image.set_data(image_buffer)
    figure.canvas.flush_events()


plt.imshow(image_buffer)
plt.show()
