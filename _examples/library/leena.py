import os
import matplotlib.pyplot as plt
import numpy as np
from math import ceil

from hermespy.modem import Modem, WaveformGeneratorPskQam
from hermespy.core.scenario import Scenario
from hermespy.simulation import SimulatedDevice
from hermespy.modem.bits_source import StreamBitsSource
from hermespy.coding.ldpc_binding.ldpc import LDPCBinding

leena = np.fromfile(os.path.join(os.path.dirname(__file__), '../resources/leena.raw'), dtype=np.uint8).reshape(512, 512)

# Create a new HermesPy simulation scenario
scenario = Scenario[SimulatedDevice]()

# Create a new simulated device
device = SimulatedDevice()
scenario.add_device(device)

waveform = WaveformGeneratorPskQam()
waveform.modulation_order = 64
waveform.num_data_symbols = 512
waveform.num_preamble_symbols = 8

source = StreamBitsSource(os.path.join(os.path.dirname(__file__), '../resources/leena.raw'))
leena_num_bits = 512 * 512 * 8
image_buffer = np.ones((512, 512), dtype=np.uint8)
image_buffer[0, 0] = 255

bits_per_frame = waveform.bits_per_frame
byte_per_frame = int(waveform.bits_per_frame / 8)
num_frames = int(leena_num_bits / bits_per_frame)

# Add a modem at the simulated device
modem = Modem()
modem.device = device
modem.bits_source = source
modem.waveform_generator = waveform
modem.encoder_manager.add_encoder(LDPCBinding())

plt.ion()
fig, axes = plt.subplots()
image = axes.imshow(image_buffer)

for f in range(num_frames):

    signal, _, _ = modem.transmit()
    device_transmissions = device.transmit(clear_cache=True)
    device.receive_signal(device_transmissions[0], snr=2)
    _, _, data_bits = modem.receive()

    image_buffer.flat[f*byte_per_frame:(f+1)*byte_per_frame] = np.packbits(data_bits)
    image.set_data(image_buffer)
    fig.canvas.flush_events()
