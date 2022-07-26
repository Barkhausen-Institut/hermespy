# -*- coding: utf-8 -*-
"""
==========================
Image Transmission Example
==========================

Demonstrate link level routines by transmitting an image over the link in (approximately) real time.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from hermespy.modem import Modem, RootRaisedCosineWaveform, SingleCarrierCorrelationSynchronization, SingleCarrierLeastSquaresChannelEstimation, SingleCarrierZeroForcingChannelEqualization
from hermespy.core.scenario import Scenario
from hermespy.modem.waveform_single_carrier import SingleCarrierSynchronization
from hermespy.simulation import SimulatedDevice
from hermespy.modem.bits_source import StreamBitsSource

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.7"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


# Create a new HermesPy simulation scenario
scenario = Scenario[SimulatedDevice]()

# Create a new simulated device
device = SimulatedDevice()
scenario.add_device(device)

waveform_generator = RootRaisedCosineWaveform(symbol_rate=1e6, num_preamble_symbols=0, num_data_symbols=40, oversampling_factor=8, roll_off=.9)
waveform_generator.num_preamble_symbols = 128
waveform_generator.num_data_symbols = 1024
waveform_generator.modulation_order = 4
waveform_generator.synchronization = SingleCarrierSynchronization()
waveform_generator.channel_estimation = SingleCarrierLeastSquaresChannelEstimation()
waveform_generator.channel_equalization = SingleCarrierZeroForcingChannelEqualization()

device.sampling_rate = waveform_generator.sampling_rate

source = StreamBitsSource(os.path.join(os.path.dirname(__file__), '../resources/leena.raw'))
leena_num_bits = 512 * 512 * 8
image_buffer = np.zeros((512, 512), dtype=np.uint8)
image_buffer[0, 0] = 255

# Add a modem at the simulated device
modem = Modem()
modem.device = device
modem.bits_source = source
modem.waveform_generator = waveform_generator

# Compute number of required frames
bits_per_frame = modem.num_data_bits_per_frame
byte_per_frame = int(bits_per_frame / 8)
num_frames = int(leena_num_bits / bits_per_frame)

plt.ion()
fig, axes = plt.subplots()
image = axes.imshow(image_buffer)

for f in range(num_frames):

    tx_signal, tx_symbols, tx_bits = modem.transmit()
    device_transmissions = device.transmit(clear_cache=True)
    device.receive_signal(device_transmissions[0])  # , snr=8.)
    rx_signal, rx_symbols, data_bits = modem.receive()

    if len(data_bits) > 0:
        image_buffer.flat[f*byte_per_frame:(f+1)*byte_per_frame] = np.packbits(data_bits)
    image.set_data(image_buffer)
    fig.canvas.flush_events()

input("Press Enter to continue...")
plt.close()
