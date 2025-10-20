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

from hermespy.channel import TDL, TDLType
from hermespy.fec import Scrambler3GPP
from hermespy.modem import SimplexLink, RootRaisedCosineWaveform, SingleCarrierLeastSquaresChannelEstimation, SingleCarrierZeroForcingChannelEqualization, StreamBitsSource
from hermespy.simulation import SimulationScenario

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


# Create a new HermesPy simulation scenario
scenario = SimulationScenario()

# Create a new simulated device
carrier_frequency = 1e8
bandwidth = 1e6
oversampling_factor = 4
tx_device = scenario.new_device(carrier_frequency=carrier_frequency, bandwidth=bandwidth, oversampling_factor=oversampling_factor)
rx_device = scenario.new_device(carrier_frequency=carrier_frequency, bandwidth=bandwidth, oversampling_factor=oversampling_factor)
scenario.set_channel(rx_device, tx_device, TDL(TDLType.E))

waveform = RootRaisedCosineWaveform(num_preamble_symbols=0 , num_data_symbols=120, roll_off=.9)
waveform.num_preamble_symbols = 128
waveform.num_data_symbols = 2048
waveform.modulation_order = 256
waveform.channel_estimation = SingleCarrierLeastSquaresChannelEstimation()
waveform.channel_equalization = SingleCarrierZeroForcingChannelEqualization()

leena_num_bits = 512 * 512 * 8
image_buffer = np.zeros((512, 512), dtype=np.uint8)

# Add a modem at the simulated device
link = SimplexLink()
link.waveform = waveform
link.encoder_manager.add_encoder(Scrambler3GPP())
link.bits_source = StreamBitsSource(os.path.join(os.path.dirname(__file__), '../resources/leena.raw'))
tx_device.transmitters.add(link)
rx_device.receivers.add(link)

# Compute number of required frames
bits_per_frame = waveform.bits_per_symbol * waveform.num_data_symbols
byte_per_frame = int(bits_per_frame / 8)
num_frames = int(leena_num_bits / bits_per_frame)

plt.ion()
fig, axes = plt.subplots()
image = axes.imshow(image_buffer, cmap='gray', vmin=0, vmax=255)

for f in range(num_frames):
    drop = scenario.drop()
    reception = drop.device_receptions[1].operator_receptions[0]
    if reception.bits.size > 0:
        image_buffer.flat[f*byte_per_frame:(f+1)*byte_per_frame] = np.packbits(reception.bits)
    image.set_data(image_buffer)
    fig.canvas.flush_events()

input("Press Enter to continue...")
plt.close()
