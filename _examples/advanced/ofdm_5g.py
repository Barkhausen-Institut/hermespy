# -*- coding: utf-8 -*-
#
# In this example we simulate a 5G signal.
# 
# We model the DL of a frame with a TDD slot format 57, DDDDFFUDDDDFFU,
# with 30kHz subcarrier spacing, using 2100 subcarriers with 4-PSK.
# 
# In each drop, 2 time slots are transmitted.
# 
# An LDPC code with rate R=1/2 and (encoded) block length 128 is considered.
# 
# A 5G-TDL channel model type E, with 100 ns rms delay spread is considered.
# Velocity is 10ms with carrier frequency 3GHz.

import os.path as path

import matplotlib.pyplot as plt

from hermespy.core import dB
from hermespy.simulation import (
    Simulation,
    EBN0,
    OFDMIdealChannelEstimation,
)
from hermespy.channel import TDL, TDLType
from hermespy.modem import (
    SimplexLink,
    OFDMWaveform,
    GridResource,
    GridElement,
    ElementType,
    PrefixType,
    SymbolSection,
    GuardSection,
    BitErrorEvaluator,
    OrthogonalZeroForcingChannelEqualization,
)
from hermespy.fec import LDPCCoding

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "André Noll-Barreto", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


# Initialize a simulation with two devices at 3 GHz.
simulation = Simulation()
cf = 3e9
tx_device = simulation.new_device(carrier_frequency=cf)
rx_device = simulation.new_device(carrier_frequency=cf)

# Configure a 5G-TDL channel model type E with 100 ns rms delay spread.
tdl_channel = TDL(TDLType.E, 100e-9)
simulation.set_channel(tx_device, rx_device, tdl_channel)

# Specify the OFDM waveform
tdd_57 = OFDMWaveform(
    num_subcarriers=4096,
    dc_suppression=False,
    modulation_order=4,
    subcarrier_spacing=30e3,
    grid_resources=[
        GridResource(1, PrefixType.CYCLIC, 0.0703125, [
            GridElement(ElementType.NULL, 998),
            GridElement(ElementType.DATA, 2100),
            GridElement(ElementType.NULL, 998)
        ]),
        GridResource(1, PrefixType.CYCLIC, 0.078125, [
            GridElement(ElementType.NULL, 998),
            GridElement(ElementType.DATA, 2100),
            GridElement(ElementType.NULL, 998)
        ]),
    ],
    grid_structure=[
        # DDDD
        SymbolSection(1, [1, 0, 0, 0]),
        # FFU
        GuardSection(35.677083e-6, 3),
        # DDDD
        SymbolSection(1, [1, 0, 0, 0]),
        # FFU
        GuardSection(35.677083e-6, 3),
    ]
)

# Link the devices with the waveform
link = SimplexLink(waveform=tdd_57)
link.connect(tx_device, rx_device)

# Configure an LDPC code with rate R=1/2 and block length 128
ldpc_code = path.join(path.dirname(path.abspath(__file__)), '..', 'resources', 'ofdm_ldpc.alist')
link.encoder_manager.add_encoder(LDPCCoding(100, ldpc_code, "", True, 10))

# Assume ideal channel state information and zero-forcing equalization
link.waveform.channel_estimation = OFDMIdealChannelEstimation(tdl_channel, tx_device, rx_device)
link.waveform.channel_equalization = OrthogonalZeroForcingChannelEqualization()

# Evaluate the link's bit error rate during simulation runtime
simulation.add_evaluator(BitErrorEvaluator(link, link))

# Sweep over the link's receive Eb/N0 from 0 dB to 20 dB
rx_device.noise_level = EBN0(link)
simulation.new_dimension("noise_level", dB(range(0, 21, 2)), rx_device)

# Run the simulation and collect 1000 samples per SNR point
simulation.num_samples = 1000
simulation.min_num_samples = 100
result = simulation.run()
result.plot()
plt.show()
