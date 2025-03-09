# -*- coding: utf-8 -*-
#
# In this example we simulate the interference between an OFDM signal and a
# single-carrier signal in an AWGN channel.
# 
# The OFDM signal has the same numerology as an LTE system with 20 MHz bandwidth,
# in which the first 8 sub-frames are used.
# However, the bits are uncoded. The carrier frequency is 3.5 GHz.
# 
# A single carrier 64-QAM interferer with 6 Mbps is also transmitting at 3.501 GHz

from os import environ

import matplotlib.pyplot as plt

from hermespy.core import dB
from hermespy.simulation import (
    Simulation,
    EBN0,
)
from hermespy.modem import (
    SimplexLink,
    TransmittingModem,
    OFDMWaveform,
    GridResource,
    GridElement,
    ElementType,
    PrefixType,
    SymbolSection,
    GuardSection,
    RootRaisedCosineWaveform,
    BitErrorEvaluator,
)

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "André Noll-Barreto", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


# This simulation can be very memory hungry
# Disable ray's memory monitor to avoid task killing
environ["RAY_memory_monitor_refresh_ms"] = "0"


# Initialize a new simulation considering three devices:
# Two devices operating at 3.5 GHz and one at 3.501 GHz
simulation = Simulation(num_actors=1)
tx_device = simulation.new_device(carrier_frequency=3.5e9)
rx_device = simulation.new_device(carrier_frequency=3.5e9)
interferer = simulation.new_device(carrier_frequency=3.501e9)

# Configure an OFDM link between the first two devices
link = SimplexLink(waveform=OFDMWaveform(
    modulation_order=16,
    subcarrier_spacing=15e3,
    dc_suppression=False,
    num_subcarriers=2048,
    grid_resources=[
        GridResource(200, PrefixType.CYCLIC, .078125, [
            GridElement(ElementType.REFERENCE, 1),
            GridElement(ElementType.DATA, 5),
        ]),
        GridResource(1200, PrefixType.CYCLIC, 0.0703125, [
            GridElement(ElementType.DATA, 1200),
        ]),
        GridResource(100, PrefixType.CYCLIC, 0.0703125, [
            GridElement(ElementType.DATA, 3),
            GridElement(ElementType.REFERENCE, 1),
            GridElement(ElementType.DATA, 5),
            GridElement(ElementType.REFERENCE, 1),
            GridElement(ElementType.DATA, 2),
        ]),
    ],
    grid_structure=[
        SymbolSection(16, [0, 1, 1, 1, 2, 1, 1]),
        GuardSection(1e-6),
    ],
))
link.connect(tx_device, rx_device)

# Configure an interfering single carrier transmitter
interferer_dsp = TransmittingModem(waveform=RootRaisedCosineWaveform(
    modulation_order=64,
    symbol_rate=link.sampling_rate / 4,
    oversampling_factor=4,
    num_preamble_symbols=16,
    num_data_symbols=1024,    
))
interferer.transmitters.add(interferer_dsp)


# Evaluate the link's bit error rate during simulation runtime
simulation.add_evaluator(BitErrorEvaluator(link, link))

# Sweep over the transmitting power of the interfering signal
# and the link's Eb/N0 from 0 dB to 20 dB
rx_device.noise_level = EBN0(link)
simulation.new_dimension("noise_level", dB(range(0, 21)), rx_device)
simulation.new_dimension('power', dB([-30, -20, -10, 0]), interferer)

result = simulation.run()
result.plot()
plt.show()
