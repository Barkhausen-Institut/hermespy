# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from hermespy.modem import (
    TransmittingModem as BaseModem,
    DFT,
    OFDMWaveform,
    GridResource,
    GridElement,
    SymbolSection,
    ElementType,
)
from hermespy.simulation import Simulation


# Create a new simulation featuring a single device
simulation = Simulation()
device = simulation.new_device()

# Create a new modem
# This should be replaced by a BaseModem implementation such as DuplexLink
modem = BaseModem()
device.add_dsp(modem)

# Configure an OFDM waveform
modem.waveform = OFDMWaveform(
    oversampling_factor=2,
    num_subcarriers=1024,
    grid_resources=[GridResource(
        repetitions=100,
        prefix_ratio=0.0684,
        elements=[
            GridElement(ElementType.DATA, 9),
            GridElement(ElementType.REFERENCE, 1),
        ]
    )],
    grid_structure=[SymbolSection(3, [0])]
)

# Configure the precoding
modem.transmit_symbol_coding[0] = DFT()

# Generate and plot a transmission
transmission = device.transmit()
transmission.mixed_signal.plot(title='OFDM Transmission')
plt.show()
