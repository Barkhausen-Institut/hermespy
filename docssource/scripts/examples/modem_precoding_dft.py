# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from hermespy.core import dB
from hermespy.modem import (
    TransmittingModem as BaseModem,
    DFT,
    OFDMWaveform,
    FrameResource,
    FrameElement,
    FrameSymbolSection,
    ElementType,
)
from hermespy.simulation import Simulation


# Create a new simulation featuring a single device
simulation = Simulation()
device = simulation.new_device()

# Create a new modem
# This should be replaced by a BaseModem implementation such as DuplexLink
modem = BaseModem(device=device)

# Configure an OFDM waveform
modem.waveform = OFDMWaveform(
    oversampling_factor=2,
    num_subcarriers=1024,
    resources=[FrameResource(
        repetitions=100,
        prefix_ratio=0.0684,
        elements=[
            FrameElement(ElementType.DATA, 9),
            FrameElement(ElementType.REFERENCE, 1),
        ]
    )],
    structure=[FrameSymbolSection(3, [0])]
)

# Configure the precoding
modem.precoding[0] = DFT()

# Generate and plot a transmission
transmission = device.transmit()
transmission.mixed_signal.plot(title='OFDM Transmission')
plt.show()
