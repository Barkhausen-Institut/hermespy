# -*- coding: utf-8 -*-
#
# Simulation of a 5G NR link numerology #1 link.
#

import matplotlib.pyplot as plt

from hermespy.core import dB
from hermespy.simulation import (
    Simulation,
    EBN0,
)
from hermespy.channel import TDL, TDLType
from hermespy.modem import (
    BitErrorEvaluator,
    ConstellationEVM,
    nr_bandwidth,
    NRSlotLink,
)

__author__ = "Jan Adler"
__copyright__ = "Copyright 2026, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "André Noll-Barreto", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "1.6.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


# Initialize a simulation with two devices at 3 GHz.
simulation = Simulation()
cf = 3e9
numerology = 1
bandwidth = nr_bandwidth(numerology)
tx_device = simulation.new_device(carrier_frequency=cf, bandwidth=bandwidth)
rx_device = simulation.new_device(carrier_frequency=cf, bandwidth=bandwidth)

# Configure a 5G-TDL channel model type E with 100 ns rms delay spread.
tdl_channel = TDL(TDLType.E, 100e-9)
simulation.set_channel(tx_device, rx_device, tdl_channel)

# Link the devices
link = NRSlotLink()
link.waveform.modulation_order = 4  # QPSK modulation
link.connect(tx_device, rx_device)

# Evaluate the link's bit error rate and error vector magnitude during simulation runtime
simulation.add_evaluator(BitErrorEvaluator(link, link))
simulation.add_evaluator(ConstellationEVM(link, link))

# Sweep over the link's receive Eb/N0 from 0 dB to 20 dB
rx_device.noise_level = EBN0(link)
simulation.new_dimension("noise_level", dB(range(0, 21, 2)), rx_device)

# Run the simulation and collect 1000 samples per SNR point
simulation.num_samples = 1000
result = simulation.run()
result.plot()
plt.show()
