# -*- coding: utf-8 -*-
#
# In this example the propagation FSK modulated chirps (similar to LORA) is simualted.
# A bandwidth of B = 500kHz is considered, with spreading factor SF = 8,
# This corresponds to M = 2^SF = 256 different initial frequencies,
# spaced by B / M = 1953.125Hz .
# The symbol rate (chirp duration) is given by Ts = 2^SF/BW = .512 ms
# Data is uncoded, and the data rate is
# SF * BW / 2 **SF = log2(M) / Ts = 15625 kbps
# 
# Frames have 160 bits, i.e., 20 FSK symbols.
# 
# A carrier frequency of 865MHz is considered, with Rayleigh fading and a speed
# of 10m/s


import matplotlib.pyplot as plt

from hermespy.core import dB
from hermespy.fec import TurboCoding
from hermespy.modem import (
    SimplexLink,
    ChirpFSKWaveform,
    BitErrorEvaluator,
    ThroughputEvaluator,
)
from hermespy.simulation import (
    Simulation,
    EBN0,
)
from hermespy.channel import (
    MultipathFadingChannel,
    StandardAntennaCorrelation,
    CorrelationType,
)

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "André Noll-Barreto", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


# Initialize a simulation considering two devices operating at 865 MHz
simulation = Simulation()
cf = 865e6
bandwidth = 500e3
tx_device = simulation.new_device(carrier_frequency=cf, bandwidth=bandwidth)
rx_device = simulation.new_device(carrier_frequency=cf, bandwidth=bandwidth)

# The channel model is a classical Rayleigh fading
simulation.set_channel(tx_device, rx_device, MultipathFadingChannel(
    delays=[0],
    power_profile=[dB(0)],
    rice_factors=[float('inf')],
    antenna_correlation=StandardAntennaCorrelation(CorrelationType.MEDIUM),
))

# Connect the devices with a simplex link
link = SimplexLink(waveform=ChirpFSKWaveform(
    chirp_duration=512e-6,
    freq_difference=1953.125,
    num_data_chirps=20,
    modulation_order=256,
    guard_interval=0.0,
))
tx_device.transmitters.add(link)
rx_device.receivers.add(link)

# Additionally, add a forward error correction algorithm to the link
link.encoder_manager.add_encoder(TurboCoding(40, 13, 15, 100))

# Evaluate bit error rates and throughtput during the simulation
simulation.add_evaluator(BitErrorEvaluator(link, link))
simulation.add_evaluator(ThroughputEvaluator(link, link))

# Sweep over the EBN0 SNR at the receiving device during simulation runtime
rx_device.noise_level = EBN0(link)
simulation.new_dimension('noise_level', dB(range(-2, 11)), rx_device)

# Collect 1000 data points
simulation.num_samples = 1000

# Run the simulation and plot the results
result = simulation.run()
result.plot()
plt.show()
