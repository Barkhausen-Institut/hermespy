# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from hermespy.channel import TDL, TDLType
from hermespy.modem import OFDMWaveform, OrthogonalLeastSquaresChannelEstimation, PilotSection, CorrelationSynchronization, ZeroForcingChannelEqualization, PrefixType, SimplexLink, GridResource, GridElement, GuardSection, ElementType, SymbolSection
from hermespy.simulation import Simulation


# Initialize a simulation with two dedicated devices for transmission and reception
carrier_frequency = 3.7e9
simulation = Simulation()
tx_device = simulation.new_device(carrier_frequency=carrier_frequency)
rx_device = simulation.new_device(carrier_frequency=carrier_frequency)

# Assume a 5G TDL channel model
channel = TDL(TDLType.A, 1e-7, doppler_frequency=10)
simulation.set_channel(tx_device, rx_device, channel)

# Configure an orthogonal waveform featuring 128 subcarriers
grid_resources = [
    GridResource(16, PrefixType.CYCLIC, .1, [GridElement(ElementType.DATA, 7), GridElement(ElementType.REFERENCE, 1)]),
    GridResource(128, PrefixType.CYCLIC, .1, [GridElement(ElementType.DATA, 1)]),
]
grid_structure = [
    SymbolSection(64, [0, 1])
]
waveform = OFDMWaveform(
    grid_resources=grid_resources,
    grid_structure=grid_structure,
    num_subcarriers=128,
    subcarrier_spacing=1e3,
)

# Configure the link to connect both devices
link = SimplexLink(waveform=waveform)
link.connect(tx_device, rx_device)

# Configure channel estimation and equalization
waveform.channel_estimation = OrthogonalLeastSquaresChannelEstimation()
waveform.channel_equalization = ZeroForcingChannelEqualization()

# Configure frame synchronization
waveform.pilot_section = PilotSection()
waveform.synchronization = CorrelationSynchronization()

# Visualize the resource grid
waveform.plot_grid()
plt.show()
