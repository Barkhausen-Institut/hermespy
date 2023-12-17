# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from hermespy.core import AntennaMode
from hermespy.modem import (
    SimplexLink,
    MaximumRatioCombining,
    RootRaisedCosineWaveform,
)
from hermespy.simulation import (
    Simulation,
    SimulatedUniformArray,
    SimulatedIdealAntenna,
    SingleCarrierIdealChannelEstimation,
)


# Create a new simulation featuring a 1x2 SIMO link between two devices
simulation = Simulation()
tx_device = simulation.new_device()
rx_device = simulation.new_device(
    antennas=SimulatedUniformArray(SimulatedIdealAntenna(AntennaMode.RX), 0.1, (2,)),
)

# Create a link between the two devices
link = SimplexLink(tx_device, rx_device)

# Configure a single carrier waveform
waveform = RootRaisedCosineWaveform(
    oversampling_factor=4,
    symbol_rate=1e6,
    num_preamble_symbols=16,
    num_data_symbols=32,
    modulation_order=64,
    roll_off=.9,
    channel_estimation=SingleCarrierIdealChannelEstimation(tx_device, rx_device),
)
link.waveform = waveform

# Configure the precoding
link.precoding[0] = MaximumRatioCombining()

# Generate a simulation drop
drop = simulation.scenario.drop()

drop.device_transmissions[0].mixed_signal.plot(title='Transmission')
drop.device_receptions[1].impinging_signals[0].plot(title='Reception')
link.transmission.symbols.plot_constellation(title='Transmitted Constellation')
link.reception.equalized_symbols.plot_constellation(title='Received Constellation')

plt.show()
