# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from hermespy.core import AntennaMode
from hermespy.modem import (
    Alamouti,
    SimplexLink,
    RootRaisedCosineWaveform,
)
from hermespy.simulation import (
    Simulation,
    SimulatedUniformArray,
    SimulatedIdealAntenna,
    SingleCarrierIdealChannelEstimation,
)


# Create a new simulation featuring a 2x2 MIMO link between two devices
simulation = Simulation()
tx_device = simulation.new_device(
    antennas=SimulatedUniformArray(SimulatedIdealAntenna(AntennaMode.TX), 0.1, (2,)),
)

rx_device = simulation.new_device(
    antennas=SimulatedUniformArray(SimulatedIdealAntenna(AntennaMode.RX), 0.1, (1,)),
)

# Create a link between the two devices
link = SimplexLink()
link.connect(tx_device, rx_device)

# Configure a single carrier waveform
waveform = RootRaisedCosineWaveform(
    oversampling_factor=4,
    symbol_rate=1e6,
    num_preamble_symbols=16,
    num_data_symbols=32,
    modulation_order=64,
    roll_off=.9,
    channel_estimation=SingleCarrierIdealChannelEstimation(simulation.scenario.channel(tx_device, rx_device), tx_device, rx_device),
)
link.waveform = waveform

# Configure the precoding
link.transmit_symbol_coding[0] = Alamouti()
link.receive_symbol_coding[0] = Alamouti()

# Generate a simulation drop
drop = simulation.scenario.drop()

drop.device_transmissions[0].mixed_signal.plot(title='Transmission')
drop.device_receptions[1].impinging_signals[0].plot(title='Reception')
drop.device_transmissions[0].operator_transmissions[0].symbols.plot_constellation(title='Transmitted Constellation')
drop.device_receptions[1].operator_receptions[0].equalized_symbols.plot_constellation(title='Received Constellation')

plt.show()
