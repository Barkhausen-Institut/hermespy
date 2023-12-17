# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from hermespy.modem import (
    CustomPilotSymbolSequence,
    RootRaisedCosineWaveform,
    SimplexLink,
    SingleCarrierCorrelationSynchronization,
    SingleCarrierLeastSquaresChannelEstimation,
    SingleCarrierZeroForcingChannelEqualization,
)
from hermespy.simulation import Simulation


# Initialize the waveform description
waveform = RootRaisedCosineWaveform(
    oversampling_factor=4,
    symbol_rate=1e6,
    num_preamble_symbols=16,
    num_data_symbols=32,
    modulation_order=64,
    roll_off=.9,
)

# Configure the waveform's synchronization routine
waveform.synchronization = SingleCarrierCorrelationSynchronization()

# Configure the waveform's channel estimation routine
waveform.channel_estimation = SingleCarrierLeastSquaresChannelEstimation()

# Configure the waveform's channel equalization routine
waveform.channel_equalization = SingleCarrierZeroForcingChannelEqualization()

# Configure the waveform's pilot symbol sequence
waveform.pilot_symbol_sequence = CustomPilotSymbolSequence(
    np.exp(.25j * np.pi * np.array([0, 1, 2, 3]))
)

# Initialize a new simulation considering a single device
simulation = Simulation()
tx_device = simulation.new_device(carrier_frequency=1e10)
rx_device = simulation.new_device(carrier_frequency=1e10)

# Configure the link modeling the devices' transmit DSP
link = SimplexLink(tx_device, rx_device)
link.waveform = waveform

# Generate a transmission to be received by the modem
transmission = tx_device.transmit()
rx_signal = transmission.mixed_signal

# Generate a single reception of the modem
modem_reception = link.receive(rx_signal)
modem_reception.signal.plot(title='Modem Base-Band Waveform')
modem_reception.symbols.plot_constellation(title='Modem Constellation Diagram')

# Equivalent:
# Generate a single transmission of the device
device_reception = rx_device.receive(transmission)
device_reception.impinging_signals[0].plot(title='Device Base-Band Waveform')
device_reception.operator_inputs[0].plot(title='Modem Base-Band Waveform')

# Equivalent:
# Generate a single simulation drop
simulation_drop = simulation.scenario.drop()
simulation_drop.device_receptions[0].impinging_signals[0].plot(title='Device Base-Band Waveform')

plt.show()
