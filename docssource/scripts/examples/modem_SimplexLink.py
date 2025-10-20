# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from hermespy.beamforming import ConventionalBeamformer
from hermespy.fec import RepetitionEncoder, BlockInterleaver
from hermespy.modem import (
    DFT,
    RandomBitsSource,
    RootRaisedCosineWaveform,
    SimplexLink,
    SingleCarrierCorrelationSynchronization,
    SingleCarrierLeastSquaresChannelEstimation,
    SingleCarrierZeroForcingChannelEqualization,
)
from hermespy.simulation import Simulation


# Initialize a new simulation considering a single device
simulation = Simulation()
tx_device = simulation.new_device(carrier_frequency=1e10, oversampling_factor=4, bandwidth=1e6)
rx_device = simulation.new_device(carrier_frequency=1e10, oversampling_factor=4, bandwidth=1e6)

# Configure the links's waveform
waveform = RootRaisedCosineWaveform(
    num_preamble_symbols=16,
    num_data_symbols=32,
    modulation_order=64,
)

# Configure the link to connect both devices
link = SimplexLink(waveform=waveform)
link.connect(tx_device, rx_device)

# Configure a custom bits source for the modem
link.bits_source = RandomBitsSource(seed=42)

# Configure the waveform's synchronization routine
link.waveform.synchronization = SingleCarrierCorrelationSynchronization()

# Add a custom stream precoding to the modem
link.transmit_signal_coding[0] = ConventionalBeamformer()
link.receive_signal_coding[0] = ConventionalBeamformer()

# Add a custom symbol precoding to the modem
link.transmit_symbol_coding[0] = DFT()
link.receive_symbol_coding[0] = DFT()

# Configure the waveform's channel estimation routine
link.waveform.channel_estimation = SingleCarrierLeastSquaresChannelEstimation()

# Configure the waveform's channel equalization routine
link.waveform.channel_equalization = SingleCarrierZeroForcingChannelEqualization()

# Add forward error correction encodings to the transmitted bit stream
link.encoder_manager.add_encoder(RepetitionEncoder(32, 3))
link.encoder_manager.add_encoder(BlockInterleaver(192, 32))

# Generate a transmission to be received by the modem
transmission = tx_device.transmit()
rx_signal = transmission.mixed_signal

# Generate a single reception of the modem
modem_reception = rx_device.receive(transmission).operator_receptions[0]
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
