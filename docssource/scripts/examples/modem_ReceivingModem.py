# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from hermespy.beamforming import ConventionalBeamformer
from hermespy.fec import RepetitionEncoder, BlockInterleaver
from hermespy.modem import (
    DFT,
    ReceivingModem,
    RootRaisedCosineWaveform,
    SingleCarrierCorrelationSynchronization,
    SingleCarrierLeastSquaresChannelEstimation,
    SingleCarrierZeroForcingChannelEqualization,
    TransmittingModem,
)
from hermespy.simulation import Simulation


# Initialize a new simulation considering a single device
simulation = Simulation()
device = simulation.new_device(carrier_frequency=1e10, oversampling_factor=4, bandwidth=1e6)

# Configure the modem modeling the device's transmit DSP
rx_modem = ReceivingModem()
device.receivers.add(rx_modem)

# Configure the modem's waveform
waveform = RootRaisedCosineWaveform(
    num_preamble_symbols=16,
    num_data_symbols=32,
    modulation_order=64,
)
rx_modem.waveform = waveform

# Configure the waveform's synchronization routine
rx_modem.waveform.synchronization = SingleCarrierCorrelationSynchronization()

# Add a custom stream precoding to the modem
rx_modem.receive_signal_coding[0] = ConventionalBeamformer()

# Add a custom symbol precoding to the modem
rx_modem.receive_symbol_coding[0] = DFT()

# Configure the waveform's channel estimation routine
rx_modem.waveform.channel_estimation = SingleCarrierLeastSquaresChannelEstimation()

# Configure the waveform's channel equalization routine
rx_modem.waveform.channel_equalization = SingleCarrierZeroForcingChannelEqualization()

# Add forward error correction encodings to the transmitted bit stream
rx_modem.encoder_manager.add_encoder(RepetitionEncoder(32, 3))
rx_modem.encoder_manager.add_encoder(BlockInterleaver(192, 32))

# Generate a transmission to be received by the modem
tx_modem = TransmittingModem()
tx_modem.waveform = waveform
tx_modem.encoder_manager.add_encoder(RepetitionEncoder(32, 3))
tx_modem.encoder_manager.add_encoder(BlockInterleaver(192, 32))
tx_modem.transmit_symbol_coding[0] = DFT()
device.transmitters.add(tx_modem)
transmission = device.transmit()

# Generate a single reception of the modem
modem_reception = rx_modem.receive(transmission.mixed_signal, device.state())
modem_reception.signal.plot(title='Modem Base-Band Waveform')
modem_reception.symbols.plot_constellation(title='Modem Constellation Diagram')

# Equivalent:
# Generate a single transmission of the device
device_reception = device.receive(transmission)
device_reception.impinging_signals[0].plot(title='Device Base-Band Waveform')
device_reception.operator_inputs[0].plot(title='Modem Base-Band Waveform')

# Equivalent:
# Generate a single simulation drop
simulation_drop = simulation.scenario.drop()
simulation_drop.device_receptions[0].impinging_signals[0].plot(title='Device Base-Band Waveform')

plt.show()
