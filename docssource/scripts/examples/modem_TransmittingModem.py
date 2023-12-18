# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from hermespy.beamforming import ConventionalBeamformer
from hermespy.fec import RepetitionEncoder, BlockInterleaver
from hermespy.modem import DFT, RandomBitsSource, TransmittingModem, RootRaisedCosineWaveform
from hermespy.simulation import Simulation


# Initialize a new simulation considering a single device
simulation = Simulation()
device = simulation.new_device(carrier_frequency=1e10)

# Configure the modem modeling the device's transmit DSP
tx_modem = TransmittingModem(device=device)

# Configure the modem's waveform
waveform = RootRaisedCosineWaveform(
    oversampling_factor=4,
    symbol_rate=1e6,
    num_preamble_symbols=16,
    num_data_symbols=32,
    modulation_order=64,
)
tx_modem.waveform = waveform

# Add a custom bits source to the modem
tx_modem.bits_source = RandomBitsSource(seed=42)

# Add forward error correction encodings to the transmitted bit stream
tx_modem.encoder_manager.add_encoder(RepetitionEncoder(32, 3))
tx_modem.encoder_manager.add_encoder(BlockInterleaver(192, 32))

# Add a custom symbol precoding to the modem
tx_modem.precoding[0] = DFT()

# Add a custom stream precoding to the modem
tx_modem.transmit_stream_coding[0] = ConventionalBeamformer()

# Generate a single transmission of the modem
modem_transmission = tx_modem.transmit()
modem_transmission.signal.plot(title='Modem Base-Band Waveform')
modem_transmission.symbols.plot_constellation(title='Modem Constellation Diagram')

# Equivalent:
# Generate a single transmission of the device
device_transmission = device.transmit()
device_transmission.mixed_signal.plot(title='Device Base-Band Waveform')
device_transmission.operator_transmissions[0].signal.plot(title='Modem Base-Band Waveform')

# Equivalent:
# Generate a single simulation drop
simulation_drop = simulation.scenario.drop()
simulation_drop.device_transmissions[0].mixed_signal.plot(title='Device Base-Band Waveform')

plt.show()
