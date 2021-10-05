from simulator_core import Factory
from scenario import Scenario
import numpy as np
import scipy.constants as const
from source.bits_source import BitsSource
from modem import Transmitter, Receiver
from modem.coding import EncoderManager, Encoder, Interleaver, RepetitionEncoder
from modem import RfChain
from modem.rf_chain_models.power_amplifier import PowerAmplifier
from modem.waveform_generator_chirp_fsk import WaveformGeneratorChirpFsk
from modem.precoding import Precoding, Precoder, DFT
from beamformer import Beamformer
from channel import Channel
import matplotlib.pyplot as plt
from ruamel.yaml import YAML, Node
from io import StringIO
import sys

# 8x8 MIMO arrays at 60Ghz
carrier_frequency = 60e9
antenna_spacing = 5 * const.c / carrier_frequency
topology = antenna_spacing * np.array([[x, y, 0.0] for y in range(8) for x in range(8)], dtype=float)

# Initialize an empty scenario
scenario = Scenario()

# Add modems
modem_configuration = {'carrier_frequency': carrier_frequency, 'topology': topology}
transmitterA = scenario.add_transmitter(**modem_configuration)
transmitterB = scenario.add_transmitter(**modem_configuration)
receiverA = scenario.add_receiver(**modem_configuration)
receiverB = scenario.add_receiver(**modem_configuration)

transmitterA.waveform_generator = WaveformGeneratorChirpFsk()
transmitterB.waveform_generator = WaveformGeneratorChirpFsk()
transmitterA.precoding[0] = DFT()
receiverB.rf_chain.power_amplifier = PowerAmplifier()
transmitterA.encoder_manager.add_encoder(Interleaver())
transmitterA.encoder_manager.add_encoder(RepetitionEncoder())

# Configure channels
scenario.channel(transmitterA, receiverA).active = True
scenario.channel(transmitterA, receiverB).active = False
scenario.channel(transmitterB, receiverA).active = False
scenario.channel(transmitterB, receiverB).active = True

# Drop
scenario.init_drop()
# signals = scenario.transmit()

# Print scenario serialization
factory = Factory()

dump = factory.to_str(scenario)
print(dump)

load = factory.from_str(dump)
print(load)
