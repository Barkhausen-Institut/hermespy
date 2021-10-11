from simulator_core import Factory
from scenario import Scenario
import numpy as np
import scipy.constants as const
from source.bits_source import BitsSource
from modem import Transmitter, Receiver
from coding import EncoderManager, Encoder, Interleaver, RepetitionEncoder, LDPC
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
import os
from simulator_core import Simulation

# 8x8 MIMO arrays at 60Ghz
carrier_frequency = 60e9
antenna_spacing = 5 * const.c / carrier_frequency
# topology = antenna_spacing * np.array([[x, y, 0.0] for y in range(8) for x in range(8)], dtype=float)
topology = antenna_spacing * np.array([[0, 0, 0]], dtype=float)
# Initialize an empty scenario
scenario = Scenario()

# Add modems
modem_configuration = {'carrier_frequency': carrier_frequency, 'topology': topology}
transmitterA = scenario.add_transmitter(**modem_configuration)
transmitterA.encoder_manager.add_encoder(Interleaver())
#transmitterB = scenario.add_transmitter(**modem_configuration)
receiverA = scenario.add_receiver(**modem_configuration)
#receiverB = scenario.add_receiver(**modem_configuration)

transmitterA.waveform_generator = WaveformGeneratorChirpFsk(num_data_chirps=4)
#transmitterB.waveform_generator = WaveformGeneratorChirpFsk(num_data_chirps=2)
receiverA.waveform_generator = WaveformGeneratorChirpFsk(num_data_chirps=4)
#receiverB.waveform_generator = WaveformGeneratorChirpFsk()
#transmitterA.encoder_manager.add_encoder(Interleaver())
#transmitterA.encoder_manager.add_encoder(RepetitionEncoder())
#transmitterB.encoder_manager.add_encoder(LDPC())

# Configure channels
scenario.channel(transmitterA, receiverA).active = True
#scenario.channel(transmitterA, receiverB).active = False
#scenario.channel(transmitterB, receiverA).active = False
#scenario.channel(transmitterB, receiverB).active = True

# Drop
simulation = Simulation()
simulation.add_scenario(scenario)
#simulation.run()

# Print scenario serialization
factory = Factory()
executable = factory.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "_examples", "_yaml"))

print(factory.to_str(scenario))
