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
transmitterA.precoding[1] = Beamformer()
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


# Simulate a configuration dump
yaml = YAML(typ='unsafe')
yaml.default_flow_style = False
yaml.compact(seq_seq=False, seq_map=False)
yaml.encoding = None

# def strip_python_tags(s):
#     result = []
#
#     for line in s.splitlines():
#
#         idx = line.find(": !<")
#         if idx > -1:
#            line = line[:idx+1]
#
#        idx = line.find("- !<")
#        if idx > -1:
#            line = line[:idx+2] + line[idx+4:-1]
#
#       result.append(line)
#
#   return '\n'.join(result)

serializable_classes = [Scenario, BitsSource, Transmitter, Receiver, EncoderManager, Encoder, RfChain, PowerAmplifier,
                        Beamformer, Channel, WaveformGeneratorChirpFsk, Precoding, Precoder,
                        DFT, Interleaver, RepetitionEncoder]

for serializable_class in serializable_classes:
    yaml.register_class(serializable_class)

stream = StringIO()
yaml.dump(scenario, stream)
print(stream.getvalue())
scenarioImport = yaml.load(stream.getvalue())

factory = Factory()
dump = factory.to_str(scenarioImport)
print(dump)