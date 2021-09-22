from scenario import Scenario
from modem import TransmissionMode
import numpy as np
import scipy.constants as const
from beamformer import ConventionalBeamformer, TransmissionDirection
from source.bits_source import BitsSource
from modem import Transmitter, Receiver
from modem.coding import EncoderManager, Encoder
from modem import RfChain
from modem.rf_chain_models.power_amplifier import PowerAmplifier
from modem.waveform_generator_chirp_fsk import WaveformGeneratorChirpFsk
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
receiverB.rf_chain.power_amplifier = PowerAmplifier()


# Configure channels
scenario.channel(transmitterA, receiverA).active = True
scenario.channel(transmitterA, receiverB).active = True
scenario.channel(transmitterB, receiverB).active = True

# Add a conventional beamformer to transmitter A, steering towards transmitter B
conventional_beamformer = transmitterA.configure_beamformer(ConventionalBeamformer, focused_modem=receiverA)

# Simulate a configuration dump

yaml = YAML(typ='safe')

serializable_classes = [Scenario, BitsSource, Transmitter, Receiver, EncoderManager, Encoder, RfChain, PowerAmplifier,
                        ConventionalBeamformer, Channel]

for serializable_class in serializable_classes:
    yaml.register_class(serializable_class)


stream = StringIO()
yaml.dump(scenario, stream)
print(stream.getvalue())
scenarioImport = yaml.load(stream.getvalue())
stream.flush()
yaml.dump(scenarioImport, stream)
print(stream.getvalue())
