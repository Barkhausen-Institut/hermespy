from scenario import Scenario
from modem import TransmissionMode
import numpy as np
import scipy.constants as const
from beamformer import ConventionalBeamformer, TransmissionDirection
from source.bits_source import BitsSource
from modem import Transmitter, Receiver
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

# Configure channels
scenario.channel(transmitterA, receiverA).active = True
scenario.channel(transmitterA, receiverB).active = True
scenario.channel(transmitterB, receiverB).active = True

# Add a conventional beamformer to transmitter A, steering towards transmitter B
conventional_beamformer = transmitterA.configure_beamformer(ConventionalBeamformer, focused_modem=receiverA)

# Simulate a configuration dump

yaml = YAML(typ='safe')
yaml.register_class(Scenario)
yaml.register_class(BitsSource)
yaml.register_class(Transmitter)
yaml.register_class(Receiver)

stream = StringIO()
yaml.dump(scenario, stream)
scenario = yaml.load(stream.getvalue())
print(scenario.transmitters)
