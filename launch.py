from scenario import Scenario
from modem import TransmissionMode
import numpy as np
import scipy.constants as const
from beamformer import ConventionalBeamformer, TransmissionDirection
import matplotlib.pyplot as plt

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

# Simulate a 1ms transmission
scenario.init_drop()
transmitted_signal = scenario.transmit(0.001)

"""    
        def __init__(self, param: P, source: BitsSource,
                 random_number_gen: rnd.RandomState, tx_modem=None) -> None:
        self.param = param
        self.source = source

        self.encoder_factory = EncoderFactory()
        self.encoder_manager = EncoderManager()

        for encoding_type, encoding_params in zip(
                            self.param.encoding_type, self.param.encoding_params):
            encoder: Encoder = self.encoder_factory.get_encoder(
                encoding_params, encoding_type,
                self.param.technology.bits_in_frame)
            self.encoder_manager.add_encoder(encoder)

        self.waveform_generator: Any
        if isinstance(param.technology, ParametersPskQam):
            self.waveform_generator = WaveformGeneratorPskQam(param.technology)
        elif isinstance(param.technology, ParametersChirpFsk):
            self.waveform_generator = WaveformGeneratorChirpFsk(param.technology)
        elif isinstance(param.technology, ParametersOfdm):
            self.waveform_generator = WaveformGeneratorOfdm(
                param.technology, random_number_gen)
        else:
            raise ValueError(
                "invalid technology in constructor of Modem class")
        # if this is a received modem, link to tx modem must be provided
        self._paired_tx_modem = tx_modem
        self.power_factor = 1.  # if this is a transmit modem, signal is scaled to the desired power, depending on the
        # current power factor

        self.rf_chain = RfChain(param.rf_chain, self.waveform_generator.get_power(), random_number_gen,)
"""