# -*- coding: utf-8 -*-

from __future__ import annotations
from typing_extensions import override

from hermespy.core import SerializationProcess, DeserializationProcess
from hermespy.tools import db2lin
from hermespy.simulation.rf import (
    ADC,
    DAC,
    RFChain,
    Mixer,
    MixerType,
    PowerAmplifier,
    Source,
    OscillatorPhaseNoise,
)

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class X410(RFChain):
    """Model of an Ettus X410 USRP Software Defined Radio."""

    __min_carrier_frequency = 1e6
    __max_carrier_frequency = 7.2e9
    __max_sampling_rate = 491.52e6  # Sps
    __num_adc_quantization_bits = 12
    __num_dac_quantization_bits = 14
    __phase_noise_1kHz = db2lin(-93)  # dBc/Hz at 1 kHz offset
    __phase_noise_10kHz = db2lin(-101)  # dBc/Hz at 10 kHz offset
    __phase_noise_100kHz = db2lin(-103)  # dBc/Hz at 100 kHz offset
    __noise_density = db2lin(-146) * 1e-3  # -146 dBm/Hz
    __max_tx_power = db2lin(23) * 1e-3  # 23 dBm

    def __init__(
        self,
        carrier_frequency: float,
        num_tx_channels: int = 1,
        num_rx_channels: int = 1,
        tx_gain: float = db2lin(30),
        rx_gain: float = db2lin(30),
        phase_noise: bool = True,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            carrier_frequency: Center frequency of the generated signal in Hz.
            sampling_rate: Sampling rate of the generated signal in Hz.
            num_tx_channels:
                Number of transmit channels to be created.
                The actual hardware has four by default.
            num_rx_channels:
                Number of receive channels to be created.
                The actual hardware has four by default.
            phase_noise:
                Whether to model the phase noise of the oscillators.
                Enabled by default.
            seed: Seed with which to initialize the block's random state.
        """

        # Initialize base class
        RFChain.__init__(self, seed)

        # Initialize class attributes
        self.__carrier_frequency = carrier_frequency

        # Build the phase noise model
        pn: OscillatorPhaseNoise | None = None
        if phase_noise:
            pn = OscillatorPhaseNoise.FromPSD(
                [1e3, 1e4, 1e5],
                [self.__phase_noise_1kHz, self.__phase_noise_10kHz, self.__phase_noise_100kHz],
            )

        # Build TX channels
        for _ in range(num_tx_channels):
            dac = self.new_block(DAC, num_quantization_bits=self.__num_dac_quantization_bits)
            tx_lo = self.new_block(Source, phase_noise=pn, carrier_frequency=carrier_frequency)
            tx_mixer = self.new_block(Mixer, mixer_type=MixerType.UP)
            tx_pa = self.new_block(PowerAmplifier, gain=tx_gain)

            self.connect(dac.port("o"), tx_mixer.port("i"))
            self.connect(tx_lo.port("o"), tx_mixer.port("lo"))
            self.connect(tx_mixer.port("o"), tx_pa.port("i"))

        # Build RX channels
        for _ in range(num_rx_channels):
            adc = self.new_block(ADC, num_quantization_bits=self.__num_adc_quantization_bits)
            rx_lo = self.new_block(Source, phase_noise=pn, carrier_frequency=carrier_frequency)
            rx_mixer = self.new_block(Mixer, mixer_type=MixerType.DOWN)
            rx_lna = self.new_block(PowerAmplifier, gain=rx_gain)

            self.connect(adc.port("i"), rx_mixer.port("o"))
            self.connect(rx_mixer.port("lo"), rx_lo.port("o"))
            self.connect(rx_mixer.port("i"), rx_lna.port("o"))

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_floating(self.__carrier_frequency, "carrier_frequency")
        if self.seed is not None:
            process.serialize_integer(self.seed, "seed")

    @classmethod
    @override
    def Deserialize(cls: type[X410], process: DeserializationProcess) -> X410:
        carrier_frequency = process.deserialize_floating("carrier_frequency")
        seed = process.deserialize_integer("seed", None)
        return cls(carrier_frequency=carrier_frequency, seed=seed)
