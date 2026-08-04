# -*- coding: utf-8 -*-

from __future__ import annotations
from typing_extensions import override

import numpy as np

from hermespy.core import SerializationProcess, DeserializationProcess
from hermespy.tools import db2lin
from hermespy.simulation.rf import (
    ADC,
    DAC,
    RFChain,
    Mixer,
    MixerType,
    N0,
    PowerAmplifier,
    Source,
    OscillatorPhaseNoise,
)

__author__ = "Jan Adler"
__copyright__ = "Copyright 2026, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.6.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class X410(RFChain):
    """Model of an Ettus X410 USRP Software Defined Radio."""

    __min_carrier_frequency = 1e6
    __max_carrier_frequency = 7.2e9
    __max_sampling_rate = 491.52e6  # Sps

    __phase_noise_1kHz = db2lin(-93)  # dBc/Hz at 1 kHz offset
    __phase_noise_10kHz = db2lin(-101)  # dBc/Hz at 10 kHz offset
    __phase_noise_100kHz = db2lin(-103)  # dBc/Hz at 100 kHz offset
    __noise_density = db2lin(-146) * 1e-3  # -146 dBm/Hz
    __max_tx_power = db2lin(23) * 1e-3  # 23 dBm
    __dac_max_output_power = 1e-3  # Equivalent Voltage of 0dBm at 50 Ohm
    __adc_max_input_power = 1e-3  # Equivalent Voltage of 0dBm at 50 Ohm
    __dac_num_quantization_bits = 14
    __adc_num_quantization_bits = 12

    # Lookup table for the maximum transmit power in dBm
    # Rough conversion from https://www.ni.com/docs/en-US/bundle/ettus-usrp-x410-specs/
    # Values are assumed to be the output power at the maximum of 60dB nominal transmit gain
    # Note that this table assumes a continuous wave, typical benchmarks seem to indicate ~20 dB less for bursts
    __max_tx_power_lookup_dBm: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.array([
        [0, 19.0],
        [0.5, 22.0],
        [1.0, 22.5],
        [4.5, 21.0],
        [5.0, 18.0],
        [6.0, 17.5],
        [6.5, 15.0],
        [7.5, 12.5],
        [8.0, 8.0],
        [9.0, 0.0],  # out of operation range
    ], dtype=np.float64)

    def __init__(
        self,
        carrier_frequency: float,
        num_tx_channels: int = 1,
        num_rx_channels: int = 1,
        tx_gain: float = db2lin(48.0),
        rx_gain: float = db2lin(48.0),
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
            tx_gain:
                Nominal configured transmit gain in dB.
                Note that the actual transmit power varies based on the selected `carrier_frequency`.
            rx_gain:
                Nominal configured receive gain in dB.
                Note that the actual receive power varies based on the selected `carrier_frequency`.
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
        dac = self.new_block(DAC, num_ports=num_tx_channels, num_quantization_bits=self.__dac_num_quantization_bits, max_output_power=self.__dac_max_output_power)
        for n in range(num_tx_channels):
            tx_lo = self.new_block(Source, phase_noise=pn, carrier_frequency=carrier_frequency)
            tx_mixer = self.new_block(Mixer, mixer_type=MixerType.UP)
            tx_pa = self.new_block(PowerAmplifier, gain=tx_gain)

            self.connect(dac.port("o")[n], tx_mixer.port("i"))
            self.connect(tx_lo.port("o"), tx_mixer.port("lo"))
            self.connect(tx_mixer.port("o"), tx_pa.port("i"))

        # Build RX channels
        adc = self.new_block(ADC, num_ports=num_rx_channels, num_quantization_bits=self.__adc_num_quantization_bits, max_input_power=self.__adc_max_input_power, noise_level=N0(self.__noise_density))
        for m in range(num_rx_channels):
            rx_lo = self.new_block(Source, phase_noise=pn, carrier_frequency=carrier_frequency)
            rx_mixer = self.new_block(Mixer, mixer_type=MixerType.DOWN)
            rx_lna = self.new_block(PowerAmplifier, gain=rx_gain, noise_level=N0(self.__noise_density))

            self.connect(adc.port("i")[m], rx_mixer.port("o"))
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
