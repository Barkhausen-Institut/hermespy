# -*- coding: utf-8 -*-

from __future__ import annotations
from typing_extensions import override

import numpy as np

from ..noise import (
    NoiseRealization,
    NoiseLevel,
    NoiseModel,
    NoPhaseNoise,
    PhaseNoise,
    PhaseNoiseRealization,
)
from ..block import RFBlock, RFBlockRealization, RFBlockPort, RFBlockPortType
from ..signal import RFSignal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SourceRealization(RFBlockRealization):
    """Realization of a source block."""

    __carrier_frequency: float
    __phase_noise: PhaseNoiseRealization
    __amplitude: float

    def __init__(
        self,
        carrier_frequency: float,
        phase_noise: PhaseNoiseRealization,
        amplitude: float,
        bandwidth: float,
        oversampling_factor: int,
        thermal_noise: NoiseRealization,
    ) -> None:
        """
        Args:
            carrier_frequency: Center frequency of the generated signal in Hz.
            amplitude: Amplitude of the generated signal.
            phase_noise: Phase noise realization applied to the source.
            bandwidth: Bandwidth of the block in Hz.
            oversampling_factor: Oversampling factor of the modeling in Hz.
            thermal_noise: Thermal noise realization applied to the source.
        """

        # Initialize base class
        RFBlockRealization.__init__(self, bandwidth, oversampling_factor, thermal_noise)

        # Store attributes
        self.__carrier_frequency = carrier_frequency
        self.__phase_noise = phase_noise
        self.__amplitude = amplitude

    @property
    def carrier_frequency(self) -> float:
        """Center frequency of the generated signal in Hz.

        If zero, the device's target carrier frequency will be generated.

        Raises:
            ValueError: If the carrier frequency is negative.
        """

        return self.__carrier_frequency

    @property
    def phase_noise(self) -> PhaseNoiseRealization:
        """Phase noise realization applied to the source."""

        return self.__phase_noise

    @property
    def amplitude(self) -> float:
        """Amplitude of the generated signal."""

        return self.__amplitude


class Source(RFBlock):
    """Model of an imperferct frequency source."""

    __carrier_frequency: float
    __amplitude: float
    __o: RFBlockPort
    __phase_noise: PhaseNoise

    def __init__(
        self,
        carrier_frequency: float = 0.0,
        phase_noise: PhaseNoise | None = None,
        amplitude: float = 1.0,
        noise_model: NoiseModel | None = None,
        noise_level: NoiseLevel | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            carrier_frequency:
                Center frequency of the generated signal in Hz.
                If zero, the device's target carrier frequency will be generated.
            phase_noise: Phase noise model to apply to the source.
            noise_model:
                Thermal noise model to apply to the source.
                If not specified, i.e. :py:obj:`None`, additive white Gaussian noise will be assumed.
            noise_level:
                Noise level of the source.
                If not specified, i.e. :py:obj:`None`, thermal noise at 300 Kelvin will be assumed.
            seed: Seed with which to initialize the block's random state.
        """

        # Initialize base class
        RFBlock.__init__(self, noise_model, noise_level, seed)

        # Initialize class attributes
        self.carrier_frequency = carrier_frequency
        self.amplitude = amplitude
        self.__o = RFBlockPort(self, 0, RFBlockPortType.OUT)
        self.__phase_noise = NoPhaseNoise() if phase_noise is None else phase_noise
        self.__phase_noise.random_mother = self

    @property
    @override
    def num_input_ports(self) -> int:
        return 0

    @property
    @override
    def num_output_ports(self) -> int:
        return 1

    @override
    def realize(
        self, bandwidth: float, oversampling_factor: int, carrier_frequency: float
    ) -> SourceRealization:
        # The realization's carrier frequency is either the device's target carrier frequency or the source's fixed carrier frequency
        _carrier_frequency = (
            self.carrier_frequency if self.carrier_frequency > 0.0 else carrier_frequency
        )

        return SourceRealization(
            _carrier_frequency,
            self.phase_noise.realize(bandwidth, oversampling_factor),
            self.__amplitude,
            bandwidth,
            oversampling_factor,
            self.noise_model.realize(self.noise_level.get_power(bandwidth)),
        )

    @property
    def carrier_frequency(self) -> float:
        """Center frequency of the generated signal in Hz.

        If zero, the device's target carrier frequency will be generated.

        Raises:
            ValueError: If the carrier frequency is negative.
        """

        return self.__carrier_frequency

    @carrier_frequency.setter
    def carrier_frequency(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("Carrier frequency must be non-negative.")
        self.__carrier_frequency = value

    @property
    def amplitude(self) -> float:
        """Amplitude of the generated signal.

        Raises:
            ValueError: If the amplitude is negative.
        """

        return self.__amplitude

    @amplitude.setter
    def amplitude(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("Amplitude must be non-negative.")
        self.__amplitude = value

    @property
    def o(self) -> RFBlockPort:
        """Output port of the source."""

        return self.__o

    @property
    def phase_noise(self) -> PhaseNoise:
        """Phase noise model applied to the source."""

        return self.__phase_noise

    @override
    def _propagate(self, realization: SourceRealization, input: RFSignal) -> RFSignal:
        output = RFSignal.FromNDArray(
            np.full((1, input.num_samples), realization.amplitude, np.complex128),
            realization.sampling_rate,
            np.full((1,), realization.carrier_frequency, np.float64),
        )

        # Add phase noise to the output signal
        noisy_output = realization.phase_noise.add_noise(output)
        return noisy_output
