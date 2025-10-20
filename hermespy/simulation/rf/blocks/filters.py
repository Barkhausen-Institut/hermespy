# -*- coding: utf-8 -*-

from __future__ import annotations
from functools import cache
from typing_extensions import override

import numpy as np
from scipy.signal import butter, filtfilt

from ..block import RFBlock, RFBlockRealization, RFBlockPort, RFBlockPortType
from ..signal import RFSignal
from ..noise import NoiseModel, NoiseLevel, NoiseRealization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class FilterRealization(RFBlockRealization):
    """Realization of a radio-frequency filter block."""

    __numerator_coefficients: np.ndarray
    __denominator_coefficients: np.ndarray

    def __init__(
        self,
        numerator_coefficients: np.ndarray,
        denominator_coefficients: np.ndarray,
        bandwidth: float,
        oversampling_factor: int,
        noise_realization: NoiseRealization,
    ) -> None:
        """
        Args:
            numerator_coefficients: Coefficients of the filter's numerator.
            denominator_coefficients: Coefficients of the filter's denominator.
            bandwidth: Bandwidth of the simulation in Hz.
            oversampling_factor: Oversampling factor of the simulation.
            noise_realization: Noise realization applied to the filter.
        """

        # Initialize base class
        RFBlockRealization.__init__(self, bandwidth, oversampling_factor, noise_realization)

        # Store attributes
        self.__numerator_coefficients = numerator_coefficients
        self.__denominator_coefficients = denominator_coefficients

    @property
    def numerator_coefficients(self) -> np.ndarray:
        """Coefficients of the filter's numerator."""
        return self.__numerator_coefficients

    @property
    def denominator_coefficients(self) -> np.ndarray:
        """Coefficients of the filter's denominator."""
        return self.__denominator_coefficients


class Filter(RFBlock):
    """A radio-frequency block that applies an arbitrary filter during signal propagation."""

    __i: RFBlockPort[Filter]
    __o: RFBlockPort[Filter]

    def __init__(
        self,
        noise_model: NoiseModel | None = None,
        noise_level: NoiseLevel | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            numerator_coefficients: Coefficients of the filter's numerator.
            denominator_coefficients: Coefficients of the filter's denominator.
            noise_model:
                Thermal noise model applied after signal propagation.
                If not specifiedm, i.e. :py:obj:`None`, AWGN is assumed.
            noise_level:
                Thermal noise level applied after signal propagation.
                If not specified, i.e. :py:obj:`None`, thermal noise at 300 K is assumed.
            seed: Seed with which to initialize the block's random state.
        """

        # Initialize base class
        RFBlock.__init__(self, noise_model, noise_level, seed)

        # Store attributes
        self.__i = RFBlockPort(self, 0, RFBlockPortType.IN)
        self.__o = RFBlockPort(self, 0, RFBlockPortType.OUT)

    @property
    @override
    def num_input_ports(self) -> int:
        return 1

    @property
    @override
    def num_output_ports(self) -> int:
        return 1

    @property
    def i(self) -> RFBlockPort[Filter]:
        """Input port of the filter block."""

        return self.__i

    @property
    def o(self) -> RFBlockPort[Filter]:
        """Output port of the filter block."""

        return self.__o

    @override
    def _propagate(self, realization: FilterRealization, input: RFSignal) -> RFSignal:
        output = filtfilt(
            realization.numerator_coefficients, realization.denominator_coefficients, input, axis=1
        )

        return RFSignal.FromNDArray(
            output, input.sampling_rate, input.carrier_frequencies, input.noise_powers, input.delay
        )


class HPF(Filter):
    """A radio-frequency block that applies a high-pass filter during signal propagation."""

    __cutoff_frequency: float

    def __init__(
        self,
        cutoff_frequency: float,
        filter_order: int = 4,
        noise_model: NoiseModel | None = None,
        noise_level: NoiseLevel | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            cutoff_frequency: Cutoff frequency of the high-pass filter in Hz.
            noise_model:
                Thermal noise model applied after signal propagation.
                If not specified, i.e. :py:obj:`None`, AWGN is assumed.
            filter_order:
                Order of the high-pass filter.
                Refer to the documentation of :func:`scipy.signal.butter` for details.
            noise_level:
                Thermal noise level applied after signal propagation.
                If not specified, i.e. :py:obj:`None`, thermal noise at 300 K is assumed.
            seed: Seed with which to initialize the block's random state.
        """

        # Initialize base class
        Filter.__init__(self, noise_model, noise_level, seed)

        # Store attributes
        self.cutoff_frequency = cutoff_frequency
        self.filter_order = filter_order

    @property
    def cutoff_frequency(self) -> float:
        """Cutoff frequency of the high-pass filter in Hz.

        Raises:
            ValueError: If the cutoff frequency is not positive.
        """

        return self.__cutoff_frequency

    @cutoff_frequency.setter
    def cutoff_frequency(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Cutoff frequency must be positive")

        self.__cutoff_frequency = value

    @property
    def filter_order(self) -> int:
        """Order of the high-pass filter.

        Refer to the documentation of :func:`scipy.signal.butter` for details.

        Raises:
            ValueError: If the filter order is less than 1.
        """

        return self.__filter_order

    @filter_order.setter
    def filter_order(self, value: int) -> None:
        if value < 1:
            raise ValueError("Filter order must be at least 1")

        self.__filter_order = value

    @cache
    @staticmethod
    def _filter_coefficients(
        cutoff_frequency: float, sampling_rate: float, filter_order: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the filter coefficients for a high-pass filter.

        Subroutine of the highpass filter realization caching its output for improved performance.

        Args:
            cutoff_frequency: Cutoff frequency of the high-pass filter in Hz.
            sampling_rate: Sampling rate of the signal in Hz.
            filter_order: Filter order. Refer to the documentation of :func:`scipy.signal.butter` for details.

        Returns:
            Tuple of numerator and denominator coefficients for the filter.
        """

        return butter(filter_order, cutoff_frequency, btype="high", fs=sampling_rate, output="ba")

    @override
    def realize(
        self, bandwidth: float, oversampling_factor: int, carrier_frequency: float
    ) -> FilterRealization:
        numerator_coefficients, denominator_coefficients = HPF._filter_coefficients(
            self.cutoff_frequency, bandwidth * oversampling_factor, self.filter_order
        )
        return FilterRealization(
            numerator_coefficients,
            denominator_coefficients,
            bandwidth,
            oversampling_factor,
            self.noise_model.realize(self.noise_level.get_power(bandwidth)),
        )
