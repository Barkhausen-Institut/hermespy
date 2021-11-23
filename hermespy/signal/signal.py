# -*- coding: utf-8 -*-
"""HermesPy Signal Model."""

from __future__ import annotations
from math import ceil
from typing import Sequence

import numpy as np
from numba import jit, complex128

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Signal:
    """Base class of signal models in HermesPy.

    Attributes:

        __samples (np.ndarray):
            An MxT matrix containing uniformly sampled base-band samples of the modeled signal.
            M is the number of individual streams, T the number of available samples.

        __sampling_rate (float):
            Sampling rate of the modeled signal in Hz (in the base-band).

        __carrier_frequency (float):
            Carrier-frequency of the modeled signal in the radio-frequency band,
            i.e. the central frequency in Hz.

        delay (float):
            Delay of the signal in seconds.
    """

    __samples: np.ndarray
    __sampling_rate: float
    __carrier_frequency: float
    delay: float

    def __init__(self,
                 samples: np.ndarray,
                 sampling_rate: float,
                 carrier_frequency: float = 0.,
                 delay: float = 0.) -> None:
        """Signal model initialization.

        Args:
            samples (np.ndarray):
                An MxT matrix containing uniformly sampled base-band samples of the modeled signal.
                M is the number of individual streams, T the number of available samples.

            sampling_rate (float):
                Sampling rate of the modeled signal in Hz (in the base-band).

            carrier_frequency (float, optional):
                Carrier-frequency of the modeled signal in the radio-frequency band,
                i.e. the central frequency in Hz.
                Zero by default.

            delay (float, optional):
                Delay of the signal in seconds.
                Zero by default.
        """

        self.samples = samples
        self.sampling_rate = sampling_rate
        self.carrier_frequency = carrier_frequency
        self.delay = delay

    @property
    def samples(self) -> np.ndarray:
        """Uniformly sampled c

        Returns:
            np.ndarray:
                An MxT matrix of samples,
                where M is the number of individual streams and T the number of samples.
        """

        return self.__samples

    @samples.setter
    def samples(self, value: np.ndarray) -> None:
        """Modify the base-band samples of the modeled signal.

        Args:
            value (np.ndarray):
                An MxT matrix of samples,
                where M is the number of individual streams and T the number of samples.

        Raises:
            ValueError: If `value` can't be interpreted as a matrix.
        """

        if value.ndim != 2:
            raise ValueError("Signal model samples must be a matrix (a two-dimensional array)")

        self.__samples = value

    @property
    def num_streams(self) -> int:
        """The number of streams within this signal model.

        Returns:
            int: The number of streams.
        """

        return self.__samples.shape[0]

    @property
    def num_samples(self) -> int:
        """The number of samples within this signal model.

        Returns:
            int: The number of samples.
        """

        return self.__samples.shape[1]

    @property
    def sampling_rate(self) -> float:
        """The rate at which the modeled signal was sampled.

        Returns:
            float: The sampling rate in Hz.
        """

        return self.__sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value: float) -> None:
        """Modify the rate at which the modeled signal was sampled.

        Args:
            value (float): The sampling rate in Hz.

        Raises:
            ValueError: If `value` is smaller or equal to zero.
        """

        if value <= 0.:
            raise ValueError("The sampling rate of modeled signals must be greater than zero")

        self.__sampling_rate = value

    @property
    def carrier_frequency(self) -> float:
        """The center frequency of the modeled signal in the radio-frequency transmit band.

        Returns:
            float: The carrier frequency in Hz.
        """

        return self.__carrier_frequency

    @carrier_frequency.setter
    def carrier_frequency(self, value: float) -> None:
        """Modify the center frequency of the modeled signal in the radio-frequency transmit band.

        Args:
            value (float): he carrier frequency in Hz.

        Raises:
            ValueError: If `value` is smaller than zero.
        """

        if value < 0.:
            raise ValueError("The carrier frequency of modeled signals must be greater or equal to zero")

        self.__carrier_frequency = value

    @property
    def power(self) -> Sequence[float]:
        """Compute the power of the modeled signal.

        Returns:
            Sequence[float]: The power of each modeled stream.
        """

        stream_power = np.sum(self.__samples.real ** 2 + self.__samples.imag ** 2, axis=1) / (2 * self.num_samples + 1)
        return stream_power.tolist()

    def resample(self, sampling_rate: float) -> Signal:
        """Resample the modeled signal to a different sampling rate.

        Args:
            sampling_rate (float):
                Sampling rate of the new signal model in Hz.

        Returns:
            Signal:
                The resampled signal model.

        Raises:
            ValueError: If `sampling_rate` is smaller or equal to zero.
        """

        if sampling_rate <= 0.:
            raise ValueError("Sampling rate for resampling must be greater than zero.")

        # Resample the internal samples
        samples = Signal.__resample(self.__samples, self.__sampling_rate, sampling_rate)

        # Create a new signal object from the resampled samples and return it as result
        return Signal(samples, sampling_rate, carrier_frequency=self.__carrier_frequency, delay=self.delay)

    @staticmethod
    @jit(nopython=True)
    def __resample(signal: np.ndarray,
                   input_sampling_rate: float,
                   output_sampling_rate: float) -> np.ndarray:
        """Internal subroutine to resample a given set of samples to a new sampling rate.

        Uses sinc-interpolation, therefore `signal` is assumed to be band-limited.

        Arguments:

            signal (np.ndarray):
                MxT matrix of M signal-streams to be resampled, each containing T time-discrete samples.

            input_sampling_rate (float):
                Rate at which `signal` is sampled in Hz.

            output_sampling_rate (float):
                Rate at which the resampled signal will be sampled in Hz.

        Returns:
            np.ndarray:
                MxT' matrix of M resampled signal streams.
                The number of samples T' depends on the sampling rate relations, i.e.
                T' = T * output_sampling_rate / input_sampling_rate .
        """

        num_streams = signal.shape[0]

        num_input_samples = signal.shape[1]
        num_output_samples = ceil(num_input_samples * output_sampling_rate / input_sampling_rate)

        input_timestamps = np.arange(num_input_samples) / input_sampling_rate
        output_timestamps = np.arange(num_output_samples) / output_sampling_rate

        output = np.empty((num_streams, num_output_samples), dtype=complex128)
        for output_idx in np.arange(num_output_samples):

            # Sinc interpolation weights for single output sample
            interpolation_weights = np.sinc((input_timestamps - output_timestamps[output_idx]) * input_sampling_rate) + 0j

            # Resample all streams simultaneously
            output[:, output_idx] = signal @ interpolation_weights

        return output
