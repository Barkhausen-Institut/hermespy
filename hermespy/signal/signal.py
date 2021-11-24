# -*- coding: utf-8 -*-
"""HermesPy Signal Model."""

from __future__ import annotations
from copy import deepcopy
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from numba import jit, complex128
from scipy.constants import pi
from scipy.fft import fft, fftshift, fftfreq


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

        self.samples = samples.copy()
        self.sampling_rate = sampling_rate
        self.carrier_frequency = carrier_frequency
        self.delay = delay

    @classmethod
    def empty(cls,
              sampling_rate: float,
              num_streams: int = 0,
              num_samples: int = 0,
              **kwargs) -> Signal:
        """Create a new empty signal model instance.

        Args:
            sampling_rate (float):
                Sampling rate of the modeled signal in Hz (in the base-band).

            num_streams (int, optional):
                Number of signal streams within this empty model.

            num_samples (int, optional):
                Number of signal samples within this empty model.

            kwargs:
                Additional initialization arguments, piped through to the class init.
        """

        return Signal(np.empty((num_streams, num_samples), dtype=complex), sampling_rate, **kwargs)

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

        # Expand value vectors to matrices, indicating a signal model with one stream
        if value.ndim == 1:
            value = value[np.newaxis, :]

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
    def power(self) -> np.ndarray:
        """Compute the power of the modeled signal.

        Returns:
            np.ndarray:: The power of each modeled stream within a numpy vector.
        """

        stream_power = np.sum(self.__samples.real ** 2 + self.__samples.imag ** 2, axis=1) / self.num_samples
        return stream_power

    def copy(self) -> Signal:
        """Copy this signal model to a new object.

        Returns:
            Signal: A copy of this signal model.
        """

        return deepcopy(self)

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
        if self.__sampling_rate != sampling_rate:
            samples = Signal.__resample(self.__samples, self.__sampling_rate, sampling_rate)

        # Skip resampling if both sampling rates are identical
        else:
            samples = self.__samples.copy()

        # Create a new signal object from the resampled samples and return it as result
        return Signal(samples, sampling_rate, carrier_frequency=self.__carrier_frequency, delay=self.delay)

    def superimpose(self, added_signal: Signal) -> None:
        """Superimpose an additive signal model to this model.

        Internally re-samples `added_signal` to this model's sampling rate, if required.
        Mixes `added_signal` according to the carrier-frequency distance.

        Raises:
            ValueError: If `added_signal` contains a different number of streams than this signal model.
            NotImplementedError: If the delays if this signal and `added_signal` differ.
        """

        if added_signal.num_streams != self.num_streams:
            raise ValueError("Superimposing signal models with different stream counts is not defined")

        if self.delay != added_signal.delay:
            raise NotImplementedError("Superimposing signal models of differing delay is not yet supported")

        # Resample the added signal
        resampled_added_signal = added_signal.resample(self.__sampling_rate)

        # Extend the samples matrix if required
        if self.num_samples < resampled_added_signal.num_samples:
            self.__samples = np.append(self.__samples, np.zeros((self.num_streams,
                                                                 resampled_added_signal.num_samples - self.num_samples),
                                                                dtype=complex))

        # Mix the added signal onto this signal's samples according to the carrier frequency distance
        frequency_distance = resampled_added_signal.carrier_frequency - self.carrier_frequency
        self.__mix(self.__samples, resampled_added_signal.samples, self.__sampling_rate, frequency_distance)

    @staticmethod
    @jit(nopython=True)
    def __mix(target_samples: np.ndarray,
              added_samples: np.ndarray,
              sampling_rate: float,
              frequency_distance: float) -> None:
        """Internal subroutine to mix two sets of signal model samples.

        Args:
            target_samples (np.ndarray):
                Target samples onto which `added_samples` will be mixed.

            added_samples (np.ndarray):
                Samples to be mixed ont `target_samples`.

            sampling_rate (float):
                Sampling rate in Hz, which both `target_samples` and `added_samples` must share.

            frequency_distance (float):
                Distance between the carrier frequencies of `target_samples` and `added_samples` in Hz.
        """

        # ToDo: Reminder, mixing like this currently does not account for possible delays.
        num_samples = added_samples.shape[1]
        mix_sinusoid = np.exp(2j * pi * np.arange(num_samples) * frequency_distance / sampling_rate)

        for stream_idx in range(added_samples.shape[0]):
            target_samples[stream_idx, :num_samples] += added_samples[stream_idx, :] * mix_sinusoid

    @property
    def timestamps(self) -> np.ndarray:
        """The sample-points of the signal model.

        Returns:
            np.ndarray: Vector of length T containing sample-timestamps in seconds.
        """

        return np.arange(self.num_samples) / self.__sampling_rate - self.delay

    def plot(self) -> None:
        """Plot the current signal in time- and frequency-domain."""

        fig, axes = plt.subplots(self.num_streams, 2, squeeze=False)
        timestamps = self.timestamps

        for stream_idx, stream_samples in enumerate(self.__samples):

            axes[stream_idx, 0].plot(timestamps, stream_samples.real)
            axes[stream_idx, 1].plot(fftshift(fftfreq(self.num_samples, 1 / self.sampling_rate)),
                                     abs(fftshift(fft(stream_samples))))

        fig.suptitle("Signal Model")

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

    def append_samples(self, signal: Signal) -> None:
        """Append samples in time-domain to the signal model.

        Args:
            signal (Signal): The signal to be appended.

        Raise:
            ValueError: If the number of streams don't align.
        """

        # Adapt number of streams to fit the appended signal if this signal is empty
        if self.num_streams < 1:
            self.__samples = np.empty((signal.num_streams, 0), dtype=complex)

        if signal.num_streams != self.num_streams:
            raise ValueError("Appending signal models with differing amount of streams is not defined")

        if signal.carrier_frequency != self.__carrier_frequency:
            raise ValueError("Appending signals of differing carrier frequencies is not defined")

        # Resample the signal if required
        if signal.sampling_rate != self.sampling_rate:
            raise NotImplementedError("Resampling not yet implemented")

        self.__samples = np.append(self.__samples, signal.samples, axis=1)

    def append_streams(self, signal: Signal) -> None:
        """Append streams to the signal model.

        Args:
            signal (Signal): The signal to be appended.

        Raise:
            ValueError: If the number of samples don't align.
        """

        # Adapt the number of samples if this signal is empty to match the signal to be appended
        if self.num_samples < 1:
            self.__samples = np.empty((0, signal.num_samples), dtype=complex)

        if signal.num_samples != self.num_samples:
            raise ValueError("Appending signal models streams with differing amount of samples is not supproted")

        if signal.carrier_frequency != self.__carrier_frequency:
            raise ValueError("Appending signals of differing carrier frequencies is not defined")

        # Resample the signal if required
        if signal.sampling_rate != self.sampling_rate:
            raise NotImplementedError("Resampling not yet implemented")

        self.__samples = np.append(self.__samples, signal.samples, axis=0)
