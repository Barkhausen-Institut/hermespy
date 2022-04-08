# -*- coding: utf-8 -*-
"""
=============================================
Frequency Modulated Continuous Waveform Radar
=============================================
"""

import numpy as np
from scipy.constants import speed_of_light, pi
from scipy.fft import fft2, fftshift

from ..core.signal_model import Signal
from .radar import RadarWaveform

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.7"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class FMCW(RadarWaveform):
    """Frequency Modulated Continuous Waveform Radar Sensing."""

    __num_chirps: int       # Number of chirps per radar frame
    __bandwidth: float      # Sweep bandwidth of the chirp in Hz
    __sampling_rate: float  # ADC sampling rate of the baseband signal in Hz
    __max_range: float      # Maximum detectable target range in m

    def __init__(self,
                 num_chirps: int = 10,
                 bandwidth: float = 1.5e9,
                 sampling_rate: float = 1.5e9,
                 max_range: float = 50.) -> None:

        self.num_chirps = num_chirps
        self.bandwidth = bandwidth
        self.sampling_rate = sampling_rate
        self.max_range = max_range

    def ping(self) -> Signal:

        return Signal(self.__frame_prototype(), self.sampling_rate)

    def estimate(self, signal: Signal) -> np.ndarray:

        chirp_prototype = self.__chirp_prototype()
        signal = signal.resample(self.sampling_rate).samples[0, :len(chirp_prototype)*self.num_chirps]
        chirp_stack = np.reshape(signal, (-1, len(chirp_prototype)))

        baseband_samples = chirp_stack.conj() * chirp_prototype
        transform = fft2(baseband_samples)

        return np.abs(transform)

    @property
    def range_bins(self) -> np.ndarray:

        return np.arange(int(self.max_range / self.range_resolution)) * self.range_resolution

    @property
    def velocity_bins(self) -> np.ndarray:

        return np.arange(self.num_chirps)

    @property
    def num_chirps(self) -> int:
        """Number of chirps per transmitted radar frame.

        Returns:
            int: Number of chirps.

        Raises:
            ValueError: If the number of chirps is smaller than one.
        """

        return self.__num_chirps

    @num_chirps.setter
    def num_chirps(self, value: int) -> None:

        if value < 1:
            raise ValueError("Number of chirps must be greater or equal to one")

        self.__num_chirps = int(value)

    @property
    def bandwidth(self) -> float:
        """Bandwidth swept during each chirp.

        Returns:
            float: Sweep bandwidth in Hz.

        Raises:
            ValueError: If bandwidth is smaller or equal to zero.
        """

        return self.__bandwidth

    @bandwidth.setter
    def bandwidth(self, value: float) -> None:

        if value < 0.:
            raise ValueError("Bandwidth must be greater than zero")

        self.__bandwidth = value

    @property
    def range_resolution(self) -> float:
        """Depth sensing resolution.

        Returns:
            float: Range resolution in m.

        Raises:
            ValueError: If resolution is smaller or equal to zero.
        """

        return speed_of_light / (2 * self.__bandwidth)

    @range_resolution.setter
    def range_resolution(self, value: float) -> None:

        if value <= 0.:
            raise ValueError("Range resolution must be greater or equal to zero.")

        self.bandwidth = speed_of_light / (2 * value)

    @property
    def sampling_rate(self) -> float:

        return self.__sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value: float) -> None:

        if value <= 0.:
            raise ValueError("Sampling rate must be greater than zero")

        self.__sampling_rate = value

    @property
    def max_range(self) -> float:
        """Maximum detectable target range.

        Returns:
            float: Maximum range in m.

        Raises:
            ValueError: If range is smaller or equal to zero.
        """

        return self.__max_range

    @max_range.setter
    def max_range(self, value: float) -> None:

        if value <= 0.:
            raise ValueError("Maximum detectable range must be greater than zero")

        self.__max_range = value

    @property
    def max_velocity(self) -> float:
        """Maximum relative target velocity detectable.

        Returns:

            float: Maximum target velocity in m/s.
        """

        raise NotImplementedError

    @property
    def slope(self) -> float:
        """Slope of the bandwidth sweep.

        Returns:
            float: Slope in Hz / s.

        Raises:
            ValueError: If slope is smaller or equal to zero.
        """

        return self.sampling_rate * speed_of_light / (2 * self.max_range)

    def __chirp_prototype(self) -> np.ndarray:

        duration = self.bandwidth / self.slope
        num_samples = int(duration * self.sampling_rate)

        timestamps = np.arange(num_samples) / self.sampling_rate
        chirp = np.exp(1j * pi * self.slope * timestamps ** 2)

        return chirp

    def __frame_prototype(self) -> np.ndarray:

        return np.tile(self.__chirp_prototype(), self.num_chirps)
