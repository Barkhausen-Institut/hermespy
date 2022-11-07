# -*- coding: utf-8 -*-
"""
=============================================
Frequency Modulated Continuous Waveform Radar
=============================================
"""

from fractions import Fraction

import numpy as np
from scipy.constants import speed_of_light, pi
from scipy.fft import fft2
from scipy import signal

from hermespy.core import Signal
from .radar import RadarWaveform

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class FMCW(RadarWaveform):
    """
    Frequency Modulated Continuous Waveform Radar Sensing with stretch processing.

    This class generates a frame consisting of a sequence of unmodulated chirps. They are used for radar detection with
    stretch processing, i.e., mixing the received signal with the transmitted sequence, (under)sampling and applying an
    FFT.

    Args:

        num_chirps (float):
            number of chirps in a radar frame

        bandwidth (float):
            Sweep bandwidth of the chirp in Hz

        sampling_rate (float):
            simulation sampling rate of the baseband signal in Hz

        chirp_duration (float):
            chirp duration in seconds

        pulse_rep_interval (float):
            pulse repetition interval in seconds

        adc_sampling_rate (float):
            sampling rate of ADC after mixing

    """

    __num_chirps: int            # Number of chirps per radar frame
    __bandwidth: float           # Sweep bandwidth of the chirp in Hz
    __sampling_rate: float       # simulation sampling rate of the baseband signal in Hz
    __chirp_duration: float      # chirp duration in seconds
    __pulse_rep_interval: float  # pulse repetition interval in seconds
    __adc_sampling_rate: float   # sampling rate of ADC after mixing

    def __init__(self,
                 num_chirps: int = 10,
                 bandwidth: float = .1e9,
                 chirp_duration: float = 1.5e-6,
                 pulse_rep_interval: float = 1.5e-6,
                 sampling_rate: float = None,
                 adc_sampling_rate: float = None) -> None:

        self.num_chirps = num_chirps
        self.bandwidth = bandwidth
        self.chirp_duration = chirp_duration
        self.pulse_rep_interval = pulse_rep_interval
        self.sampling_rate = bandwidth if sampling_rate is None else sampling_rate
        self.adc_sampling_rate = self.sampling_rate if adc_sampling_rate is None else adc_sampling_rate

    def ping(self) -> Signal:
        return Signal(self.__frame_prototype(), self.sampling_rate)

    def estimate(self, input_signal: Signal) -> np.ndarray:

        pulse_len = int(self.pulse_rep_interval * self.sampling_rate)
        input_signal = input_signal.resample(
            self.sampling_rate).samples[0, :pulse_len * self.num_chirps]
        chirp_stack = np.reshape(input_signal, (-1, pulse_len))

        chirp_prototype = self.__chirp_prototype()
        chirp_stack = chirp_stack[:, :len(chirp_prototype)]
        baseband_samples = chirp_stack.conj() * chirp_prototype

        if self.adc_sampling_rate < self.sampling_rate:
            downsampling_rate = Fraction(
                self.adc_sampling_rate / self.sampling_rate).limit_denominator(100)

            baseband_samples = signal.resample_poly(baseband_samples,
                                                    up=downsampling_rate.numerator, down=downsampling_rate.denominator,
                                                    axis=1)

        transform = fft2(baseband_samples)
        # fft shift the Doppler components
        transform = np.fft.ifftshift(transform, axes=0)

        return np.abs(transform)

    @property
    def max_range(self) -> float:

        max_range = self.adc_sampling_rate * speed_of_light / (2 * self.slope)
        return max_range

    @property
    def range_bins(self) -> np.ndarray:

        return np.arange(int(self.max_range / self.range_resolution)) * self.range_resolution

    @property
    def velocity_bins(self) -> np.ndarray:

        bins = (np.arange(self.num_chirps) -
                np.ceil(self.num_chirps / 2)) * self.doppler_resolution
        return bins

    @property
    def frame_duration(self) -> float:

        return self.pulse_rep_interval * self.num_chirps

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
            raise ValueError(
                "Number of chirps must be greater or equal to one")

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

        if value <= 0.:
            raise ValueError("Bandwidth must be greater than zero")

        self.__bandwidth = value

    @property
    def chirp_duration(self) -> float:
        """Duration of a single chirp within the FMCW frame.

        Returns: Duration in seconds.

        Raises:

            ValueErorr: For durations smaller or equal to zero.
        """

        return self.__chirp_duration

    @chirp_duration.setter
    def chirp_duration(self, value: float) -> None:

        if value <= 0.:
            raise ValueError("FMCW chirp duration must be greater than zero")

        self.__chirp_duration = value

    @property
    def adc_sampling_rate(self) -> float:
        """Sampling rate at ADC

        Returns:
            float: sampling rate in Hz.

        Raises:
            ValueError: If sampling rate is smaller or equal to zero.
        """
        return self.__adc_sampling_rate

    @adc_sampling_rate.setter
    def adc_sampling_rate(self, value: float) -> None:

        if value <= 0.:
            raise ValueError("ADC Sampling rate must be greater than zero")

        self.__adc_sampling_rate = value

    @property
    def range_resolution(self) -> float:
        """Depth sensing resolution.

        Returns:
            float: Range resolution in m.

        Raises:
            ValueError: If resolution is smaller or equal to zero.
        """

        return speed_of_light / (2 * self.bandwidth)

    @property
    def doppler_resolution(self) -> float:
        """Doppler sensing resolution.

        Returns:
            float: Doppler resolution in Hz.

        Raises:
            ValueError: If resolution is smaller or equal to zero.
        """

        return 1. / self.pulse_rep_interval / self.num_chirps

    @property
    def sampling_rate(self) -> float:

        return self.__sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value: float) -> None:

        if value <= 0.:
            raise ValueError("Sampling rate must be greater than zero")

        self.__sampling_rate = value

    @property
    def pulse_rep_interval(self) -> float:
        """Pulse repetition interval

        Returns:
            float: pulse repetition interval in seconds

        Raises:
            ValueError: If interval is smaller or equal to zero.
        """

        return self.__pulse_rep_interval

    @pulse_rep_interval.setter
    def pulse_rep_interval(self, value: float) -> None:

        if value <= 0.:
            raise ValueError(
                "Pulse repetition interval must be greater than zero")

        self.__pulse_rep_interval = value

    @property
    def max_velocity(self) -> float:
        """Maximum relative target velocity detectable.

        Returns:

            float: Maximum target velocity in m/s.
        """

        raise NotImplementedError  # pragma no cover

    @property
    def slope(self) -> float:
        """Slope of the bandwidth sweep.

        Returns:
            float: Slope in Hz / s.

        Raises:
            ValueError: If slope is smaller or equal to zero.
        """

        return self.bandwidth / self.chirp_duration

    @property
    def energy(self) -> float:
        """Energy of a radar pulse

        Returns:
            pulse energy.
        """

        return self.num_chirps * int(self.chirp_duration * self.sampling_rate)

    @property
    def power(self) -> float:

        return 1.

    def __chirp_prototype(self) -> np.ndarray:

        num_samples = int(self.chirp_duration * self.sampling_rate)

        timestamps = np.arange(num_samples) / self.sampling_rate

        # baseband chirp between -B/2 and B/2
        chirp = np.exp(1j * pi * (-self.bandwidth *
                       timestamps + self.slope * timestamps ** 2))
        return chirp

    def __pulse_prototype(self) -> np.ndarray:

        num_zero_samples = int(
            (self.pulse_rep_interval - self.chirp_duration) * self.sampling_rate)

        if num_zero_samples < 0:
            raise ValueError(
                "Pulse repetition interval cannot be less than chirp duration in FMCW radar")

        pulse = np.concatenate(
            (self.__chirp_prototype(), np.zeros(num_zero_samples)))
        return pulse

    def __frame_prototype(self) -> np.ndarray:

        frame = np.tile(self.__pulse_prototype(), self.num_chirps)
        return frame
