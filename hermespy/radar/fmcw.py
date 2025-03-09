# -*- coding: utf-8 -*-

from __future__ import annotations
from fractions import Fraction
from typing_extensions import override

import numpy as np
from scipy.constants import speed_of_light, pi
from scipy.fft import fft2
from scipy import signal

from hermespy.core import Signal, SerializationProcess
from .radar import RadarWaveform

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class FMCW(RadarWaveform):
    """
    Frequency Modulated Continuous Waveform Radar Sensing with stretch processing.

    This class generates a frame consisting of a sequence of unmodulated chirps. They are used for radar detection with
    stretch processing, i.e., mixing the received signal with the transmitted sequence, (under)sampling and applying an
    FFT.
    S
    A minimal example configuring an :class:`FMCW` radar waveform illuminating a single target
    within the context of a :class:`Simulation<hermespy.simulation.simulation.Simulation>` would look like this:

    .. literalinclude:: ../../docssource/scripts/examples/radar_fmcw_FMCW.py
        :language: python
        :linenos:
        :lines: 03-23
    """

    __DEFAULT_NUM_CHIRPS = 10
    __DEFAULT_BANDWIDTH = 0.1e9
    __DEFAULT_CHIRP_DURATION = 1.5e-6
    __DEFAULT_PULSE_REP_INTERVAL = 1.5e-6
    __DERIVE_SAMPLING_RATE: float = 0.0
    """Magic number at which FMCW waveforms will automatically derive the sampling rates."""

    __num_chirps: int  # Number of chirps per radar frame
    __bandwidth: float  # Sweep bandwidth of the chirp in Hz
    __sampling_rate: float  # simulation sampling rate of the baseband signal in Hz
    __chirp_duration: float  # chirp duration in seconds
    __pulse_rep_interval: float  # pulse repetition interval in seconds
    __adc_sampling_rate: float  # sampling rate of ADC after mixing

    def __init__(
        self,
        num_chirps: int = __DEFAULT_NUM_CHIRPS,
        bandwidth: float = __DEFAULT_BANDWIDTH,
        chirp_duration: float = __DEFAULT_CHIRP_DURATION,
        pulse_rep_interval: float = __DEFAULT_PULSE_REP_INTERVAL,
        sampling_rate: float | None = None,
        adc_sampling_rate: float | None = None,
    ) -> None:
        """
        Args:

            num_chirps (float, optional):
                Number of dedicated chirps within a single radar frame.
                :math:`10` by default.

            bandwidth (float, optional):
                Sweep bandwidth of every chirp in Hz.
                :math:`0.1~\\mathrm{GHz}` by default.

            chirp_duration (float, optional):
                Duration of every chirp in seconds.
                :math:`1.5~\\mathrm{\\mu s}` by default.

            pulse_rep_interval (float, optional):
                Repetition interval of the individual chirps in seconds.
                :math:`1.5~\\mathrm{\\mu s}` by default.

            sampling_rate (float, optional):
                Sampling rate of the baseband signal in Hz.
                If not specified, the sampling rate will be equal to the bandwidth.

            adc_sampling_rate (float, optional):
                Sampling rate of the analog-digital conversion in Hz.
                If not specified, the adc sampling rate will be equal to the bandwidth.
        """

        # Initialize base class
        RadarWaveform.__init__(self)

        # Initialize attributes
        self.num_chirps = num_chirps
        self.bandwidth = bandwidth
        self.chirp_duration = chirp_duration
        self.pulse_rep_interval = pulse_rep_interval
        self.sampling_rate = FMCW.__DERIVE_SAMPLING_RATE if sampling_rate is None else sampling_rate
        self.adc_sampling_rate = (
            FMCW.__DERIVE_SAMPLING_RATE if adc_sampling_rate is None else adc_sampling_rate
        )

    def ping(self) -> Signal:
        return Signal.Create(self.__frame_prototype(), self.sampling_rate)

    def estimate(self, input_signal: Signal) -> np.ndarray:
        num_pulse_samples = int(self.pulse_rep_interval * self.sampling_rate)
        num_frame_samples = self.num_chirps * num_pulse_samples

        resampled_input_signal = input_signal.resample(self.sampling_rate)
        input_samples = (
            resampled_input_signal.getitem((0, slice(None, num_frame_samples)))
            if resampled_input_signal.num_samples >= num_frame_samples
            else np.append(
                resampled_input_signal.getitem(0),
                np.zeros(num_frame_samples - resampled_input_signal.num_samples),
            )
        )

        chirp_stack = np.reshape(input_samples, (-1, num_pulse_samples))

        chirp_prototype = self.__chirp_prototype()
        chirp_stack = chirp_stack[:, : len(chirp_prototype)]
        baseband_samples = chirp_stack.conj() * chirp_prototype

        if self.adc_sampling_rate < self.sampling_rate:
            downsampling_rate = Fraction(
                self.adc_sampling_rate / self.sampling_rate
            ).limit_denominator(1000)
            baseband_samples = signal.resample_poly(
                baseband_samples,
                up=downsampling_rate.numerator,
                down=downsampling_rate.denominator,
                axis=1,
            )

        transform = fft2(baseband_samples)
        # fft shift the Doppler components
        transform = np.fft.ifftshift(transform, axes=0)

        return np.abs(transform)

    @property
    @override
    def max_range(self) -> float:
        max_range = self.adc_sampling_rate * speed_of_light / (2 * self.slope)
        return max_range

    @property
    @override
    def range_resolution(self) -> float:
        return speed_of_light / (2 * self.bandwidth)

    @property
    @override
    def max_relative_doppler(self) -> float:
        # The maximum velocity is the wavelength divided by four times the pulse repetition interval
        max_doppler = 1 / (4 * self.pulse_rep_interval)
        return max_doppler

    @property
    @override
    def relative_doppler_resolution(self) -> float:
        # The doppler resolution is the inverse of twice the frame duration
        resolution = 1 / (2 * self.frame_duration)
        return resolution

    @property
    @override
    def relative_doppler_bins(self) -> np.ndarray:
        return (
            np.arange(self.num_chirps) * self.relative_doppler_resolution
            - self.max_relative_doppler
        )

    @property
    @override
    def frame_duration(self) -> float:
        return self.pulse_rep_interval * self.num_chirps

    @property
    def num_chirps(self) -> int:
        """Number of chirps per transmitted radar frame.

        Changing the number of chirps per frame will result in a different radar frame length:

        .. plot:: scripts/plots/radar_fmcw_num_chirps.py

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

        .. plot:: scripts/plots/radar_fmcw_bandwidth.py

        Returns:
            float: Sweep bandwidth in Hz.

        Raises:
            ValueError: If bandwidth is smaller or equal to zero.
        """

        return self.__bandwidth

    @bandwidth.setter
    def bandwidth(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("Bandwidth must be greater than zero")

        self.__bandwidth = value

    @property
    def chirp_duration(self) -> float:
        """Duration of a single chirp within the FMCW frame.

        In combination with :meth:`pulse_rep_interval<.pulse_rep_interval>` the chirp duration
        determines the guard interval between two consecutive chirps:

        .. plot:: scripts/plots/radar_fmcw_chirp_duration.py

        Raises:

            ValueErorr: For durations smaller or equal to zero.
        """

        return self.__chirp_duration

    @chirp_duration.setter
    def chirp_duration(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("FMCW chirp duration must be greater than zero")

        self.__chirp_duration = value

    @property
    def adc_sampling_rate(self) -> float:
        """Sampling rate at the ADC in Hz.

        Raises:
            ValueError: If sampling rate is smaller or equal to zero.
        """

        # If the sampling rate is not specified, it will be derived from the bandwidth
        if self.__adc_sampling_rate == FMCW.__DERIVE_SAMPLING_RATE:
            return self.bandwidth

        return self.__adc_sampling_rate

    @adc_sampling_rate.setter
    def adc_sampling_rate(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("ADC Sampling rate must be non-negative")

        self.__adc_sampling_rate = value

    @property
    def sampling_rate(self) -> float:
        # If the sampling rate is not specified, it will be derived from the bandwidth
        if self.__sampling_rate == FMCW.__DERIVE_SAMPLING_RATE:
            return self.__bandwidth

        return self.__sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("Sampling rate must be non-negative")

        self.__sampling_rate = value

    @property
    def pulse_rep_interval(self) -> float:
        """Pulse repetition interval in seconds.

        The pulse repetition interval determines the overall frame duration.
        In combination with the :meth:`chirp_duration<.chirp_duration>` the pulse repetition interval
        it determines the guard interval between two consecutive chirps:

        .. plot:: scripts/plots/radar_fmcw_pulse_rep_interval.py

        Raises:
            ValueError: If interval is smaller or equal to zero.
        """

        return self.__pulse_rep_interval

    @pulse_rep_interval.setter
    def pulse_rep_interval(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("Pulse repetition interval must be greater than zero")

        self.__pulse_rep_interval = value

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
        """Energy of an FMCW radar pulse in :math:`\\mathrm{Wh}`."""

        return self.num_chirps * int(self.chirp_duration * self.sampling_rate)

    @property
    def power(self) -> float:
        """Power of an FMCW radar pulse in :math:`\\mathrm{W}`."""

        return 1.0

    def __chirp_prototype(self) -> np.ndarray:
        """Prototype function to generate a single chirp.

        Raises:

            RuntimeError: If the sampling rate is smaller than the chirp bandwidth.
        """

        # Validate that there's no aliasing during chirp generation
        if self.sampling_rate < self.bandwidth:
            raise RuntimeError(
                f"Sampling rate may not be smaller than chirp bandwidth ({self.sampling_rate} < {self.bandwidth})"
            )

        num_samples = int(self.chirp_duration * self.sampling_rate)
        timestamps = np.arange(num_samples) / self.sampling_rate

        # baseband chirp between -B/2 and B/2
        chirp = np.exp(1j * pi * (-self.bandwidth * timestamps + self.slope * timestamps**2))
        return chirp

    def __pulse_prototype(self) -> np.ndarray:
        """Prototype function to generate a single pulse including chirp and guard interval.

        Raises:

            RuntimeError: If the pulse repetition interval is smaller than the chirp duration.
        """

        num_zero_samples = int((self.pulse_rep_interval - self.chirp_duration) * self.sampling_rate)

        if num_zero_samples < 0:
            raise RuntimeError(
                "Pulse repetition interval cannot be less than chirp duration in FMCW radar"
            )

        pulse = np.concatenate((self.__chirp_prototype(), np.zeros(num_zero_samples)))
        return pulse

    def __frame_prototype(self) -> np.ndarray:
        """Prototype function to generate a single radar frame."""

        frame = np.tile(self.__pulse_prototype(), self.num_chirps)
        return frame

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_integer(self.num_chirps, "num_chirps")
        process.serialize_floating(self.bandwidth, "bandwidth")
        process.serialize_floating(self.chirp_duration, "chirp_duration")
        process.serialize_floating(self.pulse_rep_interval, "pulse_rep_interval")
        process.serialize_floating(self.sampling_rate, "sampling_rate")
        process.serialize_floating(self.adc_sampling_rate, "adc_sampling_rate")

    @classmethod
    @override
    def Deserialize(cls, process):
        return cls(
            process.deserialize_integer("num_chirps", cls.__DEFAULT_NUM_CHIRPS),
            process.deserialize_floating("bandwidth", cls.__DEFAULT_BANDWIDTH),
            process.deserialize_floating("chirp_duration", cls.__DEFAULT_CHIRP_DURATION),
            process.deserialize_floating("pulse_rep_interval", cls.__DEFAULT_PULSE_REP_INTERVAL),
            process.deserialize_floating("sampling_rate", cls.__DERIVE_SAMPLING_RATE),
            process.deserialize_floating("adc_sampling_rate", cls.__DERIVE_SAMPLING_RATE),
        )
