# -*- coding: utf-8 -*-

from __future__ import annotations
from fractions import Fraction
from typing_extensions import override

import numpy as np
from scipy.constants import speed_of_light, pi
from scipy.fft import fft2, ifftshift
from scipy.signal import decimate, resample_poly

from hermespy.core import ReceiveState, Signal, SerializationProcess, TransmitState
from .radar import RadarWaveform

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
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

    .. literalinclude:: ../../scripts/examples/radar_fmcw_FMCW.py
        :language: python
        :linenos:
        :lines: 03-23
    """

    __DEFAULT_NUM_CHIRPS = 10
    __DEFAULT_CHIRP_DURATION = 1.5e-6
    __DEFAULT_PULSE_REP_INTERVAL = 1.5e-6
    __DERIVE_SAMPLING_RATE: float = 0.0
    """Magic number at which FMCW waveforms will automatically derive the sampling rates."""

    __num_chirps: int  # Number of chirps per radar frame
    __chirp_duration: float  # chirp duration in seconds
    __pulse_rep_interval: float  # pulse repetition interval in seconds
    __adc_sampling_rate: float  # sampling rate of ADC after mixing

    def __init__(
        self,
        num_chirps: int = __DEFAULT_NUM_CHIRPS,
        chirp_duration: float = __DEFAULT_CHIRP_DURATION,
        pulse_rep_interval: float = __DEFAULT_PULSE_REP_INTERVAL,
        adc_sampling_rate: float | None = None,
    ) -> None:
        """
        Args:

            num_chirps:
                Number of dedicated chirps within a single radar frame.
                :math:`10` by default.

            chirp_duration:
                Duration of every chirp in seconds.
                :math:`1.5~\\mathrm{\\mu s}` by default.

            pulse_rep_interval:
                Repetition interval of the individual chirps in seconds.
                :math:`1.5~\\mathrm{\\mu s}` by default.

            adc_sampling_rate:
                Sampling rate of the analog-digital conversion in Hz.
                If not specified, the adc sampling rate will be equal to the bandwidth.
        """

        # Initialize base class
        RadarWaveform.__init__(self)

        # Initialize attributes
        self.num_chirps = num_chirps
        self.chirp_duration = chirp_duration
        self.pulse_rep_interval = pulse_rep_interval
        self.adc_sampling_rate = (
            FMCW.__DERIVE_SAMPLING_RATE if adc_sampling_rate is None else adc_sampling_rate
        )

    @override
    def ping(self, state: TransmitState) -> Signal:
        return Signal.Create(
            self.__frame_prototype(state.bandwidth, state.oversampling_factor), state.sampling_rate
        )

    @override
    def estimate(self, signal: Signal, state: ReceiveState) -> np.ndarray:
        num_pulse_samples = int(self.pulse_rep_interval * state.sampling_rate)
        num_frame_samples = self.num_chirps * num_pulse_samples

        input_samples = (
            signal[0:1, :num_frame_samples].view(np.ndarray)
            if signal.num_samples >= num_frame_samples
            else np.append(
                signal[0:1, :].view(np.ndarray),
                np.zeros((1, num_frame_samples - signal.num_samples)),
                axis=1,
            )
        )

        chirp_stack = np.reshape(input_samples, (-1, num_pulse_samples))

        chirp_prototype = self.__chirp_prototype(state.bandwidth, state.oversampling_factor)
        chirp_stack = chirp_stack[:, : len(chirp_prototype)]
        baseband_samples = chirp_stack.conj() * chirp_prototype

        if self.adc_sampling_rate > 0 and self.adc_sampling_rate < state.sampling_rate:
            downsampling_rate = Fraction(
                self.adc_sampling_rate / state.sampling_rate
            ).limit_denominator(1000)
            baseband_samples = resample_poly(
                baseband_samples, downsampling_rate.numerator, downsampling_rate.denominator, axis=1
            )
            _adc_bandwidth = self.adc_sampling_rate
        else:
            baseband_samples = decimate(
                baseband_samples, state.oversampling_factor, axis=1, zero_phase=False
            )
            _adc_bandwidth = state.bandwidth

        num_bins = int(self.max_range(_adc_bandwidth) / self.range_resolution(state.bandwidth))
        transform = fft2(baseband_samples, (baseband_samples.shape[0], num_bins))
        # fft shift the Doppler components
        transform = ifftshift(transform, axes=0)

        return np.abs(transform) ** 2

    @override
    def max_range(self, bandwidth: float) -> float:
        adc_bandwidth = self.adc_sampling_rate if self.adc_sampling_rate > 0 else bandwidth
        max_range = adc_bandwidth * speed_of_light / (2 * bandwidth / self.chirp_duration)
        return max_range

    @override
    def range_resolution(self, bandwidth: float) -> float:
        return speed_of_light / (2 * bandwidth)

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
        resolution = 1 / (2 * self.frame_duration(0.0))
        return resolution

    @property
    @override
    def relative_doppler_bins(self) -> np.ndarray:
        return (
            np.arange(self.num_chirps) * self.relative_doppler_resolution
            - self.max_relative_doppler
        )

    @override
    def frame_duration(self, bandwidth: float) -> float:
        return self.pulse_rep_interval * self.num_chirps

    @override
    def samples_per_frame(self, bandwidth: float, oversampling_factor: int) -> int:
        return int(self.frame_duration(bandwidth) * bandwidth * oversampling_factor)

    @override
    def range_bins(self, bandwidth: float) -> np.ndarray:
        # The actual bandwidth may be limited by the ADC assumptions
        adc_bandwidth = (
            bandwidth
            if self.adc_sampling_rate <= FMCW.__DERIVE_SAMPLING_RATE
            else self.adc_sampling_rate
        )
        max_range = self.max_range(adc_bandwidth)
        num_bins = int(max_range / self.range_resolution(bandwidth))
        return np.linspace(0, max_range, num_bins)

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
    def chirp_duration(self) -> float:
        """Duration of a single chirp within the FMCW frame.

        In combination with :meth:`pulse_rep_interval<.pulse_rep_interval>` the chirp duration
        determines the guard interval between two consecutive chirps:

        .. plot:: scripts/plots/radar_fmcw_chirp_duration.py

        Raises:

            ValueError: For durations smaller or equal to zero.
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
            ValueError: If sampling rate is smaller than zero.
        """

        return self.__adc_sampling_rate

    @adc_sampling_rate.setter
    def adc_sampling_rate(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("ADC Sampling rate must be non-negative")

        self.__adc_sampling_rate = value

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

    def energy(self, bandwidth: float, oversampling_factor: int) -> float:
        return self.num_chirps * int(self.chirp_duration * bandwidth * oversampling_factor)

    @property
    def power(self) -> float:
        return 1.0

    def __chirp_prototype(self, bandwidth: float, oversampling_factor: int) -> np.ndarray:
        """Prototype function to generate a single chirp.

        Args:
            bandwidth: Sweep bandwidth of the chirp in Hz.
            oversampling_factor: Oversampling factor of the pulse.

        Returns:
            np.ndarray: Chirp prototype in baseband.
        """

        sampling_rate = bandwidth * oversampling_factor
        num_samples = int(self.chirp_duration * sampling_rate)
        timestamps = np.arange(num_samples) / sampling_rate

        # baseband chirp between -B/2 and B/2
        slope = bandwidth / self.chirp_duration
        chirp = np.exp(1j * pi * (-bandwidth * timestamps + slope * timestamps**2))
        return chirp

    def __pulse_prototype(self, bandwidth: float, oversampling_factor: int) -> np.ndarray:
        """Prototype function to generate a single pulse including chirp and guard interval.

        Args:
            bandwidth: Sweep bandwidth of the chirp in Hz.
            oversampling_factor: Oversampling factor of the pulse.

        Raises:
            RuntimeError: If the pulse repetition interval is smaller than the chirp duration.
        """

        num_zero_samples = int(
            (self.pulse_rep_interval - self.chirp_duration) * bandwidth * oversampling_factor
        )

        if num_zero_samples < 0:
            raise RuntimeError(
                "Pulse repetition interval cannot be less than chirp duration in FMCW radar"
            )

        pulse = np.concatenate(
            (self.__chirp_prototype(bandwidth, oversampling_factor), np.zeros(num_zero_samples))
        )
        return pulse

    def __frame_prototype(self, bandwidth: float, oversampling_factor: int) -> np.ndarray:
        """Prototype function to generate a single radar frame."""

        frame = np.tile(
            self.__pulse_prototype(bandwidth, oversampling_factor), (1, self.num_chirps)
        )
        return frame

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_integer(self.num_chirps, "num_chirps")
        process.serialize_floating(self.chirp_duration, "chirp_duration")
        process.serialize_floating(self.pulse_rep_interval, "pulse_rep_interval")
        process.serialize_floating(self.adc_sampling_rate, "adc_sampling_rate")

    @classmethod
    @override
    def Deserialize(cls, process):
        return cls(
            process.deserialize_integer("num_chirps", cls.__DEFAULT_NUM_CHIRPS),
            process.deserialize_floating("chirp_duration", cls.__DEFAULT_CHIRP_DURATION),
            process.deserialize_floating("pulse_rep_interval", cls.__DEFAULT_PULSE_REP_INTERVAL),
            process.deserialize_floating("adc_sampling_rate", cls.__DERIVE_SAMPLING_RATE),
        )
