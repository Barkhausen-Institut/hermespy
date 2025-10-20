# -*- coding: utf-8 -*-

from __future__ import annotations
from functools import cache
from typing_extensions import override

import numpy as np

from hermespy.core import SerializationProcess, DeserializationProcess
from ..block import RFBlock, RFBlockRealization, RFBlockPort, RFBlockPortType
from ..signal import RFSignal
from ..noise import NoiseModel, NoiseLevel

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RampGenerator(RFBlock):
    """FMCW ramp generator block model."""

    __num_chirps: int
    __chirp_bandwidth: float
    __chirp_slope: float
    __chirp_interval: float
    __o: RFBlockPort[RampGenerator]

    def __init__(
        self,
        num_chirps: int,
        chirp_bandwidth: float,
        chirp_slope: float,
        chirp_interval: float,
        noise_model: NoiseModel | None = None,
        noise_level: NoiseLevel | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Args:

            num_chirps: Number of dedicated FMCW chirps per generated frame.
            chirp_bandwidth: Frequency range each chirp covers in Hz.
            chirp_slope: Slope of the FMCW chirp in Hz/s.
            chirp_interval: Interval between the start of two consecutive chirps in seconds.
            seed: Seed with which to initialize the block's random state.
        """

        # Initialize base class
        RFBlock.__init__(self, noise_model, noise_level, seed)

        # Initialize class attributes
        self.num_chirps = num_chirps
        self.chirp_bandwidth = chirp_bandwidth
        self.chirp_slope = chirp_slope
        self.chirp_interval = chirp_interval

        # Initialize output port
        self.__o = RFBlockPort(self, 0, RFBlockPortType.OUT)

    @property
    def chirp_bandwidth(self) -> float:
        """Frequency range each FMCW chirp covers in Hz.

        Raises:
            ValueError: If the chirp bandwidth is not positive.
        """

        return self.__chirp_bandwidth

    @chirp_bandwidth.setter
    def chirp_bandwidth(self, value: float) -> None:
        if value <= 0:
            raise ValueError("FMCW chirp bandwidth must be positive")

        self.__chirp_bandwidth = value

    @property
    def chirp_slope(self) -> float:
        """Slope of the FMCW chirp in Hz/s."""

        return self.__chirp_slope

    @chirp_slope.setter
    def chirp_slope(self, value: float) -> None:
        self.__chirp_slope = value

    @property
    def chirp_interval(self) -> float:
        """Interval between the start of two consecutive chirps in seconds.

        Raises:
            ValueError: If the chirp interval is not greater than zero.
        """

        return self.__chirp_interval

    @chirp_interval.setter
    def chirp_interval(self, value: float) -> None:
        if value <= 0:
            raise ValueError("FMCW chirp interval must be greater than zero")
        self.__chirp_interval = value

    @property
    def num_chirps(self) -> int:
        """Number of dedicated FMCW chirps per generated frame.

        Raises:
            ValueError: If the number of chirps is negative.
        """

        return self.__num_chirps

    @num_chirps.setter
    def num_chirps(self, value: int) -> None:
        if value < 0:
            raise ValueError("Number of FMCW chirps must be non-negative")

        self.__num_chirps = value

    @property
    def o(self) -> RFBlockPort[RampGenerator]:
        """Output port of the ramp generator."""

        return self.__o

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
    ) -> RFBlockRealization:
        return RFBlockRealization(
            bandwidth,
            oversampling_factor,
            self.noise_model.realize(self.noise_level.get_power(bandwidth)),
        )

    @cache
    @staticmethod
    def _chirp(
        sampling_rate: float, bandwidth: float, slope: float, chirp_interval: float
    ) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
        """Compute the prototype samples of a single FMCW chirp.

        Args:
            sampling_rate: Sampling rate of the chirp in Hz.
            bandwidth: Frequency range the chirp covers in Hz.
            slope: Slope of the chirp in Hz/s.
            chirp_rate: Interval between the start of two consecutive chirps in seconds.

        Returns:
            Base-band samples of the FMCW chirp.
        """

        num_guarded_chirp_samples = int(chirp_interval * sampling_rate)
        chirp_duration = bandwidth / slope
        num_chirp_samples = min(num_guarded_chirp_samples, int(chirp_duration * sampling_rate))

        chirp_samples = np.zeros(num_guarded_chirp_samples, dtype=np.complex128)

        # Base-band chirp between -B/2 and B/2
        start_freq = -bandwidth if slope > 0 else bandwidth
        timestamps = np.arange(num_chirp_samples) / sampling_rate
        chirp_samples[:num_chirp_samples] = np.exp(
            1j * np.pi * (start_freq * timestamps + slope * timestamps**2)
        )

        return chirp_samples

    @cache
    def _fmcw_frame(
        self, sampling_rate: float, bandwidth: float, slope: float, interval: float, num_chirps: int
    ) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
        """Compute the base-band samples of a single FMCW frame.

        Args:
            sampling_rate: Sampling rate of the chirp in Hz.
            bandwidth: Frequency range the chirp covers in Hz.
            slope: Slope of the chirp in Hz/s.
            rate: Interval between the start of two consecutive chirps in seconds.
            num_chirps: Number of dedicated FMCW chirps per generated frame.
        """

        chirp = RampGenerator._chirp(sampling_rate, bandwidth, slope, interval)
        frame = np.tile(chirp, num_chirps).flatten()
        return frame

    @override
    def _propagate(self, realization: RFBlockRealization, input: RFSignal) -> RFSignal:
        frame = self._fmcw_frame(
            realization.sampling_rate,
            self.chirp_bandwidth,
            self.chirp_slope,
            self.chirp_interval,
            self.num_chirps,
        )
        multidim_frame = frame.reshape((1, -1))  # Required for proper typing
        return RFSignal.FromNDArray(multidim_frame, realization.sampling_rate)

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_floating(self.chirp_bandwidth, "chirp_bandwidth")
        process.serialize_floating(self.chirp_slope, "chirp_slope")
        process.serialize_floating(self.chirp_interval, "chirp_interval")
        process.serialize_integer(self.num_chirps, "num_chirps")
        if self.seed is not None:
            process.serialize_integer(self.seed, "seed")

    @classmethod
    @override
    def Deserialize(cls: type[RampGenerator], process: DeserializationProcess) -> RampGenerator:
        return cls(
            num_chirps=process.deserialize_integer("num_chirps"),
            chirp_bandwidth=process.deserialize_floating("chirp_bandwidth"),
            chirp_slope=process.deserialize_floating("chirp_slope"),
            chirp_interval=process.deserialize_floating("chirp_interval"),
            seed=process.deserialize_integer("seed", None),
        )
