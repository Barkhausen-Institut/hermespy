# -*- coding: utf-8 -*-

from __future__ import annotations
from typing_extensions import override

import numpy as np

from hermespy.core import DeserializationProcess, SerializationProcess
from .fading import AntennaCorrelation, MultipathFadingChannel
from ..channel import Channel

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Exponential(MultipathFadingChannel):
    """Exponential multipath fading channel model."""

    __exponential_truncation: float = 1e-5
    __tap_interval: float
    __rms_delay: float

    def __init__(
        self,
        tap_interval: float,
        rms_delay: float,
        correlation_distance: float = MultipathFadingChannel._DEFAULT_DECORRELATION_DISTANCE,
        num_sinusoids: int = MultipathFadingChannel._DEFAULT_NUM_SINUSOIDS,
        los_angle: float | None = None,
        doppler_frequency: float = MultipathFadingChannel._DEFAULT_DOPPLER_FREQUENCY,
        los_doppler_frequency: float | None = None,
        antenna_correlation: AntennaCorrelation | None = None,
        gain: float = MultipathFadingChannel._DEFAULT_GAIN,
        seed: int | None = None,
    ) -> None:
        """
        Args:

            tap_interval (float):
                Tap interval in seconds.

            rms_delay (float):
                Root-Mean-Squared delay in seconds.

            correlation_distance (float, optional):
                Distance at which channel samples are considered to be uncorrelated.
                :math:`\\infty` by default, i.e. the channel is considered to be fully correlated in space.

            num_sinusoids (int, optional):
                Number of sinusoids used to sample the statistical distribution.
                Denoted by :math:`N` within the respective equations.

            los_angle (float, optional):
                Angle phase of the line of sight component within the statistical distribution.

            doppler_frequency (float, optional):
                Doppler frequency shift of the statistical distribution.
                Denoted by :math:`\\omega_{\\ell}` within the respective equations.

            antenna_correlation (AntennaCorrelation, optional):
                Antenna correlation model.
                By default, the channel assumes ideal correlation, i.e. no cross correlations.

            gain (float, optional):
                Linear power gain factor a signal experiences when being propagated over this realization.
                :math:`1.0` by default.

            seed (int, optional):
                Seed used to initialize the pseudo-random number generator.

        Raises:
            ValueError: On invalid arguments.
        """

        if tap_interval <= 0.0:
            raise ValueError("Tap interval must be greater than zero")

        if rms_delay <= 0.0:
            raise ValueError("Root-Mean-Squared delay must be greater than zero")

        self.__tap_interval = tap_interval
        self.__rms_delay = rms_delay

        rms_norm = rms_delay / tap_interval

        # Calculate the decay exponent alpha based on an infinite power delay profile, in which case
        # rms_delay = exp(-alpha/2)/(1-exp(-alpha)), cf. geometric distribution.
        # Truncate the distributions for paths whose average power is very
        # small (less than exponential_truncation).
        alpha = -2 * np.log((-1 + np.sqrt(1 + 4 * rms_norm**2)) / (2 * rms_norm))
        max_delay_in_samples = int(-np.ceil(np.log(Exponential.__exponential_truncation) / alpha))

        delays = np.arange(max_delay_in_samples + 1) * tap_interval
        power_profile = np.exp(-alpha * np.arange(max_delay_in_samples + 1))
        rice_factors = np.zeros(delays.shape)

        # Init base class with pre-defined model parameters
        MultipathFadingChannel.__init__(
            self,
            delays,
            power_profile,
            rice_factors,
            correlation_distance,
            num_sinusoids,
            los_angle,
            doppler_frequency,
            los_doppler_frequency,
            antenna_correlation,
            gain,
            seed,
        )

    @property
    def tap_interval(self) -> float:
        """Tap interval.

        Returns: Tap interval in seconds.
        """

        return self.__tap_interval

    @property
    def rms_delay(self) -> float:
        """Root mean squared channel delay.

        Returns: Delay in seconds.
        """

        return self.__rms_delay

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_floating(self.tap_interval, "tap_interval")
        process.serialize_floating(self.rms_delay, "rms_delay")
        process.serialize_floating(self.correlation_distance, "correlation_distance")
        process.serialize_integer(self.num_sinusoids, "num_sinusoids")
        process.serialize_floating(self.los_angle, "los_angle")
        process.serialize_floating(self.doppler_frequency, "doppler_frequency")
        process.serialize_floating(self.los_doppler_frequency, "los_doppler_frequency")
        if self.antenna_correlation is not None:
            process.serialize_object(self.antenna_correlation, "antenna_correlation")
        Channel.serialize(self, process)

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> Exponential:
        return cls(
            tap_interval=process.deserialize_floating("tap_interval"),
            rms_delay=process.deserialize_floating("rms_delay"),
            correlation_distance=process.deserialize_floating(
                "correlation_distance", cls._DEFAULT_DECORRELATION_DISTANCE
            ),
            num_sinusoids=process.deserialize_integer("num_sinusoids", cls._DEFAULT_NUM_SINUSOIDS),
            los_angle=process.deserialize_floating("los_angle"),
            doppler_frequency=process.deserialize_floating(
                "doppler_frequency", cls._DEFAULT_DOPPLER_FREQUENCY
            ),
            los_doppler_frequency=process.deserialize_floating("los_doppler_frequency"),
            antenna_correlation=process.deserialize_object(
                "antenna_correlation", AntennaCorrelation, None
            ),
            **Channel._DeserializeParameters(process),  # type: ignore[arg-type]
        )
