# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Any

import numpy as np

from .fading import MultipathFadingChannel

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

    yaml_tag = "Exponential"

    __exponential_truncation: float = 1e-5
    __tap_interval: float
    __rms_delay: float

    def __init__(
        self, tap_interval: float, rms_delay: float, gain: float = 1.0, **kwargs: Any
    ) -> None:
        """
        Args:

            tap_interval (float):
                Tap interval in seconds.

            rms_delay (float):
                Root-Mean-Squared delay in seconds.

            gain (float, optional):
                Linear power gain factor a signal experiences when being propagated over this realization.
                :math:`1.0` by default.

            \**kwargs (Any):
                `MultipathFadingChannel` initialization parameters.

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
            gain=gain,
            delays=delays,
            power_profile=power_profile,
            rice_factors=rice_factors,
            **kwargs,
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
