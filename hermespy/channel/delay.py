# -*- coding: utf-8 -*-
"""
=============
Delay Channel
=============
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from scipy.constants import speed_of_light

from hermespy.channel import Channel

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class DelayChannelBase(ABC, Channel):
    """Base of delay channel models."""
    
    yaml_tag = u'DelayChannelBase'

    @abstractmethod
    def _realize_delay(self) -> float:
        """Generate a delay realization.

        Returns: The delay in seconds.
        """
        ...  # pragma no cover

    @abstractmethod
    def _realize_response(self) -> np.ndarray:
        """Realize the channel's spatial response.

        Returns: Two dimensional numpy array.
        """
        ...  # pragma no cover

    def impulse_response(self,
                         num_samples: int,
                         sampling_rate: float) -> np.ndarray:

        delay = self._realize_delay()
        delay_samples = int(delay * sampling_rate)

        spatial_response = self._realize_response()

        time_response = np.zeros((num_samples, 1 + delay_samples), dtype=complex)
        time_response[:, -1] = np.sqrt(self.gain)

        # The impulse response is an elment-wise matrix multiplication 
        # exploding two two-dimensional matrices into a four-dimensional tensor
        impulse = np.einsum('ab,cd->abcd', spatial_response, time_response)
        return impulse.transpose((2, 0, 1, 3,))


class SpatialDelayChannel(DelayChannelBase):
    """Delay channel based on spatial dimensions.
    
    The spatial delay channel requires both linked devices to specify their assumed positions.
    """
    
    yaml_tag: str = u'SpatialDelay'

    def _realize_delay(self) -> float:

        if self.transmitter.position is None or self.receiver.position is None:
            raise RuntimeError("The spatial delay channel requires the linked devices positions to be specified")

        distance = float(np.linalg.norm(self.transmitter.position - self.receiver.position))
        delay = distance / speed_of_light

        return delay

    def _realize_response(self) -> np.ndarray:

        transmit_response = self.transmitter.antennas.cartesian_response(self.transmitter.carrier_frequency,
                                                                         self.receiver.position - self.transmitter.position)
        receive_response = self.receiver.antennas.cartesian_response(self.receiver.carrier_frequency,
                                                                     self.transmitter.position - self.receiver.position)

        return np.outer(receive_response, transmit_response)
                                            

class RandomDelayChannel(DelayChannelBase):
    """Delay channel based on random delays."""

    yaml_tag: str = u'RandomDelay'

    __delay: float | Tuple[float, float]

    def __init__(self,
                 delay: float | Tuple[float, float],
                 *args, **kwargs) -> None:
        """
        Args:

            delay (float | Tuple[float, float]):
                Assumed propagation delay in seconds.
                If a scalar floating point, the delay is assumed to be constant.
                If a tuple of two floats, the tuple values indicate the mininum and maxium values of a uniform distribution, respectively.

            *args:
                :class:`.Channel` base class initialization parameters.
            
            **kwargs:
                :class:`.Channel` base class initialization parameters.
        """

        self.delay = delay
        DelayChannelBase.__init__(self, *args, **kwargs)

    @property
    def delay(self) -> float | Tuple[float, float]:
        """Assumed propagation delay.

        Returns: Delay in seconds.
        """

        return self.__delay

    @delay.setter
    def delay(self, value: float | Tuple[float, float]) -> None:

        if isinstance(value, float):

            if value < 0.:
                raise ValueError(f"Delay must be greater or equal to zero (not {value})")

        elif isinstance(value, tuple):

            if len(value) != 2:
                raise ValueError("Delay limit tuple must contain two entries")

            if any(v < 0. for v in value):
                raise ValueError(f"Delay must be greater or equal to zero (not {value[0]} and {value[1]})")

            if value[0] > value[1]:
                raise ValueError("First tuple entry must be smaller or equal to second tuple entry")

        else:
            raise ValueError("Unsupported value type")

        self.__delay = value

    def _realize_delay(self) -> float:

        if isinstance(self.delay, float):
            return self.delay

        if isinstance(self.delay, tuple):
            return self._rng.uniform(self.delay[0], self.delay[1])

        raise RuntimeError("Unsupported type of delay")

    def _realize_response(self) -> np.ndarray:

        return np.eye(self.receiver.num_antennas, self.transmitter.num_antennas, dtype=complex)
