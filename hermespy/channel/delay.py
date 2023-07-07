# -*- coding: utf-8 -*-
"""
=============
Delay Channel
=============
"""

from __future__ import annotations
from abc import abstractmethod
from typing import Tuple

import numpy as np
from scipy.constants import speed_of_light

from hermespy.channel import Channel, ChannelRealization
from hermespy.tools import amplitude_path_loss

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class DelayChannelRealization(ChannelRealization):
    """Realization of a delay channel model."""

    ...  # pragma: no cover


class DelayChannelBase(Channel[DelayChannelRealization]):
    """Base of delay channel models."""

    yaml_tag = "DelayChannelBase"

    __model_propagation_loss: bool

    def __init__(self, model_propagation_loss: bool = True, **kwargs) -> None:
        """
        Args:

            model_propagation_loss (bool, optional):
                Should free space propagation loss be modeled?
                Enabled by default.

            **kawrgs:
                :class:`Channel` base class initialization arguments.
        """

        # Initialize base class
        Channel.__init__(self, **kwargs)

        # Initialize class attributes
        self.__model_propagation_loss = model_propagation_loss

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

    @property
    def model_propagation_loss(self) -> bool:
        """Should free space propagation loss be modeled?

        Returns: Enabled flag.
        """

        return self.__model_propagation_loss

    @model_propagation_loss.setter
    def model_propagation_loss(self, value: bool) -> None:
        self.__model_propagation_loss = value

    def realize(self, num_samples: int, sampling_rate: float) -> DelayChannelRealization:
        delay = self._realize_delay()
        delay_samples = int(delay * sampling_rate)

        loss_factor = 1.0
        if self.model_propagation_loss:
            carrier_frequency = self.transmitter.carrier_frequency

            if carrier_frequency == 0.0:
                raise RuntimeError("Transmitting device's carrier frequency may not be zero, disable propagation path loss modeling")

            loss_factor = amplitude_path_loss(carrier_frequency, delay * speed_of_light)

        spatial_response = self._realize_response()

        time_response = np.zeros((num_samples, 1 + delay_samples), dtype=complex)
        time_response[:, -1] = loss_factor * np.sqrt(self.gain)

        # The impulse response is an elment-wise matrix multiplication
        # exploding two two-dimensional matrices into a four-dimensional tensor
        channel_impulse_response = np.einsum("ab,cd->abcd", spatial_response, time_response)

        realization = DelayChannelRealization(self, channel_impulse_response)
        return realization


class SpatialDelayChannel(DelayChannelBase):
    """Delay channel based on spatial dimensions.

    The spatial delay channel requires both linked devices to specify their assumed positions.
    """

    yaml_tag: str = "SpatialDelay"

    def _realize_delay(self) -> float:
        if self.transmitter is None or self.receiver is None:
            raise RuntimeError("The spatial delay channel requires the linked devices positions to be specified")

        distance = float(np.linalg.norm(self.transmitter.global_position - self.receiver.global_position))
        delay = distance / speed_of_light

        return delay

    def _realize_response(self) -> np.ndarray:
        transmit_response = self.transmitter.antennas.cartesian_array_response(self.transmitter.carrier_frequency, self.receiver.global_position, "global")
        receive_response = self.receiver.antennas.cartesian_array_response(self.receiver.carrier_frequency, self.transmitter.global_position, "global")

        return receive_response @ transmit_response.T


class RandomDelayChannel(DelayChannelBase):
    """Delay channel based on random delays."""

    yaml_tag: str = "RandomDelay"

    __delay: float | Tuple[float, float]

    def __init__(self, delay: float | Tuple[float, float], *args, **kwargs) -> None:
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
            if value < 0.0:
                raise ValueError(f"Delay must be greater or equal to zero (not {value})")

        elif isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError("Delay limit tuple must contain two entries")

            if any(v < 0.0 for v in value):
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
