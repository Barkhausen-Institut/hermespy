# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Set, Tuple

from h5py import Group

from hermespy.core import HDFSerializable
from ..channel import ChannelSampleHook, LinkState
from ..consistent import ConsistentGenerator, ConsistentRealization, ConsistentUniform
from .delay import DelayChannelBase, DelayChannelRealization, DelayChannelSample

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RandomDelayChannelRealization(DelayChannelRealization):
    """Realization of a random delay channel.

    Generated from :class:`RandomDelayChannel's<RandomDelayChannel>` :meth:`_realize<RandomDelayChannel._realize>` routine.
    """

    def __init__(
        self,
        consistent_realization: ConsistentRealization,
        delay_variable: ConsistentUniform,
        delay: float | Tuple[float, float],
        model_propagation_loss: bool,
        sample_hooks: Set[ChannelSampleHook[DelayChannelSample]],
        gain: float,
    ) -> None:

        # Initialize base class
        DelayChannelRealization.__init__(self, model_propagation_loss, sample_hooks, gain)

        # Store attributes
        self.__consistent_realization = consistent_realization
        self.__delay_variable = delay_variable
        self.__delay = delay

    @property
    def delay(self) -> float | Tuple[float, float]:
        """Assumed propagation delay in seconds.

        If set to a scalar floating point, the delay is assumed to be constant.
        If set to a tuple of two floats, the tuple values indicate the mininum and maxium values of a uniform distribution, respectively.

        Raises:

            ValueError: If the delay is set to a negative value.
            ValueError: If the delay is set to a tuple of two values where the first value is greater than the second value.
        """

        return self.__delay

    def _sample(self, state: LinkState) -> DelayChannelSample:

        if isinstance(self.__delay, float):
            delay = self.__delay

        else:
            # Sample the consistent realization
            consistent_sample = self.__consistent_realization.sample(
                state.transmitter.position, state.receiver.position
            )

            # Realize the delay
            delay = float(
                self.__delay[0]
                + (self.__delay[1] - self.__delay[0])
                * self.__delay_variable.sample(consistent_sample)
            )

        # Generate a sample
        return DelayChannelSample(delay, self.model_propagation_loss, self.gain, state)

    def to_HDF(self, group: Group) -> None:
        # Serialize base class
        DelayChannelRealization.to_HDF(self, group)

        # Serialize attributes
        self.__consistent_realization.to_HDF(
            HDFSerializable._create_group(group, "consistent_realization")
        )
        HDFSerializable._range_to_HDF(group, "delay", self.__delay)

    @staticmethod
    def From_HDF(
        group: Group,
        delay_variable: ConsistentUniform,
        sample_hooks: Set[ChannelSampleHook[DelayChannelSample]],
    ) -> RandomDelayChannelRealization:

        # Deserialize attributes
        consistent_realization = ConsistentRealization.from_HDF(group["consistent_realization"])
        delay = HDFSerializable._range_from_HDF(group, "delay")

        # Return the realization
        return RandomDelayChannelRealization(
            consistent_realization,
            delay_variable,
            delay,
            group.attrs["model_propagation_loss"],
            sample_hooks,
            group.attrs["gain"],
        )


class RandomDelayChannel(DelayChannelBase[RandomDelayChannelRealization]):
    """Delay channel assuming random propagation delays."""

    yaml_tag: str = "RandomDelay"

    __delay: float | Tuple[float, float]

    def __init__(
        self,
        delay: float | Tuple[float, float],
        decorrelation_distance: float = float("inf"),
        **kwargs,
    ) -> None:
        """
        Args:

            delay (float | Tuple[float, float]):
                Assumed propagation delay in seconds.
                If a scalar floating point, the delay is assumed to be constant.
                If a tuple of two floats, the tuple values indicate the mininum and maxium values of a uniform distribution, respectively.

            decorrelation_distance (float, optional):
                Distance in meters at which the channel decorrelates.
                By default, the channel is assumed to be static in space.

            **kwargs:
                :class:`.Channel` base class initialization parameters.
        """

        # Initialize base class
        DelayChannelBase.__init__(self, **kwargs)

        # Store attributes
        self.delay = delay
        self.decorrelation_distance = decorrelation_distance
        self.__consistent_generator = ConsistentGenerator(self)
        self.__delay_variable = self.__consistent_generator.uniform()

    @property
    def delay(self) -> float | Tuple[float, float]:
        """Assumed propagation delay in seconds.

        If set to a scalar floating point, the delay is assumed to be constant.
        If set to a tuple of two floats, the tuple values indicate the mininum and maxium values of a uniform distribution, respectively.

        Raises:

            ValueError: If the delay is set to a negative value.
            ValueError: If the delay is set to a tuple of two values where the first value is greater than the second value.
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
                raise ValueError(
                    f"Delay must be greater or equal to zero (not {value[0]} and {value[1]})"
                )

            if value[0] > value[1]:
                raise ValueError("First tuple entry must be smaller or equal to second tuple entry")

        else:
            raise ValueError("Unsupported value type")

        self.__delay = value

    @property
    def decorrelation_distance(self) -> float:
        """Distance in meters at which the channel decorrelates.

        Raises:

            ValueError: If the decorrelation distance is set to a negative value.
        """

        return self.__decorrelation_distance

    @decorrelation_distance.setter
    def decorrelation_distance(self, value: float) -> None:
        if value < 0.0:
            raise ValueError(
                f"Decorrelation distance must be greater or equal to zero (not {value})"
            )

        self.__decorrelation_distance = value

    def _realize(self) -> RandomDelayChannelRealization:

        # Realize the consistent generator
        consistent_realization = self.__consistent_generator.realize(self.decorrelation_distance)

        # Return the realization
        return RandomDelayChannelRealization(
            consistent_realization,
            self.__delay_variable,
            self.__delay,
            self.model_propagation_loss,
            self.sample_hooks,
            self.gain,
        )

    def recall_realization(self, group: Group) -> RandomDelayChannelRealization:
        return RandomDelayChannelRealization.From_HDF(
            group, self.__delay_variable, self.sample_hooks
        )
