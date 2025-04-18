# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Set, Tuple
from typing_extensions import override

from hermespy.core import DeserializationProcess, SerializationProcess
from ..channel import ChannelSampleHook, LinkState
from ..consistent import ConsistentGenerator, ConsistentRealization, ConsistentUniform
from .delay import DelayChannelBase, DelayChannelRealization, DelayChannelSample

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RandomDelayChannelRealization(DelayChannelRealization):
    """Realization of a random delay channel.

    Generated from :class:`RandomDelayChannel's<RandomDelayChannel>` :meth:`_realize<RandomDelayChannel._realize>` routine.
    """

    __consistent_realization: ConsistentRealization
    __delay_variable: ConsistentUniform
    __delay: float | Tuple[float, float]

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

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_object(self.__consistent_realization, "consistent_realization")
        process.serialize_object(self.__delay_variable, "delay_variable")
        process.serialize_range(self.__delay, "delay")
        DelayChannelRealization.serialize(self, process)

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> RandomDelayChannelRealization:
        return RandomDelayChannelRealization(
            process.deserialize_object("consistent_realization", ConsistentRealization),
            process.deserialize_object("delay_variable", ConsistentUniform),
            process.deserialize_range("delay"),
            sample_hooks=set(),
            **DelayChannelRealization._DeserializeParameters(process),  # type: ignore[arg-type]
        )


class RandomDelayChannel(DelayChannelBase[RandomDelayChannelRealization]):
    """Delay channel assuming random propagation delays."""

    __DEFAULT_DECORRELATION_DISTANCE = float("inf")

    __delay: float | Tuple[float, float]

    def __init__(
        self,
        delay: float | Tuple[float, float],
        decorrelation_distance: float = __DEFAULT_DECORRELATION_DISTANCE,
        model_propagation_loss: bool = True,
        gain: float = DelayChannelBase._DEFAULT_GAIN,
        seed: int | None = None,
    ) -> None:
        """
        Args:

            delay:
                Assumed propagation delay in seconds.
                If a scalar floating point, the delay is assumed to be constant.
                If a tuple of two floats, the tuple values indicate the mininum and maxium values of a uniform distribution, respectively.

            decorrelation_distance:
                Distance in meters at which the channel decorrelates.
                By default, the channel is assumed to be static in space.

            model_propagation_loss:
                Should free space propagation loss be modeled?
                Enabled by default.

            gain:
                Linear power gain factor a signal experiences when being propagated over this realization.
                :math:`1.0` by default.

            seed:
                Seed used to initialize the pseudo-random number generator.
        """

        # Initialize base class
        DelayChannelBase.__init__(self, model_propagation_loss, gain, seed)

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

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_object(self.__delay_variable, "delay_variable")
        process.serialize_range(self.delay, "delay")
        process.serialize_floating(self.__decorrelation_distance, "decorrelation_distance")
        DelayChannelBase.serialize(self, process)

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> RandomDelayChannel:
        return RandomDelayChannel(
            process.deserialize_range("delay"),
            process.deserialize_floating(
                "decorrelation_distance", cls.__DEFAULT_DECORRELATION_DISTANCE
            ),
            **DelayChannelBase._DeserializeParameters(process),  # type: ignore[arg-type]
        )
