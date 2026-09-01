# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence, Type
from typing_extensions import override


import numpy as np

from hermespy.core import Serializable, SerializationProcess, DeserializationProcess

__author__ = "Jan Adler"
__copyright__ = "Copyright 2026, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.6.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class DCPowerModel(ABC, Serializable):
    """Base class for all direct-current power consumption models.

    Power consumption models describe the amount of direct current power an
    :class:`ActiveRFBlock<hermespy.simulation.rf.block.ActiveRFBlock>`
    draws from its supply while operating at a given signal power level.

    They are the direct-current counterpart of
    :class:`NoiseLevel<hermespy.simulation.rf.noise.level.NoiseLevel>`,
    which describes the noise a block contributes to the processed signal.
    """

    @abstractmethod
    def get_power(
        self, input_signal: np.ndarray[tuple[int, ...], np.dtype[np.complex128]]
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Direct current power consumed by the block.

        Args:
            input_power: Average power of the signal feeding into the block in Watt.

        Returns: Consumed direct current power in Watt.
        """
        ...  # pragma: no cover


class NoDCPowerModel(DCPowerModel):
    """Placeholder model for blocks without configured power consumption.

    Blocks configured with this model are assumed to be ideal in the sense that
    they do not draw any direct current power, i.e. :meth:`get_power<DCPowerModel.get_power>`
    always returns zero.
    """

    @override
    def get_power(
        self, input_signal: np.ndarray[tuple[int, ...], np.dtype[np.complex128]]
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        return np.zeros(input_signal.shape, dtype=np.float64)

    @override
    def serialize(self, process: SerializationProcess) -> None:
        return

    @classmethod
    @override
    def Deserialize(cls: Type[NoDCPowerModel], process: DeserializationProcess) -> NoDCPowerModel:
        return cls()


class ConstantDCPowerModel(DCPowerModel):
    """Model of a constant direct current power consumption.

    The consumed power is independent of the processed signal,

    .. math::

        P_\\mathrm{DC} = P_\\mathrm{const} \\ \\text{.}

    This is an adequate approximation for blocks biased at a fixed operating point,
    such as local oscillators or class-A amplifiers driven well below compression.
    """

    __power: float

    def __init__(self, power: float) -> None:
        """
        Args:
            power: Constantly consumed direct current power in Watt.

        Raises:
            ValueError: If the power is negative.

        """
        self.power = power

    @property
    def power(self) -> float:
        """Constantly consumed direct current power in Watt.

        Raises:
            ValueError: If the power is negative.
        """
        return self.__power

    @power.setter
    def power(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("Direct current power consumption must be non-negative")

        self.__power = float(value)

    @override
    def get_power(
        self, input_signal: np.ndarray[tuple[int, ...], np.dtype[np.complex128]]
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        return np.full(input_signal.shape, self.__power, dtype=np.float64)

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_floating(self.__power, "power")

    @classmethod
    @override
    def Deserialize(
        cls: Type[ConstantDCPowerModel], process: DeserializationProcess
    ) -> ConstantDCPowerModel:
        return cls(process.deserialize_floating("power"))


class SampledDCPowerModel(DCPowerModel):
    """Model of a direct current power consumption sampled from a datasheet lookup table.

    Datasheets of active radio-frequency components commonly specify the consumed direct
    current power for a discrete set of signal power levels. This model linearly interpolates
    between those sampling points,

    .. math::

        P_\\mathrm{DC}(P_\\mathrm{in}) = P_k + (P_\\mathrm{in}- S_k)
        \\frac{P_{k+1}-P_k}{S_{k+1}-S_k}
        \\quad \\text{for} \\quad S_k \\leq P_\\mathrm{in}<S_{k+1} \\ \\text{,}

    with :math:`S` denoting the sampled input powers and :math:`P` the respective
    consumed powers. Input powers outside of the sampled range are clipped to the
    respective boundary sample.

    The model is evaluated per sample, so that the nonlinearity of the
    lookup table is correctly reflected for modulated signals.


    """

    __input_powers: np.ndarray
    __consumed_powers: np.ndarray

    def __init__(
        self,
        input_powers: Sequence[float] | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
        consumed_powers: Sequence[float] | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    ) -> None:
        """
        Args:
            input_powers:
                Vector of sampled signal powers at the block's input in Watt.
                Must be sorted in strictly ascending order.

            consumed_powers:
                Vector of direct current powers consumed at the respective input powers in Watt.
                Must be of identical length as 'input_powers'.

        Raises:
            ValueError:
                If the arguments are not of identical length,
                if fewer than two sampling points are provided,
                if the input powers are not strictly ascending,
                or if any of the powers is negative.
        """
        _input_powers = np.asarray(input_powers, dtype=np.float64).flatten()
        _consumed_powers = np.asarray(consumed_powers, dtype=np.float64).flatten()

        if _input_powers.size != _consumed_powers.size:
            raise ValueError(
                "Sampled input and consumed powers must be vectors of identical length "
                f"({_input_powers.size} != {_consumed_powers.size})"
            )

        if _input_powers.size < 2:
            raise ValueError(
                "Sampled power consumption model requires at least two sampling points, "
                "consider a ConstantDCPowerModel instead"
            )

        if np.any(np.diff(_input_powers) <= 0.0):
            raise ValueError("Sampled input powers must be sorted in strictly ascending order")

        if np.any(_input_powers < 0.0):
            raise ValueError("Sampled input powers must be non-negative")

        if np.any(_consumed_powers < 0.0):
            raise ValueError("Sampled consumed powers must be non-negative")

        # The arrays are treated as immutable, so that they may safely be cached during serialization

        _input_powers.flags.writeable = False
        _consumed_powers.flags.writeable = False

        self.__input_powers = _input_powers
        self.__consumed_powers = _consumed_powers

    @property
    def input_powers(self) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Vector of sampled signal powers at the block's input in Watt."""

        return self.__input_powers

    @property
    def consumed_powers(self) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Vector of direct current powers consumed at the respective input powers in Watt."""

        return self.__consumed_powers

    @override
    def get_power(
        self, input_signal: np.ndarray[tuple[int, ...], np.dtype[np.complex128]]
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        instantaneous_power = np.abs(input_signal) ** 2
        return np.interp(instantaneous_power, self.input_powers, self.consumed_powers)

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_array(self.__input_powers, "input_powers")
        process.serialize_array(self.__consumed_powers, "consumed_powers")

    @classmethod
    @override
    def Deserialize(
        cls: Type[SampledDCPowerModel], process: DeserializationProcess
    ) -> SampledDCPowerModel:
        return cls(
            process.deserialize_array("input_powers", np.float64),
            process.deserialize_array("consumed_powers", np.float64),
        )
