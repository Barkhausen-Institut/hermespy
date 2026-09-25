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


class WaldenADCPowerModel(DCPowerModel):
    """Model of an analog-to-digital converter's direct current power consumption.

    Converter power consumption is commonly characterized by the Walden figure of merit

    .. math::

        \\mathrm{FOM} = \\frac{P_\\mathrm{DC}}{f_\\mathrm{s} 2^{b}} \\ \\text{,}

    relating the consumed power to the sampling rate :math:`f_\\mathrm{s}` and the
    amplitude resolution :math:`b`. Surveys of published converters exhibit an empirical
    lower bound on the figure of merit that is approximately constant at low bandwidths
    and degrades quadratically beyond a corner frequency. Assuming Nyquist rate sampling,
    so that the sampling rate equals the bandwidth :math:`B`, this bound implies

    .. math::

        P_\\mathrm{DC} = c \\, 2^{b} B \\sqrt{1 + \\left(\\frac{B}{f_\\mathrm{b}}\\right)^2}
        \\ \\text{.}

    The consumed power therefore grows exponentially with the amplitude resolution and,
    in bandwidth, linearly at first and quadratically once the corner frequency is passed.

    The model is independent of the processed signal, so that the consumed power is
    fully determined by the converter's configuration. Note that the expression is a
    lower bound rather than a prediction: a specific converter consumes more.

    Default coefficients follow the survey reported in Florian Gast, *Optimizing
    Transceiver Energy Efficiency with a Gearbox Physical Layer*, Dissertation,
    Technische Universitaet Dresden, 2025, equations 4.6 to 4.8.
    """

    __num_quantization_bits: int
    __bandwidth: float
    __figure_of_merit: float
    __corner_frequency: float

    def __init__(
        self,
        num_quantization_bits: int,
        bandwidth: float,
        figure_of_merit: float = 0.67e-15,
        corner_frequency: float = 560e6,
    ) -> None:
        """
        Args:
            num_quantization_bits:
                Amplitude resolution of the converter in bits.
                Should match the resolution configured at the modeled block.

            bandwidth:
                Bandwidth processed by the converter in Hz.
                Nyquist rate sampling is assumed, so that the sampling rate equals
                the bandwidth.

            figure_of_merit:
                Walden figure of merit in Joule per conversion step.
                Defaults to the empirical lower bound of :math:`0.67 \\cdot 10^{-15}`.

            corner_frequency:
                Bandwidth in Hz beyond which the figure of merit degrades.
                Defaults to :math:`560` MHz.

        Raises:
            ValueError:
                If the resolution is not positive,
                or if the bandwidth, figure of merit or corner frequency is not positive.
        """

        self.num_quantization_bits = num_quantization_bits
        self.bandwidth = bandwidth
        self.figure_of_merit = figure_of_merit
        self.corner_frequency = corner_frequency

    @property
    def num_quantization_bits(self) -> int:
        """Amplitude resolution of the converter in bits.

        Raises:
            ValueError: If the resolution is not positive.
        """

        return self.__num_quantization_bits

    @num_quantization_bits.setter
    def num_quantization_bits(self, value: int) -> None:
        if value <= 0:
            raise ValueError("Amplitude resolution must be positive")

        self.__num_quantization_bits = int(value)

    @property
    def bandwidth(self) -> float:
        """Bandwidth processed by the converter in Hz.

        Raises:
            ValueError: If the bandwidth is not positive.
        """

        return self.__bandwidth

    @bandwidth.setter
    def bandwidth(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("Converter bandwidth must be positive")

        self.__bandwidth = float(value)

    @property
    def figure_of_merit(self) -> float:
        """Walden figure of merit in Joule per conversion step.

        Raises:
            ValueError: If the figure of merit is not positive.
        """

        return self.__figure_of_merit

    @figure_of_merit.setter
    def figure_of_merit(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("Walden figure of merit must be positive")

        self.__figure_of_merit = float(value)

    @property
    def corner_frequency(self) -> float:
        """Bandwidth beyond which the figure of merit degrades in Hz.

        Raises:
            ValueError: If the corner frequency is not positive.
        """

        return self.__corner_frequency

    @corner_frequency.setter
    def corner_frequency(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("Corner frequency must be positive")

        self.__corner_frequency = float(value)

    @property
    def power(self) -> float:
        """Direct current power consumed by the converter in Watt.

        Fully determined by the converter's configuration.
        """

        return float(
            self.__figure_of_merit
            * 2**self.__num_quantization_bits
            * self.__bandwidth
            * np.sqrt(1.0 + (self.__bandwidth / self.__corner_frequency) ** 2)
        )

    @override
    def get_power(
        self, input_signal: np.ndarray[tuple[int, ...], np.dtype[np.complex128]]
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        return np.full(input_signal.shape, self.power, dtype=np.float64)

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_integer(self.__num_quantization_bits, "num_quantization_bits")
        process.serialize_floating(self.__bandwidth, "bandwidth")
        process.serialize_floating(self.__figure_of_merit, "figure_of_merit")
        process.serialize_floating(self.__corner_frequency, "corner_frequency")

    @classmethod
    @override
    def Deserialize(
        cls: Type[WaldenADCPowerModel], process: DeserializationProcess
    ) -> WaldenADCPowerModel:
        return cls(
            process.deserialize_integer("num_quantization_bits"),
            process.deserialize_floating("bandwidth"),
            process.deserialize_floating("figure_of_merit"),
            process.deserialize_floating("corner_frequency"),
        )
