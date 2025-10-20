# -*- coding: utf-8 -*-

from __future__ import annotations
from typing_extensions import override

import numpy as np

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


class Shift(RFBlock):
    """A radio-frequency block that applies a phase shift to the input signal."""

    __NO_QUANTIZATION = 0
    __DEFAULT_PHASE = 0.0

    __phase: float
    __num_quantization_bits: int
    __num_ports: int
    __i: RFBlockPort[Shift]
    __o: RFBlockPort[Shift]

    def __init__(
        self,
        phase: float = __DEFAULT_PHASE,
        num_quantization_bits: int = __NO_QUANTIZATION,
        num_ports: int = 1,
        noise_model: NoiseModel | None = None,
        noise_level: NoiseLevel | None = None,
        seed: int | None = None,
    ) -> None:

        # Initialize base class
        RFBlock.__init__(self, noise_model, noise_level, seed)

        # Store attributes
        self.phase = phase
        self.num_quantization_bits = num_quantization_bits
        self.__num_ports = num_ports
        self.__i = RFBlockPort(self, range(num_ports), RFBlockPortType.IN)
        self.__o = RFBlockPort(self, range(num_ports), RFBlockPortType.OUT)

    @property
    def phase(self) -> float:
        """Phase shift applied by the block."""
        return self.__phase

    @phase.setter
    def phase(self, value: float) -> None:
        self.__phase = value

    @property
    def num_quantization_bits(self) -> int:
        """Number of quantization bits encoding the available discrete phase shift levels.

        Zero indicates that no quantization is applied, and the phase shift is continuous.
        """

        return self.__num_quantization_bits

    @num_quantization_bits.setter
    def num_quantization_bits(self, value: int) -> None:
        if value < 0:
            raise ValueError("Number of quantization bits must be non-negative")

        self.__num_quantization_bits = value

    @property
    @override
    def num_input_ports(self) -> int:
        return self.__num_ports

    @property
    @override
    def num_output_ports(self) -> int:
        return self.__num_ports

    @override
    def realize(
        self, bandwidth: float, oversampling_factor: int, carrier_frequency: float
    ) -> RFBlockRealization:
        return RFBlockRealization(
            bandwidth,
            oversampling_factor,
            self.noise_model.realize(self.noise_level.get_power(bandwidth)),
        )

    @override
    def _propagate(self, realization: RFBlockRealization, input: RFSignal) -> RFSignal:
        return input * np.exp(1j * self.__phase)

    @property
    def i(self) -> RFBlockPort[Shift]:
        """Input ports of the phase shifter.

        Number of ports is equal to the number of input ports specified during initialization.
        """

        return self.__i

    @property
    def o(self) -> RFBlockPort[Shift]:
        """Output ports of the phase shifter.

        Number of ports is equal to the number of input ports specified during initialization.
        """

        return self.__o
