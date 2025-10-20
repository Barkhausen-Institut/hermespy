# -*- coding: utf-8 -*-

from __future__ import annotations
from typing_extensions import override

from ..block import RFBlock, RFBlockRealization, RFBlockPort, RFBlockPortType
from ..signal import RFSignal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class X4(RFBlock):
    """A radio-frequency block that multiplies the input signal's frequency by 4."""

    __i: RFBlockPort[X4]
    __o: RFBlockPort[X4]

    def __init__(self, seed: int | None = None) -> None:
        """
        Args:
            seed: Seed with which to initialize the block's random state.
        """

        # Initialize base class
        RFBlock.__init__(self, seed=seed)

        # Initialize input and output ports
        self.__i = RFBlockPort(self, 0, RFBlockPortType.IN)
        self.__o = RFBlockPort(self, 0, RFBlockPortType.OUT)

    @property
    def i(self) -> RFBlockPort[X4]:
        """Input port of the X4 block."""
        return self.__i

    @property
    def o(self) -> RFBlockPort[X4]:
        """Output port of the X4 block."""
        return self.__o

    @property
    @override
    def num_input_ports(self) -> int:
        return 1

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

    @override
    def _propagate(self, realization: RFBlockRealization, input: RFSignal) -> RFSignal:
        # Multiply the input signal's frequency by 4
        output: RFSignal = input.copy()
        output.carrier_frequencies = input.carrier_frequencies * 4
        return output
