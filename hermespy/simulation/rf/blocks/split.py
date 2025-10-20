# -*- coding: utf-8 -*-

from __future__ import annotations
from typing_extensions import override

import numpy as np

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


class Split(RFBlock):
    """A simple radio-frequency block that splits the input signal into multiple outputs."""

    __num_outputs: int
    __i: RFBlockPort[Split]
    __o: RFBlockPort[Split]

    def __init__(self, num_outputs: int, seed: int | None = None) -> None:

        # Initialize base class
        RFBlock.__init__(self, seed=seed)

        # Initialize attributes
        self.__num_outputs = num_outputs
        self.__i = RFBlockPort(self, 0, RFBlockPortType.IN)
        self.__o = RFBlockPort(self, range(num_outputs), RFBlockPortType.OUT)

    @property
    @override
    def num_input_ports(self) -> int:
        return 1

    @property
    @override
    def num_output_ports(self) -> int:
        return self.__num_outputs

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
        return np.tile(input[[0], :], (self.__num_outputs, 1))  # type: ignore

    @property
    def i(self) -> RFBlockPort[Split]:
        """Splitter input port."""

        return self.__i

    @property
    def o(self) -> RFBlockPort[Split]:
        """Splitter output ports.

        Number of ports is equal to the number of outputs specified during initialization.
        """

        return self.__o
