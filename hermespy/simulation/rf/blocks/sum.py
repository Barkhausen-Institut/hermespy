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


class Sum(RFBlock):
    """A simple radio-frequency block that sums all input signals."""

    __num_input_ports: int
    __i: RFBlockPort[Sum]
    __o: RFBlockPort[Sum]

    def __init__(self, num_input_ports: int, seed: int | None = None) -> None:
        """
        Args:
            num_input_ports: Number of input ports to the block.
            seed: Seed with which to initialize the block's random state.
        """

        # Initialize base class
        super().__init__(seed=seed)

        # Store the number of input ports
        self.__num_input_ports = num_input_ports
        self.__i = RFBlockPort(self, range(num_input_ports), RFBlockPortType.IN)
        self.__o = RFBlockPort(self, 0, RFBlockPortType.OUT)

    @property
    @override
    def num_input_ports(self) -> int:
        return self.__num_input_ports

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
        return np.sum(input, axis=0, keepdims=True)

    @property
    def i(self) -> RFBlockPort[Sum]:
        """Input ports of the summation block.

        Number of ports is equal to the number of input ports specified during initialization.
        """

        return self.__i

    @property
    def o(self) -> RFBlockPort[Sum]:
        """Output port of the summation block."""

        return self.__o
