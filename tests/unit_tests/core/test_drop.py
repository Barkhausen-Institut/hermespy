# -*- coding: utf-8 -*-
"""Test HermesPy Simulation Drop."""

import unittest

import numpy as np
from scipy.constants import pi

from hermespy.core.drop import Drop

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestDrop(unittest.TestCase):
    """Test simulation drop."""

    def setUp(self) -> None:

        self.rng = np.random.default_rng(42)
        self.num_transmitters = 2
        self.num_receivers = 2
        self.num_bits = 256
        self.num_symbols = 64
        self.num_samples = 128
        self.block_size = 32

        self.transmitted_bits = [self.rng.integers(0, 2, self.num_bits) for _ in range(self.num_transmitters)]
        self.received_bits = [self.rng.integers(0, 2, self.num_bits) for _ in range(self.num_receivers)]
        self.transmitted_symbols = [np.exp(2j * self.rng.uniform(0, 2*pi, self.num_symbols))
                                    for _ in range(self.num_transmitters)]
        self.received_symbols = [np.exp(2j * self.rng.uniform(0, 2*pi, self.num_symbols))
                                 for _ in range(self.num_receivers)]
        self.transmitted_signals = [np.exp(2j * self.rng.uniform(0, 2*pi, self.num_samples))
                                    for _ in range(self.num_transmitters)]
        self.received_signals = [np.exp(2j * self.rng.uniform(0, 2*pi, self.num_samples))
                                 for _ in range(self.num_receivers)]
        self.block_sizes = [self.block_size for _ in range(self.num_receivers)]

        self.drop = Drop(transmitted_bits=self.transmitted_bits, received_bits=self.received_bits,
                         transmitted_symbols=self.transmitted_symbols, received_symbols=self.received_symbols,
                         transmitted_signals=self.transmitted_signals, received_signals=self.received_signals,
                         transmit_block_sizes=self.block_sizes, receive_block_sizes=self.block_sizes)
