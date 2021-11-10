# -*- coding: utf-8 -*-
"""Test prototype for waveform generation modeling."""

import unittest
from unittest.mock import Mock
from typing import Tuple

import numpy as np
import numpy.random as rnd
from scipy.constants import pi
from math import ceil

from hermespy.modem import WaveformGenerator

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class WaveformGeneratorDummy(WaveformGenerator):
    """Dummy of the abstract waveform generator base class for testing."""

    def __init__(self, **kwargs) -> None:
        """Waveform Generator Dummy object initialization."""

        WaveformGenerator.__init__(self, **kwargs)

    def samples_in_frame(self) -> int:
        return 20

    def bits_per_frame(self) -> int:
        return 8

    def symbols_per_frame(self) -> int:
        return 1

    def bit_energy(self) -> float:
        return 1/8

    def symbol_energy(self) -> float:
        return 1.0

    def power(self) -> float:
        return 1.0

    def map(self, data_bits: np.ndarray) -> np.ndarray:
        return data_bits

    def unmap(self, data_symbols: np.ndarray) -> np.ndarray:
        return data_symbols

    def modulate(self, data_symbols: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        return data_symbols

    def demodulate(self,
                   signal: np.ndarray,
                   impulse_response: np.ndarray,
                   noise_variance: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return signal, impulse_response, np.repeat(noise_variance, signal.shape)

    def bandwidth(self) -> float:
        return 100.


class TestWaveformGenerator(unittest.TestCase):
    """Test the communication waveform generator unit."""

    def setUp(self) -> None:

        self.rnd = rnd.default_rng(42)

        self.modem = Mock()
        self.modem.scenario.sampling_rate = 1e6

        self.waveform_generator = WaveformGeneratorDummy(modem=self.modem)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as attributes."""

        self.assertIs(self.modem, self.waveform_generator.modem)

    def test_oversampling_factor(self) -> None:
        """Oversampling factor property getter should return setter argument."""

        oversampling_factor = 4
        self.waveform_generator.oversampling_factor = oversampling_factor

        self.assertEqual(oversampling_factor, self.waveform_generator.oversampling_factor)

    def test_modulation_order_setget(self) -> None:
        """Modulation order property getter should return setter argument."""

        order = 256
        self.waveform_generator.modulation_order = order

        self.assertEqual(order, self.waveform_generator.modulation_order)

    def test_modulation_order_validation(self) -> None:
        """Modulation order property setter should raise ValueErrors on arguments which aren't powers of two."""

        with self.assertRaises(ValueError):
            self.waveform_generator.modulation_order = 0

        with self.assertRaises(ValueError):
            self.waveform_generator.modulation_order = -1

        with self.assertRaises(ValueError):
            self.waveform_generator.modulation_order = 20

    def test_synchronize(self) -> None:
        """Default synchronization routine should properly split signals into frame-sections."""

        frames

        signal = np.exp(2j * self.rnd.uniform(0, pi))