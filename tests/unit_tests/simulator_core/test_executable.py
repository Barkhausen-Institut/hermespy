# -*- coding: utf-8 -*-
"""Test HermesPy base executable."""

import unittest
import numpy as np
from unittest.mock import Mock
from numpy.testing import assert_array_equal

from simulator_core import Executable

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class ExecutableStub(Executable):
    """Testing instantiation of the abstract Executable prototype, mock implementing all abstract functions."""

    def __init__(self, *args) -> None:
        Executable.__init__(self, *args)

    def run(self) -> None:
        pass


class TestExecutable(unittest.TestCase):
    """Test the base executable prototype, the base class for hermes operations."""

    def setUp(self) -> None:

        self.executable = ExecutableStub()

    def test_init(self) -> None:
        """Executable initialization parameters should be properly stored."""

        plot_drop = True
        calc_transmit_spectrum = True
        calc_receive_spectrum = True
        calc_transmit_stft = True
        calc_receive_stft = True
        spectrum_fft_size = 20

        executable = ExecutableStub(plot_drop, calc_transmit_spectrum, calc_receive_spectrum, calc_transmit_stft,
                                    calc_receive_stft, spectrum_fft_size)

        self.assertEqual(plot_drop, executable.plot_drop)
        self.assertEqual(calc_transmit_spectrum, executable.calc_transmit_spectrum)
        self.assertEqual(calc_receive_spectrum, executable.calc_receive_spectrum)
        self.assertEqual(calc_transmit_stft, executable.calc_transmit_stft)
        self.assertEqual(calc_receive_stft, executable.calc_receive_stft)
        self.assertEqual(spectrum_fft_size, executable.spectrum_fft_size)

    def test_add_scenario(self) -> None:
        """Scenario property should return scenarios added by the add_scenario function."""

        scenarios = [Mock() for _ in range(10)]

        for scenario in scenarios:
            self.executable.add_scenario(scenario)

        self.assertCountEqual(scenarios, self.executable.scenarios)
