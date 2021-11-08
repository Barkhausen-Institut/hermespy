# -*- coding: utf-8 -*-
"""Test HermesPy simulation executable."""

import unittest
import numpy as np
from numpy.testing import assert_array_equal

from hermespy.simulator_core import Simulation, SNRType

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestSimulation(unittest.TestCase):
    """Test the simulation executable, the base class for simulation operations."""

    def setUp(self) -> None:

        self.simulation = Simulation()

    def test_snr_type_setget(self) -> None:
        """SNR type property getter should return setter argument."""

        for snr_type in SNRType:

            # Enum set
            self.simulation.snr_type = snr_type
            self.assertEqual(snr_type, self.simulation.snr_type)

            # String set
            self.simulation.snr_type = str(snr_type.name)
            self.assertEqual(snr_type, self.simulation.snr_type)

    def test_noise_loop_setget(self) -> None:
        """Noise loop property getter should return setter argument for both arrays and lists."""

        noise_loop = np.arange(20)

        # Set via vector
        self.simulation.noise_loop = noise_loop
        assert_array_equal(noise_loop, self.simulation.noise_loop)

        # Set via list
        self.simulation.noise_loop = noise_loop.tolist()
        assert_array_equal(noise_loop, self.simulation.noise_loop)

    def test_noise_loop_validation(self) -> None:
        """Noise loop property setter should raise ValueErrors on invalid arguments."""

        with self.assertRaises(ValueError):
            self.simulation.noise_loop = np.empty(0, dtype=float)

        with self.assertRaises(ValueError):
            self.simulation.noise_loop = np.zeros((2, 2), dtype=float)

        with self.assertRaises(ValueError):
            self.simulation.noise_loop = []

        with self.assertRaises(ValueError):
            self.simulation.noise_loop = [[2, 3], [4, 5]]

    def test_to_yaml(self) -> None:
        """Test YAML serialization dump validity."""
        pass

    def test_from_yaml(self) -> None:
        """Test YAML serialization recall validity."""
        pass
