# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import patch, Mock

import numpy as np
from numpy.testing import assert_array_almost_equal

from hermespy.radar import RadarCube
from unit_tests.utils import SimulationTestContext

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestCube(TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        self.angle_bins = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        self.doppler_bins = np.array([0.0])
        self.range_bins = np.arange(10)
        self.carrier_frequency = 72e9

        self.data = rng.random((4, 1, 10))

        self.cube = RadarCube(self.data, self.angle_bins, self.doppler_bins, self.range_bins, self.carrier_frequency)

    def test_init(self) -> None:
        """Initialization arguments should be properly stored as class attributes"""

        assert_array_almost_equal(self.angle_bins, self.cube.angle_bins)
        assert_array_almost_equal(self.doppler_bins, self.cube.doppler_bins)
        assert_array_almost_equal(self.range_bins, self.cube.range_bins)
        self.assertEqual(self.carrier_frequency, self.cube.carrier_frequency)

    def test_angle_bin_inference(self) -> None:
        """Angle bins should be inferred from data if not provided"""

        cube = RadarCube(self.data[[0], :, :], None, self.doppler_bins, self.range_bins)

        assert_array_almost_equal(np.zeros((1, 2)), cube.angle_bins)

    def test_doppler_bin_inference(self) -> None:
        """Doppler bins should be inferred from data if not provided"""

        cube = RadarCube(self.data, self.angle_bins, None, self.range_bins)

        assert_array_almost_equal(self.doppler_bins, cube.doppler_bins)

    def test_init_validation(self) -> None:
        """Radar cube initializations should raise ValueErrors on invalid arguments"""

        with self.assertRaises(ValueError):
            _ = RadarCube(np.zeros((1, 2, 3, 4)), self.angle_bins, self.doppler_bins, self.range_bins)

        with self.assertRaises(ValueError):
            _ = RadarCube(self.data, np.array([[1, 2]]), self.doppler_bins, self.range_bins)

        with self.assertRaises(ValueError):
            _ = RadarCube(self.data, self.angle_bins, np.array([1, 2]), self.range_bins)

        with self.assertRaises(ValueError):
            _ = RadarCube(self.data, self.angle_bins, self.doppler_bins, np.array([1, 2, 3]))

        with self.assertRaises(ValueError):
            _ = RadarCube(np.zeros((2, 1, 1)), None, self.doppler_bins, self.range_bins)

        with self.assertRaises(ValueError):
            _ = RadarCube(np.zeros((1, 2, 1)), self.angle_bins, None, self.range_bins)

        with self.assertRaises(ValueError):
            _ = RadarCube(self.data, self.angle_bins, self.doppler_bins, self.range_bins, -1.0)

    def test_velocity_bins_validation(self) -> None:
        """Velocity bin comutation should raise RuntimeError if carrier frequency is unknown"""

        self.cube = RadarCube(self.data, self.angle_bins, self.doppler_bins, self.range_bins, 0.0)
        with self.assertRaises(RuntimeError):
            _ = self.cube.velocity_bins

    def test_plot_range(self) -> None:
        """Range plots should be created properly"""

        with SimulationTestContext():
            _ = self.cube.plot_range(scale="lin")
            _ = self.cube.plot_range(scale="log")

    def test_plot_range_velocity(self) -> None:
        """Range-velocity plots should be created properly"""

        with SimulationTestContext():
            _ = self.cube.plot_range_velocity()
            _ = self.cube.plot_range_velocity(scale="velocity")
            _ = self.cube.plot_range_velocity(scale="frequency")
            
    def test_plot_angles(self) -> None:
        """Angle plots should be created properly"""

        with SimulationTestContext():
            _ = self.cube.plot_angles()

    def test_normalize_power(self) -> None:
        """Power normalization should be performed properly"""

        self.cube.normalize_power()
        self.assertEqual(1, self.cube.data.max())
