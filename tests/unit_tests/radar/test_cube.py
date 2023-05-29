# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import patch, Mock

import numpy as np
from numpy.testing import assert_array_almost_equal

from hermespy.radar import RadarCube

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestCube(TestCase):
    
    def setUp(self) -> None:
        
        self.angle_bins = np.array([[0., 0.]])
        self.velocity_bins = np.array([0.])
        self.range_bins = np.arange(10)
        
        self.data = np.array([[np.arange(10)]])
        
        self.cube = RadarCube(self.data, self.angle_bins, self.velocity_bins, self.range_bins)
        
    def test_init(self) -> None:
        """Initialization arguments should be properly stored as class attributes"""
        
        assert_array_almost_equal(self.angle_bins, self.cube.angle_bins)
        assert_array_almost_equal(self.velocity_bins, self.cube.velocity_bins)
        assert_array_almost_equal(self.range_bins, self.cube.range_bins)
        
    def test_angle_bin_inference(self) -> None:
        """Angle bins should be inferred from data if not provided"""
        
        cube = RadarCube(self.data, None, self.velocity_bins, self.range_bins)
        
        assert_array_almost_equal(self.angle_bins, cube.angle_bins)
        
    def test_velocity_bin_inference(self) -> None:
        """Velocity bins should be inferred from data if not provided"""
        
        cube = RadarCube(self.data, self.angle_bins, None, self.range_bins)
        
        assert_array_almost_equal(self.velocity_bins, cube.velocity_bins)

    def test_init_validation(self) -> None:
        """Radar cube initializations should raise ValueErrors on invalid arguments"""
        
        with self.assertRaises(ValueError):
            _ = RadarCube(np.zeros((1, 2, 3, 4)), self.angle_bins, self.velocity_bins, self.range_bins)
        
        with self.assertRaises(ValueError):
            _ = RadarCube(self.data, np.array([[1, 2], [3, 4]]), self.velocity_bins, self.range_bins)
            
        with self.assertRaises(ValueError):
            _ = RadarCube(self.data, self.angle_bins, np.array([1, 2]), self.range_bins)
    
        with self.assertRaises(ValueError):
            _ = RadarCube(self.data, self.angle_bins, self.velocity_bins, np.array([1, 2, 3]))

        with self.assertRaises(ValueError):
            _ = RadarCube(np.zeros((2, 1, 1)), None, self.velocity_bins, self.range_bins)
            
        with self.assertRaises(ValueError):
            _ = RadarCube(np.zeros((1, 2, 1)), self.angle_bins, None, self.range_bins)

    def test_plot_range_validation(self) -> None:
        """Plotting range should raise ValueErrors on invalid arguments"""
        
        with self.assertRaises(ValueError):
            self.cube.plot_range(axes=Mock(), scale='invalid')

    def test_plot_range(self) -> None:
        """Range plots should be created properly"""
        
        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            
            figure = Mock()
            axes = Mock()
            mock_subplots.return_value = (figure, axes)
            
            self.cube.plot_range()
            mock_subplots.assert_called_once()
            
            mock_subplots.reset_mock()
            axes.reset_mock()
            self.cube.plot_range(axes=axes)
            mock_subplots.assert_not_called()
            
            axes.reset_mock()
            self.cube.plot_range(scale='lin')
            axes.plot.assert_called_once()
            
            axes.reset_mock()
            self.cube.plot_range(scale='log')
            axes.semilogy.assert_called_once()
    
    def test_plot_range_velocity(self) -> None:
        """Range-velocity plots should be created properly"""
        
        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            
            figure = Mock()
            axes = Mock()
            mock_subplots.return_value = (figure, axes)
            
            self.cube.plot_range_velocity()
            
            axes.pcolormesh.assert_called_once()

    def test_normalize_power(self) -> None:
        """Power normalization should be performed properly"""
        
        self.cube.normalize_power()
        self.assertEqual(1, self.cube.data.max())
