# -*- coding: utf-8 -*-

from unittest import TestCase

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
        self.velocity_bins = np.array([[0.]])
        self.range_bins = np.array([0.])
        
        self.data = np.array([[[1.]]])
        
        self.cube = RadarCube(self.data, self.angle_bins, self.velocity_bins, self.range_bins)
        
    def test_init(self) -> None:
        """Initialization arguments should be properly stored as class attributes"""
        
        assert_array_almost_equal(self.angle_bins, self.cube.angle_bins)
        assert_array_almost_equal(self.velocity_bins, self.cube.velocity_bins)
        assert_array_almost_equal(self.range_bins, self.cube.range_bins)

    def test_init_validation(self) -> None:
        """Radar cube initializations should raise ValueErrors on invalid arguments"""
        
        with self.assertRaises(ValueError):
            _ = RadarCube(self.data, np.array([[1, 2], [3, 4]]), self.velocity_bins, self.range_bins)
            
        with self.assertRaises(ValueError):
            _ = RadarCube(self.data, self.angle_bins, np.array([1, 2]), self.range_bins)
    
        with self.assertRaises(ValueError):
            _ = RadarCube(self.data, self.angle_bins, self.velocity_bins, np.array([1, 2, 3]))
