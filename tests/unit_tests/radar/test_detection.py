# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal

from hermespy.radar import PointDetection

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestPointDetection(TestCase):
    """Test the base class for radar point detections"""
    
    def setUp(self) -> None:
        
        self.rng = np.random.default_rng(42)
        
        self.position = self.rng.normal(size=3)
        self.velocity = self.rng.normal(size=3)
        self.power = 1.2345
        
        self.point = PointDetection(self.position, self.velocity, self.power)
        
    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""

        assert_array_equal(self.position, self.point.position)
        assert_array_equal(self.velocity, self.point.velocity)
        self.assertEqual(self.power, self.point.power)
        
    def test_init_validation(self) -> None:
        """Initialization should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            _ = PointDetection(np.zeros(2), self.velocity, self.power)

        with self.assertRaises(ValueError):
            _ = PointDetection(self.position, np.zeros(2), self.power)

        with self.assertRaises(ValueError):
            _ = PointDetection(self.position, self.velocity, 0.)
