# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

from hermespy.core import Moveable, Transformation

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestMoveable(TestCase):
    """Test moveable base class"""
    
    def setUp(self) -> None:
        
        self.transmformation = Transformation.From_RPY(np.array([1, 2, 3]), np.array([4, 5, 6]))
        self.velocity = np.array([7, 8, 9])
        
        self.moveable = Moveable(self.transmformation, self.velocity)
        
    def test_init(self) -> None:
        """Initialization parameters are stored correctly"""
        
        assert_array_equal(self.moveable.pose.translation, self.transmformation.translation)
        assert_almost_equal(self.moveable.pose.rotation_rpy, self.transmformation.rotation_rpy)
        assert_array_equal(self.moveable.velocity, self.velocity)
        
    def test_velocity_setget(self) -> None:
        """Velocity property getter should return setter argument"""
        
        expected_veloicty = np.array([10, 11, 12])
        self.moveable.velocity = expected_veloicty
        
        assert_array_equal(self.moveable.velocity, expected_veloicty)
