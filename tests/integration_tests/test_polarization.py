# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal

from typing import Tuple

from hermespy.core import Antenna, Signal, UniformArray
from hermespy.channel import SpatialDelayChannel
from hermespy.simulation import SimulationScenario

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class HorizontallyPolarizedAntenna(Antenna):
    
    def characteristics(self, azimuth: float, elevation) -> np.ndarray:
        
        return np.array([1., 0.], dtype=float)


class TestSingleAntennaPolarization(TestCase):
    
    def setUp(self) -> None:
        
        scenario = SimulationScenario()
        
        self.device_alpha = scenario.new_device(carrier_frequency=1e9, antennas=UniformArray(HorizontallyPolarizedAntenna, 1., [1, 1, 1]))
        self.device_beta = scenario.new_device(carrier_frequency=1e9, antennas=UniformArray(HorizontallyPolarizedAntenna, 1., [1, 1, 1]))
        
        self.channel = SpatialDelayChannel()
        scenario.set_channel(self.device_beta, self.device_alpha, self.channel)
        
    def test_translation(self) -> None:
        
        test_signal = Signal(np.ones(100), 1, carrier_frequency=1e9)
        expected_power = test_signal.power
        
        self.device_alpha.position = np.zeros(3)
        position_candidates = 100 * np.array([[0, 0, 1],
                                              [0, 1, 0],
                                              [1, 0, 0],
                                              [0, 0, -1],
                                              [0, -1, 0],
                                              [-1, 0, 0]], dtype=float)
        
        for position in position_candidates:
            
            self.device_beta.position = position
            propagated_signal, _, _ = self.channel.propagate(test_signal)
        
            assert_array_almost_equal(expected_power, propagated_signal[0].power)

    def test_rotation(self) -> None:
        
        self.device_alpha.position = np.array([-100, 0, -100])
        self.device_beta.position = np.array([100, 0, 100])
        test_signal = Signal(np.ones(100), 1, carrier_frequency=1e9)
        
        orientation_candidates = np.pi * np.array([[0., .5, 0.],
                                                   [0., -.5, 0.],
                                                   [0., 0., 0.],
                                                   [0., 1., 0.,],
                                                   [0., .25, 0.],
                                                   [0., -.25, 0]], dtype=float)
        
        expected_powers = np.array([0., 0., 1., 1., .5, .5])
        
        powers = np.empty(orientation_candidates.shape[0], dtype=float)
        for o, orientation in enumerate(orientation_candidates):
            
            self.device_beta.orientation = orientation
            propagated_signal, _, _ = self.channel.propagate(test_signal)
            power = propagated_signal[0].power

            powers[o] = power
            
        assert_array_almost_equal(expected_powers, powers)
