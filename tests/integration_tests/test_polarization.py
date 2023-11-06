# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal

from hermespy.core import Antenna, Signal, UniformArray
from hermespy.channel import SpatialDelayChannel
from hermespy.simulation import SimulationScenario

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class HorizontallyPolarizedAntenna(Antenna):
    
    def local_characteristics(self, azimuth: float, elevation) -> np.ndarray:
        
        return np.array([1., 0.], dtype=float)
    
class VerticallyPolarizedAntenna(Antenna):
    
    def local_characteristics(self, azimuth: float, elevation) -> np.ndarray:
        
        return np.array([1., 0.], dtype=float)


class TestSingleAntennaPolarization(TestCase):
    
    def setUp(self) -> None:
        
        scenario = SimulationScenario()
        
        self.device_alpha = scenario.new_device(carrier_frequency=1e9, antennas=UniformArray(HorizontallyPolarizedAntenna, 1., [1, 1, 1]))
        self.device_beta = scenario.new_device(carrier_frequency=1e9, antennas=UniformArray(HorizontallyPolarizedAntenna, 1., [1, 1, 1]))
        
        self.channel = SpatialDelayChannel(model_propagation_loss=False, alpha_device=self.device_alpha, beta_device=self.device_beta, seed=42)
        scenario.set_channel(self.device_beta, self.device_alpha, self.channel)
        
        self.orientation_candidates = np.pi * np.array([[0., 0., 0.],
                                                        [.5, 0., 0.],
                                                        [-.5, 0., 0.],
                                                        [1, 0., 0.],
                                                        [-1, 0., 0.],
                                                        [0., .5, 0.],
                                                        [0., -.5, 0.],
                                                        [0., 1., 0.],
                                                        [0., -1., 0.],
                                                        [0., 0., .5],
                                                        [0., 0., -.5],
                                                        [0., 0., 1.],
                                                        [0., 0., -1.]], dtype=float)
        
        self.test_signal = Signal(np.ones(100), 1, carrier_frequency=1e9)
        
    def test_translation(self) -> None:
        
        expected_power = self.test_signal.power
        
        self.device_alpha.position = np.zeros(3)
        position_candidates = 100 * np.array([[1, 1, 1],
                                              [0, 0, 1],
                                              [0, 1, 0],
                                              [1, 0, 0],
                                              [0, 0, -1],
                                              [0, -1, 0],
                                              [-1, 0, 0]], dtype=float)
        
        powers = np.empty(position_candidates.shape[0], dtype=float)
        for p, position in enumerate(position_candidates):
            
            self.device_beta.position = position
            propagation = self.channel.propagate(self.test_signal)
            
            powers[p] = propagation.signal.power
        
        assert_array_almost_equal(expected_power * np.ones(position_candidates.shape[0]), powers)

    def __assert_rotation_power(self, expected_powers: np.ndarray) -> None:

        powers = np.empty(self.orientation_candidates.shape[0], dtype=float)
        for o, orientation in enumerate(self.orientation_candidates):
            
            self.device_beta.orientation = orientation
            propagation = self.channel.propagate(self.test_signal)
            
            powers[o] = propagation.signal.power
            
        assert_array_almost_equal(expected_powers, powers)

    def test_rotation_horizontal_polarization(self) -> None:
        
        self.device_alpha.antennas = UniformArray(HorizontallyPolarizedAntenna, 1., [1, 1, 1])
        self.device_beta.antennas = UniformArray(HorizontallyPolarizedAntenna, 1., [1, 1, 1])

        # For propagations along the x-axis only the rotation around the x-axis should be relevant for polarization loss
        self.device_alpha.position = np.array([100, 0, 0])
        self.device_beta.position = np.array([0, 0, 0])
        expected_powers_x = np.array([1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
        self.__assert_rotation_power(expected_powers_x)
        
        # For propagations along the y-axis only the rotation around the x-axis should be relevant for polarization loss
        self.device_alpha.position = np.array([0, 0, 0])
        self.device_beta.position = np.array([0, 100, 0.])
        expected_powers_y = np.array([1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1.])
        self.__assert_rotation_power(expected_powers_y)
        
        # For propagations along the z-axis only the rotation around the x-axis should be relevant for polarization loss
        self.device_alpha.position = np.array([0, 0, 0])
        self.device_beta.position = np.array([0, 0, 100])
        expected_powers_z = np.array([1., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 1., 1.])
        self.__assert_rotation_power(expected_powers_z)

    def test_rotation_vertical_polarization(self) -> None:
        
        self.device_alpha.antennas = UniformArray(VerticallyPolarizedAntenna, 1., [1, 1, 1])
        self.device_beta.antennas = UniformArray(VerticallyPolarizedAntenna, 1., [1, 1, 1])
        self.device_alpha.position = np.array([-100, 0, -100])
        self.device_beta.position = np.array([100, 0, 100])
        
        expected_powers_xy = np.array([1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
        self.__assert_rotation_power(expected_powers_xy)
