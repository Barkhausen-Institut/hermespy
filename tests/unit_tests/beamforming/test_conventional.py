# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.constants import pi

from hermespy.core import UniformArray, IdealAntenna
from hermespy.beamforming import ConventionalBeamformer

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.7"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestConventionalBeamformer(TestCase):
    """Test the conventional beamformer implementation"""
    
    def setUp(self) -> None:
        
        self.rng = np.random.default_rng(42)
        
        self.operator = Mock()
        self.operator.device = Mock()
        self.operator.device.antennas = UniformArray(IdealAntenna(), .01, (5, 1, 1))
        
        self.beamformer = ConventionalBeamformer(operator=self.operator)
        
    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""
        
        self.assertIs(self.operator, self.beamformer.operator)
        
    def test_static_properties(self) -> None:
        """Static properties should report the correct values."""
        
        self.assertEqual(1, self.beamformer.num_receive_focus_angles)
        self.assertEqual(5, self.beamformer.num_receive_input_streams)
        self.assertEqual(1, self.beamformer.num_receive_output_streams)
        self.assertEqual(1, self.beamformer.num_transmit_focus_angles)
        self.assertEqual(5, self.beamformer.num_transmit_output_streams)
        self.assertEqual(1, self.beamformer.num_transmit_input_streams)
        
    def test_encode_decode(self) -> None:
        """Encoding and decoding towards identical angles should recover the signal"""
        
        focus_angles = [np.array([[[0., 0.]]])]
        expected_samples = np.exp(2j * pi * self.rng.uniform(0, 1, (1, 5)))
        carrier_frequency = 1e9
        
        for focus_angle in focus_angles:
            
            encoded_samples = self.beamformer._encode(expected_samples, carrier_frequency, focus_angle[0, :])
            decoded_samples = self.beamformer._decode(encoded_samples, carrier_frequency, focus_angle)
            
            assert_array_almost_equal(expected_samples, decoded_samples[0, ::])
