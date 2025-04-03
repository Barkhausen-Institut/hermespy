# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock, patch, PropertyMock

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.constants import pi

from hermespy.beamforming import ConventionalBeamformer
from hermespy.simulation import SimulatedDevice, SimulatedIdealAntenna, SimulatedUniformArray
from unit_tests.core.test_factory import test_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestConventionalBeamformer(TestCase):
    """Test the conventional beamformer implementation"""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.device = SimulatedDevice(carrier_frequency=1e9)
        self.device.antennas = SimulatedUniformArray(SimulatedIdealAntenna, 0.01, (5, 5, 1))
        self.beamformer = ConventionalBeamformer()

    def test_static_properties(self) -> None:
        """Static properties should report the correct values."""

        self.assertEqual(1, self.beamformer.num_receive_focus_points)
        self.assertEqual(1, self.beamformer.num_receive_output_streams(10))
        self.assertEqual(1, self.beamformer.num_transmit_focus_points)
        self.assertEqual(1, self.beamformer.num_transmit_input_streams(10))

    def test_encode_decode(self) -> None:
        """Encoding and decoding towards identical angles should recover the signal"""

        focus_angles = [np.array([[[0.0, 0.0]]])]
        expected_samples = np.exp(2j * pi * self.rng.uniform(0, 1, (1, 5)))
        carrier_frequency = 1e9

        for focus_angle in focus_angles:
            encoded_samples = self.beamformer._encode(expected_samples, carrier_frequency, focus_angle[0, :], self.device.antennas)
            decoded_samples = self.beamformer._decode(encoded_samples, carrier_frequency, focus_angle, self.device.antennas)

            assert_array_almost_equal(expected_samples, decoded_samples[0, ::])

    def test_serialization(self) -> None:
        """Test conventional beamformer serialization"""

        test_roundtrip_serialization(self, self.beamformer)
