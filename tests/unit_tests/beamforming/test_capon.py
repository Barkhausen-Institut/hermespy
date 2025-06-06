# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock, patch, PropertyMock

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.constants import pi

from hermespy.beamforming import CaponBeamformer
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


class TestCaponBeamformer(TestCase):
    """Test the Capon beamformer implementation"""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

        self.device = SimulatedDevice(carrier_frequency=1e9, antennas=SimulatedUniformArray(SimulatedIdealAntenna, 0.01, (5, 5, 1)))

        self.loading = 1e-4
        self.beamformer = CaponBeamformer(loading=self.loading)
        self.beamformer.loading = 1e-4

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""

        self.assertEqual(self.loading, self.beamformer.loading)

    def test_static_properties(self) -> None:
        """Static properties should report the correct values"""

        self.assertEqual(1, self.beamformer.num_receive_focus_points)
        self.assertEqual(1, self.beamformer.num_receive_output_streams(15))

    def test_loading_validation(self) -> None:
        """Loading property setter should raise ValueError on negative arguments"""

        with self.assertRaises(ValueError):
            self.beamformer.loading = -1.0

    def test_loading_setget(self) -> None:
        """Loading property getter should return setter argument"""

        expected_loading = 1.234
        self.beamformer.loading = expected_loading

        self.assertEqual(expected_loading, self.beamformer.loading)

    def test_decode(self) -> None:
        """Encoding and decoding towards identical angles should recover the signal"""

        focus_angles = 0.25 * pi * np.array([[0.0, 0.0], [0, 1], [1, 1], [1, -1], [1, 2], [1, -2]])
        expected_samples = np.exp(2j * pi * self.rng.uniform(0, 1, 10))
        noise = 1e-2j * self.rng.normal(size=(25, 10)) + 1e-2j * self.rng.normal(size=(25, 10))

        carrier_frequency = 1e9

        for f, focus_angle in enumerate(focus_angles):
            response = self.device.antennas.spherical_phase_response(carrier_frequency, focus_angle[0], focus_angle[1])
            noiseless_spatial_samples = response[:, np.newaxis] @ expected_samples[np.newaxis, :]
            noisy_spatial_samples = noiseless_spatial_samples + noise

            noiseless_decoded_samples = self.beamformer._decode(noiseless_spatial_samples, carrier_frequency, focus_angles[:, np.newaxis, :], self.device.antennas)
            noisy_decoded_samples = self.beamformer._decode(noisy_spatial_samples, carrier_frequency, focus_angles[:, np.newaxis, :], self.device.antennas)

            noisy_directive_power = np.linalg.norm(noisy_decoded_samples, axis=(1, 2))

            self.assertEqual(f, np.argmax(noisy_directive_power))
            assert_array_almost_equal(expected_samples, noiseless_decoded_samples[f, 0, :])

    def test_serialization(self) -> None:
        """Test capon beamformer serialization"""

        test_roundtrip_serialization(self, self.beamformer)
