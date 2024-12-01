# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock, patch, PropertyMock

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.constants import pi

from hermespy.simulation import SimulatedDevice, SimulatedIdealAntenna, SimulatedUniformArray
from hermespy.beamforming.nullsteeringbeamformer import NullSteeringBeamformer
from hermespy.core import AntennaArray
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization


class TestNullSteeringBeamformer(TestCase):
    """Test the NullSteering beamformer implementation"""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

        self.device = SimulatedDevice(
            carrier_frequency=1e9,
            antennas=SimulatedUniformArray(SimulatedIdealAntenna, 0.01, (5, 5, 1)),
        )
        self.operator = Mock()
        self.operator.device = self.device

        self.loading = 1e-4
        self.beamformer = NullSteeringBeamformer(operator=self.operator)
        self.beamformer.loading = 1e-4

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""

        self.assertIs(self.operator, self.beamformer.operator)


    def test_static_properties(self) -> None:
        """Static properties should report the correct values."""

        self.assertEqual(3, self.beamformer.num_receive_focus_points)
        self.assertEqual(25, self.beamformer.num_receive_input_streams)
        self.assertEqual(1, self.beamformer.num_receive_output_streams)
    

    def test_null_steering_effectiveness(self) -> None:
        """Test whether the NullSteeringBeamformer significantly nulls a1 and a2 directions."""

        # Define the focus angles: a0 (maximum power direction), a1, and a2 (null directions)
        a0_angle = np.array([0.0, 0.0])  # Target direction
        a1_angle = np.array([1.0, 1.0])  # First null direction
        a2_angle = np.array([1.0, -1.0])  # Second null direction
        focus_angles = np.array([a0_angle, a1_angle, a2_angle])

        expected_samples = np.exp(2j * pi * self.rng.uniform(0, 1, (1, 5)))

        # Set carrier frequency
        carrier_frequency = 1e9

        # Compute the null steering beamformer weights
        Weights = self.beamformer._weights(
            carrier_frequency, focus_angles, self.device.antennas
        )

        # Compute spherical phase responses for a0, a1, and a2
        a0_response = self.operator.device.antennas.spherical_phase_response(
            carrier_frequency, a0_angle[0], a0_angle[1]
        )
        a1_response = self.operator.device.antennas.spherical_phase_response(
            carrier_frequency, a1_angle[0], a1_angle[1]
        )
        a2_response = self.operator.device.antennas.spherical_phase_response(
            carrier_frequency, a2_angle[0], a2_angle[1]
        )

        # Applying weights to the array resposnes of the angles
        a0_beamformed = Weights.T.conj() @ a0_response
        a1_beamformed = Weights.T.conj() @ a1_response
        a2_beamformed = Weights.T.conj() @ a2_response

        # calculating the powers from each directions
        a0_power = np.linalg.norm(a0_beamformed)
        a1_power = np.linalg.norm(a1_beamformed)
        a2_power = np.linalg.norm(a2_beamformed)

        # Check that a0 power is much greater than a1 and a2 (nulling effectiveness)
        self.assertGreater(
            a0_power,
            a1_power * 10,
            "a0 power should be at least 10x greater than a1 power (nulling issue).",
        )
        self.assertGreater(
            a0_power,
            a2_power * 10,
            "a0 power should be at least 10x greater than a2 power (nulling issue).",
        )
