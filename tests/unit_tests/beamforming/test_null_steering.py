# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.constants import pi

from hermespy.core import Signal
from hermespy.simulation import SimulatedDevice, SimulatedIdealAntenna, SimulatedUniformArray
from hermespy.beamforming import BeamformingReceiver, BeamformingTransmitter, NullSteeringBeamformer, SphericalFocus

__author__ = "Alan Thomas"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Alan Thomas", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestNullSteeringBeamformer(TestCase):
    """Test the NullSteering beamformer implementation"""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

        self.carrier_frequency = 1e9
        self.device = SimulatedDevice(
            carrier_frequency=self.carrier_frequency,
            antennas=SimulatedUniformArray(SimulatedIdealAntenna, 0.01, (5, 5, 1)),
        )

        self.beamformer = NullSteeringBeamformer()
    
        self.expected_samples = Signal.Create(np.exp(2j * pi * self.rng.uniform(0, 1, (1, 5))), self.device.sampling_rate, self.device.carrier_frequency)
        self.transmitter = BeamformingTransmitter(self.expected_samples, self.beamformer)
        self.receiver = BeamformingReceiver(self.beamformer, self.expected_samples.num_samples, self.device.sampling_rate)

        self.device.add_dsp(self.transmitter)
        self.device.add_dsp(self.receiver)

    def test_static_properties(self) -> None:
        """Static properties should report the correct values"""

        self.assertEqual(3, self.beamformer.num_transmit_focus_points)
        self.assertEqual(3, self.beamformer.num_receive_focus_points)
        self.assertEqual(1, self.beamformer._num_transmit_input_streams(5))
        self.assertEqual(1, self.beamformer.num_receive_output_streams(10))

    def test_null_steering_effectiveness(self) -> None:
        """Test whether the NullSteeringBeamformer significantly nulls a1 and a2 directions"""

        # Define the focus angles: a0 (maximum power direction), a1, and a2 (null directions)
        focus_angles = np.array([
            [1.5, 0.75],  # Target direction
            [-1.5, 0.75],  # First null direction
            [0.0, -1.0],  # Second null direction
        ])
        # Configure the beamformer to focus the respective angles
        self.beamformer.transmit_focus = [SphericalFocus(a) for a in focus_angles]
        self.beamformer.receive_focus = [SphericalFocus(a) for a in focus_angles]
    
        # Compute thet array responses for the focus angles
        # Compute spherical phase responses for a0, a1, and a2
        focus_responses = np.array([self.device.antennas.spherical_phase_response(self.carrier_frequency, a[0], a[1]) for a in focus_angles])

        # Test transmission
        transmission = self.device.transmit().mixed_signal.getitem()
        beamformed_powers = np.array([np.linalg.norm(focus_response @ transmission) ** 2 for focus_response in focus_responses])
        self.assertGreater(beamformed_powers[0], beamformed_powers[1] * 10, "a0 power should be at least 10x greater than a1 power (nulling issue).")
        self.assertGreater(beamformed_powers[0], beamformed_powers[2] * 10, "a0 power should be at least 10x greater than a2 power (nulling issue).")

        # Test reception
        received_powers = np.array([self.device.receive(Signal.Create(focus_response[:, None], self.device.sampling_rate, self.device.carrier_frequency)).operator_receptions[0].signal.power[0] for focus_response in focus_responses])
        self.assertGreater(received_powers[0], received_powers[1] * 10, "a0 power should be at least 10x greater than a1 power (nulling issue).")
        self.assertGreater(received_powers[0], received_powers[2] * 10, "a0 power should be at least 10x greater than a2 power (nulling issue).")
