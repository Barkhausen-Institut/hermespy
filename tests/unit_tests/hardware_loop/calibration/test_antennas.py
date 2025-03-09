# -*- coding: utf-8 -*-

from __future__ import annotations
from unittest import TestCase

import numpy as np
from scipy.constants import speed_of_light

from hermespy.channel import SpatialDelayChannel
from hermespy.core import AntennaMode, Transformation
from hermespy.simulation import SimulatedUniformArray, SimulatedIdealAntenna, StaticTrajectory
from hermespy.hardware_loop import ScalarAntennaCalibration, PhysicalDeviceDummy, PhysicalScenarioDummy
from unit_tests.core.test_factory import test_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestScalarAntennaCalibration(TestCase):
    """Test the scalar antenna calibration"""

    calibrated_device: PhysicalDeviceDummy
    scenario: PhysicalScenarioDummy

    def setUp(self) -> None:

        self.rng = np.random.default_rng(42)
        self.carrier_frequency = 6e9
        self.sampling_rate = 400e6
        self.wavelength = speed_of_light / self.carrier_frequency

        self.calibrated_antennas = SimulatedUniformArray(SimulatedIdealAntenna, .5 * self.wavelength, [3, 2, 1])
        for antenna in self.calibrated_antennas.antennas[:3]:
            antenna.mode = AntennaMode.TX
        for antenna in self.calibrated_antennas.antennas[3:]:
            antenna.mode = AntennaMode.RX

        self.calibrated_device = PhysicalDeviceDummy(
            antennas=self.calibrated_antennas,
            carrier_frequency=self.carrier_frequency,
            sampling_rate=self.sampling_rate,
            pose=StaticTrajectory(Transformation.From_Translation(np.zeros(3))),
        )
        self.calibrated_device.pose = Transformation.From_Translation(np.zeros(3))

        self.reference_antennas = SimulatedUniformArray(SimulatedIdealAntenna, .5 * self.wavelength, [2, 1, 1])
        self.reference_antennas.antennas[0].mode = AntennaMode.TX
        self.reference_antennas.antennas[1].mode = AntennaMode.RX

        self.reference_device = PhysicalDeviceDummy(
            antennas=self.reference_antennas,
            carrier_frequency=self.carrier_frequency,
            sampling_rate=self.sampling_rate,
            pose=StaticTrajectory(Transformation.From_Translation(np.array([0, 0, 2]))),
        )
        self.reference_device.pose = Transformation.From_Translation(np.array([0, 0, 2]))

        self.scenario = PhysicalScenarioDummy(42, [self.calibrated_device, self.reference_device])
        self.scenario.set_channel(self.calibrated_device, self.reference_device, SpatialDelayChannel(model_propagation_loss=False))

    def test_estimate(self) -> None:
        """Estimation routine should correctly estimate the calibration matrix"""

        # Set the calibration matrices to random weights
        expected_transmit_weights = np.exp(1j * self.rng.uniform(0, 2 * np.pi, self.calibrated_antennas.num_transmit_antennas))
        expected_receive_weights = np.exp(1j * self.rng.uniform(0, 2 * np.pi, self.calibrated_antennas.num_receive_antennas))
        for antenna, weight in zip(self.calibrated_antennas.transmit_antennas, expected_transmit_weights):
            antenna.weight = weight
        for antenna, weight in zip(self.calibrated_antennas.receive_antennas, expected_receive_weights):
            antenna.weight = weight

        scalar_calibration = ScalarAntennaCalibration.Estimate(self.scenario, self.calibrated_device, self.reference_device)

        # Check that the estimated relative phase correction is correct
        relative_transmit_weights = expected_transmit_weights[1:] / expected_transmit_weights[0]
        relative_receive_weights = expected_receive_weights[1:] / expected_receive_weights[0]
        relative_receive_calibration = scalar_calibration.receive_correction_weights[1:] / scalar_calibration.receive_correction_weights[0]
        relative_transmit_calibration = scalar_calibration.transmit_correction_weights[1:] / scalar_calibration.transmit_correction_weights[0]
        np.testing.assert_allclose(relative_transmit_weights, 1/relative_transmit_calibration, atol=1e-2)
        np.testing.assert_allclose(relative_receive_weights, 1/relative_receive_calibration, atol=1e-2)

    def test_serialization(self) -> None:
        """Test scalar antenna calibration serialization"""
        
        transmit_weights = np.exp(1j * self.rng.uniform(0, 2 * np.pi, self.calibrated_antennas.num_transmit_antennas))
        receive_weights = np.exp(1j * self.rng.uniform(0, 2 * np.pi, self.calibrated_antennas.num_receive_antennas))
        calibration = ScalarAntennaCalibration(transmit_weights, receive_weights)
        test_roundtrip_serialization(self, calibration)
