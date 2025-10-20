# -*- coding: utf-8 -*-
"""
FMCW radar evaluation with radar channel simulation.
"""

from __future__ import annotations
from unittest import TestCase

import numpy as np

from hermespy.beamforming import ConventionalBeamformer
from hermespy.core import AntennaMode, Direction, Transformation
from hermespy.channel import MultiTargetRadarChannel, VirtualRadarTarget, FixedCrossSection
from hermespy.radar.radar import Radar
from hermespy.radar.fmcw import FMCW
from hermespy.simulation import SimulatedDevice, SimulatedIdealAntenna, SimulatedUniformArray, StaticTrajectory
from scipy.constants import speed_of_light

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class FMCWRadarSimulation(TestCase):
    def setUp(self) -> None:
        antenna_array = SimulatedUniformArray(SimulatedIdealAntenna(AntennaMode.RX), 0.5 * speed_of_light / 70e9, (8, 8))
        antenna_array.antennas[0].mode = AntennaMode.DUPLEX
        self.device = SimulatedDevice(
            carrier_frequency=70e9,
            bandwidth=3e9,
            oversampling_factor=2,
            antennas=antenna_array,
        )

        self.waveform = FMCW()
        self.beamformer = ConventionalBeamformer()

        self.radar = Radar()
        self.radar.waveform = self.waveform
        self.device.transmitters.add(self.radar)
        self.device.receivers.add(self.radar)
        self.radar.receive_beamformer = self.beamformer

        self.device.transmitters.add(self.radar)
        self.device.receivers.add(self.radar)

        self.target_range = 0.25 * self.waveform.max_range(self.device.bandwidth)
        self.channel = MultiTargetRadarChannel(attenuate=False)

        self.virtual_target = VirtualRadarTarget(FixedCrossSection(1.0), trajectory=StaticTrajectory(Transformation.From_Translation(np.array([0, 0, self.target_range]))))
        self.channel.add_target(self.virtual_target)

    def test_beamforming(self) -> None:
        """The radar channel target located should be estimated correctly by the beamformer"""

        # This test doesn't consider velocity, so a single chirp is sufficient
        self.waveform.num_chirps = 1

        focus_points = np.pi * np.mgrid[0:2:.1, .1:.5:.1].reshape((2, -1)).T
        focus_points = np.append(focus_points, np.zeros((1, 2)), axis=0)

        self.radar.receive_beamformer.probe_focus_points = focus_points[:, np.newaxis, :]
        device_state = self.device.state(0.0)

        for angle_index, (azimuth, zenith) in enumerate(focus_points):
            with self.subTest(focus_points=(azimuth, zenith)):
                # Configure the channel's only target
                self.virtual_target.trajectory = StaticTrajectory(Transformation.From_Translation(Direction.From_Spherical(azimuth, zenith) * self.target_range))

                # Generate the radar cube

                propagation = self.channel.propagate(self.device.transmit(device_state), self.device, self.device)
                reception = self.device.receive(propagation, device_state)

                directive_powers = np.linalg.norm(reception.operator_receptions[0].cube.data, axis=(1, 2))
                self.assertEqual(angle_index, directive_powers.argmax())

    def test_detection(self) -> None:
        """Test FMCW detection"""

        # Generate the radar cube
        device_state = self.device.state(0.0)
        propagation = self.channel.propagate(self.device.transmit(device_state), self.device, self.device)
        reception = self.device.receive(propagation, device_state).operator_receptions[0]
        expected_velocity_peak = 5

        range_profile = np.sum(reception.cube.data, axis=(0, 1))
        velocity_profile = np.sum(reception.cube.data, axis=(0, 2))

        self.assertAlmostEqual(self.target_range / self.waveform.range_resolution(self.device.bandwidth), np.argmax(range_profile), -1)
        self.assertEqual(expected_velocity_peak, np.argmax(velocity_profile))

    def test_doppler(self) -> None:
        """Test doppler shift generation"""

        velocity_candidates = [1.5*self.radar.velocity_resolution(self.device.carrier_frequency), 0, -1.5*self.radar.velocity_resolution(self.device.carrier_frequency)]
        expected_bin_indices = [4, 5, 6]

        for expected_bin_index, target_velocity in zip(expected_bin_indices, velocity_candidates):
            self.virtual_target.trajectory = StaticTrajectory(self.virtual_target.trajectory.pose, np.array([0, 0, target_velocity], dtype=np.float64))

            propagation = self.channel.propagate(self.device.transmit(), self.device, self.device)
            reception = self.device.receive(propagation).operator_receptions[0]

            velocity_bins = np.sum(reception.cube.data, axis=(0, 2))
            bin_index = np.argmax(velocity_bins)

            self.assertEqual(expected_bin_index, bin_index)
