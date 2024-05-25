# -*- coding: utf-8 -*-
"""
FMCW radar evaluation with radar channel simulation.
"""

from __future__ import annotations
from unittest import TestCase

import numpy as np

from hermespy.beamforming import ConventionalBeamformer
from hermespy.core import Direction, Transformation
from hermespy.channel import MultiTargetRadarChannel, VirtualRadarTarget, FixedCrossSection
from hermespy.radar.radar import Radar
from hermespy.radar.fmcw import FMCW
from hermespy.simulation import SimulationScenario, SimulatedIdealAntenna, SimulatedUniformArray, StaticTrajectory
from scipy.constants import speed_of_light

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class FMCWRadarSimulation(TestCase):
    def setUp(self) -> None:
        self.scenario = SimulationScenario()
        self.device = self.scenario.new_device()
        self.device.carrier_frequency = 1e8
        self.device.antennas = SimulatedUniformArray(SimulatedIdealAntenna, 0.5 * speed_of_light / self.device.carrier_frequency, (5, 5))

        self.waveform = FMCW()
        self.beamformer = ConventionalBeamformer()

        self.radar = Radar()
        self.radar.waveform = self.waveform
        self.radar.transmit_beamformer = self.beamformer
        self.radar.receive_beamformer = self.beamformer

        self.device.transmitters.add(self.radar)
        self.device.receivers.add(self.radar)

        self.target_range = 0.5 * self.waveform.max_range
        self.channel = MultiTargetRadarChannel(attenuate=False)

        self.virtual_target = VirtualRadarTarget(FixedCrossSection(1.0), trajectory=StaticTrajectory(Transformation.From_Translation(np.array([0, 0, self.target_range]))))
        self.channel.add_target(self.virtual_target)

        self.scenario.set_channel(self.device, self.device, self.channel)

    def test_beamforming(self) -> None:
        """The radar channel target located should be estimated correctly by the beamformer"""

        self.radar.receive_beamformer.probe_focus_points = 0.25 * np.pi * np.array([[[0.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]], [[2.0, 1.0]], [[3.0, 1.0]], [[4.0, 1.0]], [[5.0, 1.0]], [[6.0, 1.0]], [[7.0, 1.0]]])

        for angle_index, (azimuth, zenith) in enumerate(self.radar.receive_beamformer.probe_focus_points[:, 0, :]):
            # Configure the channel's only target
            self.virtual_target.trajectory = StaticTrajectory(Transformation.From_Translation(Direction.From_Spherical(azimuth, zenith) * self.target_range))

            # Generate the radar cube
            propagation = self.channel.propagate(self.device.transmit())
            self.device.process_input(propagation)
            reception = self.radar.receive()

            directive_powers = np.linalg.norm(reception.cube.data, axis=(1, 2))
            self.assertEqual(angle_index, directive_powers.argmax())

    def test_detection(self) -> None:
        """Test FMCW detection"""

        propagation = self.channel.propagate(self.device.transmit())
        self.device.process_input(propagation)
        reception = self.radar.receive()

        expected_velocity_peak = 5

        range_profile = np.sum(reception.cube.data, axis=(0, 1))
        velocity_profile = np.sum(reception.cube.data, axis=(0, 2))

        self.assertAlmostEqual(self.target_range / self.waveform.range_resolution, np.argmax(range_profile), -1)
        self.assertEqual(expected_velocity_peak, np.argmax(velocity_profile))

    def test_doppler(self) -> None:
        """Test doppler shift generation"""

        velocity_candidates = [-1.5*self.radar.velocity_resolution, 0, 1.5*self.radar.velocity_resolution]
        expected_bin_indices = [4, 5, 6]

        for expected_bin_index, target_velocity in zip(expected_bin_indices, velocity_candidates):
            self.virtual_target.trajectory = StaticTrajectory(self.virtual_target.trajectory.pose, np.array([0, 0, target_velocity], dtype=np.float_))

            propagation = self.channel.propagate(self.device.transmit())
            self.device.process_input(propagation)
            reception = self.radar.receive()

            velocity_bins = np.sum(reception.cube.data, axis=(0, 2))
            bin_index = np.argmax(velocity_bins)

            self.assertEqual(expected_bin_index, bin_index)
