# -*- coding: utf-8 -*-
"""
FMCW radar evaluation with radar channel simulation.
"""

from unittest import TestCase

import numpy as np

from hermespy.beamforming import ConventionalBeamformer
from hermespy.core import UniformArray, IdealAntenna
from hermespy.channel import RadarChannel
from hermespy.radar.radar import Radar
from hermespy.radar.fmcw import FMCW
from hermespy.simulation import Simulation
from scipy.constants import speed_of_light

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class FMCWRadarSimulation(TestCase):

    def setUp(self) -> None:

        self.simulation = Simulation()
        self.device = self.simulation.scenario.new_device()
        self.device.carrier_frequency = 1e8
        self.device.antennas = UniformArray(IdealAntenna(), .5 * speed_of_light / self.device.carrier_frequency, (3, 3))

        self.waveform = FMCW()
        self.beamformer = ConventionalBeamformer()

        self.radar = Radar()
        self.radar.waveform = self.waveform
        self.radar.transmit_beamformer = self.beamformer
        self.radar.receive_beamformer = self.beamformer
        
        self.device.transmitters.add(self.radar)
        self.device.receivers.add(self.radar)

        self.target_range = int(.5 * self.waveform.max_range) * self.waveform.range_resolution
        self.channel = RadarChannel(target_range=self.target_range,
                                    radar_cross_section=1.)
        self.simulation.scenario.set_channel(self.device, self.device, self.channel)

    def test_beamforming(self) -> None:
        """The radar channel target located should be estimated correctly by the beamformer"""

        self.radar.receive_beamformer.probe_focus_points = .25 * np.pi * np.array([[[0., 0.]],
                                                                                   [[0., 1.]],
                                                                                   [[1., 1.]],
                                                                                   [[2., 1.]],
                                                                                   [[3., 1.]],
                                                                                   [[4., 1.]],
                                                                                   [[5., 1.]],
                                                                                   [[6., 1.]],
                                                                                   [[7., 1.]]])

        for angle_index, (azimuth, zenith) in enumerate(self.radar.receive_beamformer.probe_focus_points[:, 0, :]):

            # Configure the channel
            self.channel.target_range = self.target_range
            self.channel.target_azimuth = azimuth
            self.channel.target_zenith = zenith

            # Generate the radar cube
            self.radar.transmit()
            tx_signals = self.device.transmit()
            rx_signals, _, _ = self.channel.propagate(tx_signals)
            self.device.process_input(rx_signals)
            reception = self.radar.receive()

            directive_powers = np.linalg.norm(reception.cube.data, axis=(1, 2))
            self.assertEqual(angle_index, directive_powers.argmax())

    def test_detection(self) -> None:

        _ = self.radar.transmit()
        rx_signals, _, _ = self.channel.propagate(self.device.transmit())
        self.device.process_input(rx_signals)
        reception = self.radar.receive()

        expected_velocity_peak = 5

        range_profile = np.sum(reception.cube.data, axis=(0, 1))
        velocity_profile = np.sum(reception.cube.data, axis=(0, 2))

        self.assertAlmostEqual(self.target_range / self.waveform.range_resolution, np.argmax(range_profile), -1)
        self.assertEqual(expected_velocity_peak, np.argmax(velocity_profile))
