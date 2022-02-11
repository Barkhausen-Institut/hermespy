# -*- coding: utf-8 -*-
"""
FMCW radar evaluation with radar channel simulation.
"""

from unittest import TestCase

import numpy as np

from hermespy.channel import RadarChannel
from hermespy.radar.radar import Radar
from hermespy.radar.fmcw import FMCW
from hermespy.simulation import Simulation

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.3"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class FMCWRadarSimulation(TestCase):

    def setUp(self) -> None:

        self.simulation = Simulation()
        self.device = self.simulation.new_device()
        self.device.carrier_frequency = 1e8

        self.waveform = FMCW()

        self.radar = Radar()
        self.radar.waveform = self.waveform

        self.radar.device = self.device
        self.device.sampling_rate = self.radar.sampling_rate

        self.channel = RadarChannel(target_range=.5*self.waveform.max_range,
                                    radar_cross_section=1.)
        self.simulation.set_channel(self.device, self.device, self.channel)

    def test_detection(self) -> None:

        signal, = self.radar.transmit()
        rx_signals, _, csi = self.channel.propagate(signal)
        self.device.receive_signal(rx_signals[0], csi)
        cube, = self.radar.receive()

        expected_velocity_peak = 0
        expected_range_peak = int(self.channel.target_range / self.waveform.range_resolution)

        range_profile = np.sum(cube.data, axis=(0, 1))
        velocity_profile = np.sum(cube.data, axis=(0, 2))

        self.assertEqual(expected_range_peak, np.argmax(range_profile))
        self.assertEqual(expected_velocity_peak, np.argmax(velocity_profile))
