# -*- coding: utf-8 -*-

from hermespy.channel.sionna_rt_channel import SionnaRTChannel
from sionna import rt
from hermespy.radar import Radar, FMCW, CFARDetector
from hermespy.simulation import Simulation, SimulatedUniformArray, SimulatedIdealAntenna, StaticTrajectory
from hermespy.beamforming import ConventionalBeamformer
from hermespy.core import Transformation

import unittest
import numpy as np
from scipy.constants import speed_of_light


class TestRadarSionna(unittest.TestCase):
    def test_detection_over_sionnaRT(self):
        # Initialize a single device
        simulation = Simulation()
        carrier_frequency = 24e9
        device = simulation.new_device(carrier_frequency=carrier_frequency)
        device.antennas = SimulatedUniformArray(SimulatedIdealAntenna, .5 * speed_of_light / carrier_frequency, [4, 4, 1])

        # Create a radar channel modeling a single target
        channel = SionnaRTChannel(rt.load_scene(rt.scene.simple_reflector), 1.0, 42)
        simulation.set_channel(device, device, channel)

        # Configure an FMCW radar illuminating the target
        radar = Radar(device)
        radar.waveform = FMCW()
        radar.receive_beamformer = ConventionalBeamformer()
        radar.detector = CFARDetector((2, 2), (0, 0), 0.001)

        positions = [
            np.asarray([0., 0., -100.]),
        ]
        for position in positions:
            device.trajectory = StaticTrajectory(Transformation.From_Translation(position))
            simulation.scenario.drop()
            self.assertEqual(1, radar.reception.cloud.num_points)
            for coord_dist_act, coord_dist_exp in zip(np.abs(position), np.abs(radar.reception.cloud.points[0].position)):
                self.assertAlmostEqual(coord_dist_act, coord_dist_exp, delta=radar.waveform.range_resolution)
