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

__author__ = "Egor Achkasov"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Egor Achkasov", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestRadarSionna(unittest.TestCase):
    def test_detection_over_sionnaRT(self):
        # Initialize a single device
        simulation = Simulation()
        carrier_frequency = 24e9
        device = simulation.new_device(carrier_frequency=carrier_frequency)
        device.antennas = SimulatedUniformArray(SimulatedIdealAntenna, .5 * speed_of_light / carrier_frequency, [4, 4, 1])
        device.transmit_coding[0] = ConventionalBeamformer()

        # Create a radar channel modeling a single target
        channel = SionnaRTChannel(rt.scene.simple_reflector, 1.0, 42)
        simulation.set_channel(device, device, channel)

        # Configure an FMCW radar illuminating the target
        radar = Radar(device)
        radar.waveform = FMCW()
        radar.receive_beamformer = ConventionalBeamformer()
        radar.detector = CFARDetector((2, 2), (0, 0), 0.001)
        device.add_dsp(radar)

        positions = [
            np.asarray([0., 0., -100.]),
        ]
        for position in positions:
            device.trajectory = StaticTrajectory(Transformation.From_Translation(position))
            drop = simulation.scenario.drop()
            radar_reception = drop.device_receptions[0].operator_receptions[0]
            self.assertEqual(1, radar_reception.cloud.num_points)
            for coord_dist_act, coord_dist_exp in zip(np.abs(position), np.abs(radar_reception.cloud.points[0].position)):
                self.assertAlmostEqual(coord_dist_act, coord_dist_exp, delta=radar.waveform.range_resolution)
