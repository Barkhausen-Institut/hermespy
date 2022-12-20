# -*- coding: utf-8 -*-
"""Test communication evaluators."""

from unittest import TestCase
from unittest.mock import PropertyMock, patch, Mock

from numpy.random import default_rng

from hermespy.channel import RadarChannel
from hermespy.radar import DetectionProbEvaluator, FMCW, Radar, ReceiverOperatingCharacteristic
from hermespy.simulation import SimulationScenario
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestDetectionProbEvaluator(TestCase):
    """Test detection probability evaluation"""

    def setUp(self) -> None:

        self.rng = default_rng(42)
        self.num_frames = 10

        self.threshold = 2.

        self.radar = Mock()
        self.radar.reception = PropertyMock()

        self.evaluator = DetectionProbEvaluator(self.radar)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""

        self.assertIs(self.radar, self.evaluator.receiving_radar)

    def test_evaluate(self) -> None:
        """Evaluator should compute the proper detection evaluation"""
        
        self.radar.reception.cloud.num_points = 0
        evaluation = self.evaluator.evaluate()
        self.assertEqual(0., evaluation.artifact().to_scalar())
        
        self.radar.reception.cloud.num_points = 1
        evaluation = self.evaluator.evaluate()
        self.assertEqual(1., evaluation.artifact().to_scalar())


class TestReceiverOperatingCharacteristic(TestCase):
    
    def setUp(self) -> None:
        
        self.scenario = SimulationScenario()
        self.device = self.scenario.new_device()
        self.channel = RadarChannel(1., 1.)
        self.scenario.set_channel(self.device, self.device, self.channel)
        
        self.radar = Radar()
        self.radar.waveform = FMCW()
        self.radar.device = self.device
        
        self.evaluator = ReceiverOperatingCharacteristic(self.radar, self.channel)
        
    def test_evaluate(self) -> None:
        """Test evaluation"""
        
        _ = self.radar.transmit()
        forwards_propagation, _, _ = self.channel.propagate(self.device.transmit())
        self.device.receive(forwards_propagation)
        _ = self.radar.receive()
        
        evaluation = self.evaluator.evaluate()
