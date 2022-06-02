# -*- coding: utf-8 -*-
"""Test communication evaluators."""

from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from numpy.random import default_rng

from hermespy.modem.evaluators import ThroughputEvaluator

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "3.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestThroughputEvaluator(TestCase):
    """Test throughput evaluation."""

    def setUp(self) -> None:

        self.rng = default_rng(42)
        self.num_frames = 10
        self.bits_per_frame = 100
        self.frame_duration = 1e-3

        self.transmitter = Mock()
        self.transmitter.num_data_bits_per_frame = self.bits_per_frame
        self.transmitter.frame_duration = self.frame_duration

        self.receiver = Mock()
        self.receiver.num_data_bits_per_frame = self.bits_per_frame
        self.receiver.frame_duration = self.frame_duration

        self.evaluator = ThroughputEvaluator(self.transmitter, self.receiver)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes."""

        self.assertIs(self.transmitter, self.evaluator.transmitting_modem)
        self.assertIs(self.receiver, self.evaluator.receiving_modem)

    def test_evaluate(self) -> None:
        """Evaluator should compute the proper throughput rate."""

        self.transmitter.transmitted_bits = self.rng.integers(0, 2, self.num_frames * self.bits_per_frame)
        self.receiver.received_bits = self.transmitter.transmitted_bits.copy()

        # Assert throughput without any frame errors
        expected_throughput = self.bits_per_frame / self.frame_duration
        throughput = self.evaluator.evaluate(Mock())
        self.assertAlmostEqual(expected_throughput, throughput.to_scalar())

        # Assert throughput with frame errors
        self.receiver.received_bits[0:int(.5*self.bits_per_frame)] = 1.
        expected_throughput = (self.num_frames - 1) * self.bits_per_frame / (self.num_frames * self.frame_duration)
        throughput = self.evaluator.evaluate(Mock())
        self.assertEqual(expected_throughput, throughput.to_scalar())
