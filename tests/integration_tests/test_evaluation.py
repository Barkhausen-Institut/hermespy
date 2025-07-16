# -*- coding: utf-8 -*-
from unittest import TestCase

import numpy as np

from hermespy.modem import DuplexModem, RootRaisedCosineWaveform, BitErrorEvaluator, BlockErrorEvaluator, FrameErrorEvaluator, ThroughputEvaluator
from hermespy.simulation import SimulatedDevice
from hermespy.core.evaluators import ReceivePowerEvaluator
from hermespy.core.pymonte import Evaluator, GridDimension
from unit_tests.utils import SimulationTestContext

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class InvestigatedObject(object):
    def __init__(self) -> None:
        self.dim = 0

    @property
    def dimension(self) -> int:
        return self.dim

    @dimension.setter
    def dimension(self, value: int) -> None:
        self.dim = value


class TestEvaluators(TestCase):
    def setUp(self) -> None:
        waveform = RootRaisedCosineWaveform(symbol_rate=1, num_preamble_symbols=0, num_data_symbols=100, modulation_order=64, oversampling_factor=1)
        self.device = SimulatedDevice()

        self.modem = DuplexModem()
        self.modem.waveform = waveform
        self.device.transmitters.add(self.modem)
        self.device.receivers.add(self.modem)

        investigated_object = InvestigatedObject()
        self.dimension = GridDimension(investigated_object, "dimension", [0], "title")

    def _test_evaluator(self, evaluator: Evaluator) -> None:
        """Generate a result from a given evaluator and test its plotting routine."""

        result = evaluator.initialize_result([self.dimension])

        transmission = self.device.transmit()
        self.device.receive(transmission)

        try:
            evaluation = evaluator.evaluate()
            result.add_artifact((0,), evaluation.artifact(), False)

            with SimulationTestContext():
                _ = result.visualize()

        except BaseException as e:
            self.fail(msg=str(e))

    def test_bit_error_evaluator(self) -> None:
        """Test the bit error communication evaluation"""

        ber = BitErrorEvaluator(self.modem, self.modem)
        self._test_evaluator(ber)

    def test_block_error_evaluator(self) -> None:
        """Test the block error communication evaluation"""

        ber = BlockErrorEvaluator(self.modem, self.modem)
        self._test_evaluator(ber)

    def test_frame_error_evaluator(self) -> None:
        """Test the frame error communication evaluation"""

        ber = FrameErrorEvaluator(self.modem, self.modem)
        self._test_evaluator(ber)

    def test_throughput_evaluator(self) -> None:
        """Test the throughput communication evaluation"""

        ber = ThroughputEvaluator(self.modem, self.modem)
        self._test_evaluator(ber)

    def test_received_power_evaluator(self) -> None:
        """Test the received power evaluation"""

        pow = ReceivePowerEvaluator(self.modem)
        self._test_evaluator(pow)
