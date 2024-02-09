# -*- coding: utf-8 -*-
from unittest import TestCase

import numpy as np

from hermespy.modem import DuplexModem, RootRaisedCosineWaveform, BitErrorEvaluator, BlockErrorEvaluator, FrameErrorEvaluator, ThroughputEvaluator
from hermespy.simulation import SimulatedDevice
from hermespy.core.evaluators import ReceivedPowerEvaluator
from hermespy.core.monte_carlo import Evaluator, GridDimension
from unit_tests.utils import SimulationTestContext

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
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
        self.modem.device = self.device

        investigated_object = InvestigatedObject()
        self.dimension = GridDimension(investigated_object, "dimension", [0], "title")

    def _test_evaluator(self, evaluator: Evaluator) -> None:
        """Generate a result from a given evaluator and test its plotting routine."""

        transmission = self.modem.transmit()
        self.device.process_input(transmission.signal)
        _ = self.modem.receive()

        try:
            evaluation = evaluator.evaluate()

            artifact = evaluation.artifact()
            artifact_grid = np.empty(1, dtype=object)
            artifact_grid[0] = [artifact, artifact]

            result = evaluator.generate_result([self.dimension], artifact_grid)

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

        pow = ReceivedPowerEvaluator(self.modem)
        self._test_evaluator(pow)
