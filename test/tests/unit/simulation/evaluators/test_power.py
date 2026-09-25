# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np

from hermespy.simulation import (
    ConstantDCPowerModel,
    DCPowerConsumptionEvaluator,
    PowerAmplifier,
    PowerConsumptionEvaluator,
    RFSignal,
    RampGenerator,
    WaldenADCPowerModel,
)
from hermespy.simulation.rf.blocks.shift import Shift
from ...core.test_factory import test_roundtrip_serialization

__author__ = "Emre Can Ataş"
__copyright__ = "Copyright 2026, Barkhausen Institut gGmbH"
__credits__ = ["Emre Can Ataş"]
__license__ = "AGPLv3"
__version__ = "1.6.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestPowerConsumptionEvaluator(TestCase):
    """Test the radio-frequency block power consumption evaluator"""

    def setUp(self) -> None:
        self.bandwidth = 1e6
        self.oversampling_factor = 1
        self.num_samples = 100

        self.amplifier = PowerAmplifier(gain=2.0, dc_power_model=ConstantDCPowerModel(2.5))
        self.evaluator = PowerConsumptionEvaluator(self.amplifier)

        self.signal = RFSignal.FromNDArray(
            np.ones((1, self.num_samples), dtype=np.complex128), self.bandwidth
        )

    def __propagate(self, block) -> None:
        """Propagate the test signal through a block"""

        realization = block.realize(self.bandwidth, self.oversampling_factor, 0.0)
        block.propagate(realization, self.signal, False)

    def test_properties(self) -> None:
        """Test static properties of the evaluator"""

        self.assertEqual("Eff", self.evaluator.abbreviation)
        self.assertEqual("Block Efficiency", self.evaluator.title)
        self.assertIs(self.amplifier, self.evaluator.block)

    def test_evaluate_validation(self) -> None:
        """Evaluating before any propagation should raise a RuntimeError"""

        with self.assertRaises(RuntimeError):
            self.evaluator.evaluate()

    def test_evaluate_active_block(self) -> None:
        """Active blocks should report the power drawn by their consumption model"""

        self.__propagate(self.amplifier)
        evaluation = self.evaluator.evaluate()

        expected_shape = (1, self.num_samples)
        self.assertSequenceEqual(expected_shape, evaluation.input_power.shape)
        self.assertSequenceEqual(expected_shape, evaluation.output_power.shape)
        self.assertSequenceEqual(expected_shape, evaluation.dc_power.shape)

        np.testing.assert_allclose(1.0, evaluation.input_power)
        np.testing.assert_allclose(4.0, evaluation.output_power)
        np.testing.assert_allclose(2.5, evaluation.dc_power)

    def test_evaluate_passive_block(self) -> None:
        """Passive blocks should not report any direct current power consumption"""

        shift = Shift(0.25)
        evaluator = PowerConsumptionEvaluator(shift)
        self.__propagate(shift)

        evaluation = evaluator.evaluate()
        np.testing.assert_array_equal(0.0, evaluation.dc_power)

    def test_artifact(self) -> None:
        """The generated artifact should carry the evaluated efficiency"""

        self.__propagate(self.amplifier)
        evaluation = self.evaluator.evaluate()

        self.assertEqual(evaluation.efficiency, evaluation.artifact().artifact)

    def test_visualization(self) -> None:
        """Visualizing an evaluation should generate a plot"""

        self.__propagate(self.amplifier)
        evaluation = self.evaluator.evaluate()

        visualization = evaluation.visualize()
        self.assertIsNotNone(visualization.figure)

    def test_serialization(self) -> None:
        """Test evaluator serialization"""

        test_roundtrip_serialization(self, self.evaluator)

    def test_evaluate_source_block(self) -> None:
        """Blocks without input ports should report a finite efficiency"""

        generator = RampGenerator(3, 100e6, 1e13, 50e-6, dc_power_model=ConstantDCPowerModel(2.5))
        evaluator = PowerConsumptionEvaluator(generator)

        realization = generator.realize(100e6, 1, 0.0)
        empty = RFSignal.FromNDArray(np.zeros((0, 1), dtype=np.complex128), 100e6)
        generator.propagate(realization, empty, False)

        evaluation = evaluator.evaluate()

        self.assertSequenceEqual(evaluation.output_power.shape, evaluation.input_power.shape)
        self.assertSequenceEqual(evaluation.output_power.shape, evaluation.dc_power.shape)
        self.assertFalse(np.isnan(evaluation.efficiency))


class TestDCPowerConsumptionEvaluator(TestCase):
    """Test the radio-frequency block supply power evaluator"""

    def setUp(self) -> None:
        self.bandwidth = 1e6
        self.oversampling_factor = 1
        self.num_samples = 100

        self.amplifier = PowerAmplifier(gain=2.0, dc_power_model=ConstantDCPowerModel(2.5))
        self.evaluator = DCPowerConsumptionEvaluator(self.amplifier)

        self.signal = RFSignal.FromNDArray(
            np.ones((1, self.num_samples), dtype=np.complex128), self.bandwidth
        )

    def __propagate(self, block) -> None:
        """Propagate the test signal through a block"""

        realization = block.realize(self.bandwidth, self.oversampling_factor, 0.0)
        block.propagate(realization, self.signal, False)

    def test_properties(self) -> None:
        """Test static properties of the evaluator"""

        self.assertEqual("P_DC", self.evaluator.abbreviation)
        self.assertEqual("Block Supply Power", self.evaluator.title)
        self.assertIs(self.amplifier, self.evaluator.block)

    def test_evaluate(self) -> None:
        """The evaluation should report the power drawn by the consumption model"""

        self.__propagate(self.amplifier)
        evaluation = self.evaluator.evaluate()

        self.assertAlmostEqual(2.5, evaluation.mean_dc_power)

    def test_artifact(self) -> None:
        """The generated artifact should carry the drawn power rather than the efficiency"""

        self.__propagate(self.amplifier)
        evaluation = self.evaluator.evaluate()

        self.assertEqual(evaluation.mean_dc_power, evaluation.artifact().artifact)
        self.assertNotAlmostEqual(evaluation.efficiency, evaluation.artifact().artifact)

    def test_evaluate_converter_model(self) -> None:
        """The evaluation should report the power predicted by a converter model"""

        model = WaldenADCPowerModel(8, 100e6)
        amplifier = PowerAmplifier(gain=1.0, dc_power_model=model)
        evaluator = DCPowerConsumptionEvaluator(amplifier)
        self.__propagate(amplifier)

        self.assertAlmostEqual(model.power, evaluator.evaluate().mean_dc_power)

    def test_visualization(self) -> None:
        """Visualizing an evaluation should generate a plot"""

        self.__propagate(self.amplifier)
        evaluation = self.evaluator.evaluate()

        visualization = evaluation.visualize()
        self.assertIsNotNone(visualization.figure)

    def test_serialization(self) -> None:
        """Test evaluator serialization"""

        test_roundtrip_serialization(self, self.evaluator)
