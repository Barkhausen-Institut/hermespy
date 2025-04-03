# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock, patch, PropertyMock

import numpy as np
from numpy.testing import assert_almost_equal

from hermespy.core import Signal, SignalTransmitter, SignalReceiver
from hermespy.core.evaluators import PowerArtifact, PowerEvaluation, ReceivePowerEvaluator, TransmitPowerEvaluator, PowerResult
from hermespy.simulation import SimulatedDevice, SimulatedIdealAntenna, SimulatedUniformArray

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestPowerArtifact(TestCase):
    """Test the received power artifact"""

    def setUp(self) -> None:
        self.power = 1.234
        self.artifact = PowerArtifact(self.power)

    def test_power(self) -> None:
        """Power property should be properly initialized"""

        self.assertEqual(self.artifact.power, self.power)

    def test_string_conversion(self) -> None:
        """String conversion should return power"""

        self.assertIsInstance(str(self.artifact), str)

    def test_to_scalar(self) -> None:
        """To scalar conversion should return power"""

        self.assertEqual(self.artifact.to_scalar(), self.power)


class TestReceivePowerEvaluation(TestCase):
    """Test the received power evaluation"""

    def setUp(self) -> None:
        self.powers = np.array([1, 2, 4, 5])
        self.evaluation = PowerEvaluation(self.powers)

    def test_artifact(self) -> None:
        """Artifact function should return a correct artifact"""

        artifact = self.evaluation.artifact()
        self.assertIsInstance(artifact, PowerArtifact)
        assert_almost_equal(artifact.power, self.powers)

    def test_title(self) -> None:
        """Title property should return a correct title"""

        self.assertEqual(self.evaluation.title, "Received Power")

    def test_plot(self) -> None:
        """Plot function should return a correct plot"""

        axes = Mock()
        axes_array = np.empty((1, 1), dtype=np.object_)
        axes_array[0, 0] = axes
        self.evaluation.visualize(axes_array)
        axes.stem.assert_called_once()


class TestPowerResult(TestCase):
    """Test the received power result"""

    def setUp(self) -> None:
        self.average_powers = np.array([[1, 2, 4, 5]])
        self.receiver = SignalReceiver(10, 1e7)
        self.grid = Mock()
        self.evaluator = ReceivePowerEvaluator(Mock())

        self.result = PowerResult(self.average_powers, self.grid, self.evaluator)

    def test_average_powers(self) -> None:
        """Average powers property should be properly initialized"""

        assert_almost_equal(self.result.average_powers, self.average_powers)

    def test_to_array(self) -> None:
        """To array conversion should return average powers"""

        assert_almost_equal(self.result.to_array(), self.average_powers)


class TestReceivePowerEvaluator(TestCase):
    """Test received power evaluator"""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.num_samples = 100
        self.sampling_rate = 1e6
        self.num_antennas = 2

        self.transmitted_signal = Signal.Create(self.rng.standard_normal((self.num_antennas, self.num_samples)) + 1j * self.rng.standard_normal((self.num_antennas, self.num_samples)), self.sampling_rate, 0, 0, 0)
        
        self.transmitter = SignalTransmitter(self.transmitted_signal)
        self.receiver = SignalReceiver(self.num_samples, self.sampling_rate)
        self.evaluator = ReceivePowerEvaluator(self.receiver)
        
        self.device = SimulatedDevice(antennas=SimulatedUniformArray(SimulatedIdealAntenna, 1.0, [self.num_antennas, 1, 1]))
        self.device.transmitters.add(self.transmitter)
        self.device.receivers.add(self.receiver)

    def test_propeties(self) -> None:
        """Properties should be properly initialized"""

        self.assertEqual("RxPwr", self.evaluator.abbreviation)
        self.assertEqual("Receive Power", self.evaluator.title)

    def test_evaluate_validation(self) -> None:
        """Evaluate should raise a RuntimeError if no reception is available"""

        with self.assertRaises(RuntimeError):
            self.evaluator.evaluate()

    def test_evaluation(self) -> None:
        num_drops = 10
        signal_scales = self.rng.random(num_drops)
        expected_powers = self.transmitted_signal.power * np.sum(signal_scales**2) / num_drops

        # Collect drop artifacts
        grid = []
        artifacts = np.empty(1, dtype=np.object_)
        artifacts[0] = list()

        for signal_scale in signal_scales:
            signal = self.transmitted_signal.copy()
            for block in signal:
                block *= signal_scale

            _ = self.device.receive(signal)
            artifacts[0].append(self.evaluator.evaluate().artifact())

        # Generate result
        result = self.evaluator.generate_result(grid, artifacts)
        assert_almost_equal(result.average_powers[0, :], expected_powers)


class TestTransmitPowerEvaluator(TestCase):
    """Test transmit power evaluator"""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.num_samples = 100
        self.sampling_rate = 1e6
        self.num_antennas = 2

        self.transmitted_signal = Signal.Create(self.rng.standard_normal((self.num_antennas, self.num_samples)) + 1j * self.rng.standard_normal((self.num_antennas, self.num_samples)), self.sampling_rate, 0, 0, 0)

        self.device = SimulatedDevice(antennas=SimulatedUniformArray(SimulatedIdealAntenna, 1.0, [self.num_antennas, 1, 1]))
        
        self.transmitter = SignalTransmitter(self.transmitted_signal)
        self.device.transmitters.add(self.transmitter)

        self.evaluator = TransmitPowerEvaluator(self.transmitter)

    def test_propeties(self) -> None:
        """Properties should be properly initialized"""

        self.assertEqual("TxPwr", self.evaluator.abbreviation)
        self.assertEqual("Transmit Power", self.evaluator.title)

    def test_evaluate_validation(self) -> None:
        """Evaluate should raise a RuntimeError if no reception is available"""

        with self.assertRaises(RuntimeError):
            self.evaluator.evaluate()

    def test_evaluation(self) -> None:
        num_drops = 10
        signal_scales = self.rng.random(num_drops)
        expected_powers = self.transmitted_signal.power * np.sum(signal_scales**2) / num_drops

        # Collect drop artifacts
        grid = []
        artifacts = np.empty(1, dtype=np.object_)
        artifacts[0] = list()

        for signal_scale in signal_scales:
            signal = self.transmitted_signal.copy()
            for block in signal:
                block *= signal_scale

            _ = self.transmitter.signal = signal
            self.device.transmit()
            artifacts[0].append(self.evaluator.evaluate().artifact())

        # Generate result
        result = self.evaluator.generate_result(grid, artifacts)
        assert_almost_equal(result.average_powers[0, :], expected_powers)
