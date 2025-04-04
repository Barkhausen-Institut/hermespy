# -*- coding: utf-8 -*-
"""Test communication evaluators"""

from unittest import TestCase
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

from hermespy.core.monte_carlo import ScalarEvaluationResult, ArtifactTemplate
from hermespy.modem import TransmittingModem, ReceivingModem, RootRaisedCosineWaveform
from hermespy.modem.evaluators import BitErrorEvaluation, BitErrorEvaluator, BlockErrorEvaluation, BlockErrorEvaluator, CommunicationEvaluator, FrameErrorEvaluation, FrameErrorEvaluator, ThroughputEvaluation, ThroughputEvaluator, ConstellationEVM, EVMEvaluation
from hermespy.simulation import SimulatedDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class CommunicationEvaluatorMock(CommunicationEvaluator):
    """Mock communication evaluator for testing"""

    @property
    def title(self) -> str:
        return "Mock Communication Evaluator"

    @property
    def abbreviation(self) -> str:
        return "MCE"

    def evaluate(self) -> np.ndarray:
        return np.array([1.0])


class TestCommunicationEvaluator(TestCase):
    """Test communication evaluator base class"""

    def setUp(self) -> None:
        self.transmitter = Mock()
        self.receiver = Mock()

        self.evaluator = CommunicationEvaluatorMock(self.transmitter, self.receiver)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored"""

        self.assertIs(self.transmitter, self.evaluator.transmitting_modem)
        self.assertIs(self.receiver, self.evaluator.receiving_modem)

    def test_generate_result(self) -> None:
        """Result should be properly generated"""

        artifacts = np.empty(1, dtype=np.object_)
        artifacts[0] = [ArtifactTemplate(n) for n in range(10)]

        result = self.evaluator.generate_result([], artifacts)
        self.assertIsInstance(result, ScalarEvaluationResult)


class TestBitErrorEvaluation(TestCase):
    """Test bit error evaluation"""

    def setUp(self) -> None:
        data = np.random.randint(0, 2, 10)
        self.evaluation = BitErrorEvaluation(data)

    def test_title(self) -> None:
        """Title should be properly generated"""

        self.assertEqual("Bit Error Evaluation", self.evaluation.title)

    def test_plot(self) -> None:
        """Plotting should generate a valid plot"""

        figure_mock = Mock(spec=plt.Figure)
        axes_mock = Mock(spec=plt.Axes)
        axes_collection = np.array([[axes_mock]], dtype=np.object_)

        with patch("matplotlib.pyplot.subplots") as subplots_mock:
            subplots_mock.return_value = (figure_mock, axes_collection)

            self.evaluation.visualize()
            subplots_mock.assert_called_once()


class TestBitErrorEvaluator(TestCase):
    """Test bit error evaluator"""

    def setUp(self) -> None:
        self.waveform = RootRaisedCosineWaveform(symbol_rate=1e9, num_preamble_symbols=0, num_data_symbols=10)
        self.transmitter = TransmittingModem()
        self.transmitter.waveform = self.waveform
        self.receiver = ReceivingModem()
        self.receiver.waveform = self.waveform

        self.transmit_device = SimulatedDevice()
        self.transmit_device.transmitters.add(self.transmitter)
        self.receive_device = SimulatedDevice()
        self.transmit_device.receivers.add(self.receiver)

        self.evaluator = BitErrorEvaluator(self.transmitter, self.receiver)

    def test_evaluate(self) -> None:
        """Evaluator should compute the proper bit error rate"""

        transmission = self.transmitter.transmit(self.transmit_device.state())
        self.receiver.receive(transmission.signal, self.receive_device.state())

        evaluation = self.evaluator.evaluate()
        self.assertEqual(0.0, evaluation.artifact().to_scalar())

    def test_abbreviation(self) -> None:
        """Abbreviation should be properly generated"""

        self.assertEqual("BER", self.evaluator.abbreviation)

    def test_title(self) -> None:
        """Title should be properly generated"""

        self.assertEqual("Bit Error Rate Evaluation", self.evaluator.title)

    def test_scalar_cdf(self) -> None:
        """CDF should be properly generated"""

        self.assertEqual(0.0, self.evaluator._scalar_cdf(0.0))
        self.assertEqual(1.0, self.evaluator._scalar_cdf(1.0))


class TestBlockErrorEvaluation(TestCase):
    """Test block error evaluation"""

    def setUp(self) -> None:
        data = np.random.randint(0, 2, 10)
        self.evaluation = BlockErrorEvaluation(data)

    def test_title(self) -> None:
        """Title should be properly generated"""

        self.assertEqual("Block Error Evaluation", self.evaluation.title)

    def test_plot(self) -> None:
        """Plotting should generate a valid plot"""

        figure_mock = Mock(spec=plt.Figure)
        axes_mock = Mock(spec=plt.Axes)
        axes_collection = np.array([[axes_mock]], dtype=np.object_)

        with patch("matplotlib.pyplot.subplots") as subplots_mock:
            subplots_mock.return_value = (figure_mock, axes_collection)

            self.evaluation.visualize()
            subplots_mock.assert_called_once()


class TestBlockErrorEvaluator(TestCase):
    """Test block error evaluator"""

    def setUp(self) -> None:
        self.waveform = RootRaisedCosineWaveform(symbol_rate=1e9, num_preamble_symbols=0, num_data_symbols=10)
        self.transmitter = TransmittingModem()
        self.transmitter.waveform = self.waveform
        self.receiver = ReceivingModem()
        self.receiver.waveform = self.waveform

        self.transmit_device = SimulatedDevice()
        self.transmit_device.transmitters.add(self.transmitter)
        self.receive_device = SimulatedDevice()
        self.transmit_device.receivers.add(self.receiver)

        self.evaluator = BlockErrorEvaluator(self.transmitter, self.receiver)

    def test_evaluate(self) -> None:
        """Evaluator should compute the proper block error rate"""

        transmission = self.transmitter.transmit(self.transmit_device.state())
        self.receiver.receive(transmission.signal, self.receive_device.state())

        evaluation = self.evaluator.evaluate()
        self.assertEqual(0.0, evaluation.artifact().to_scalar())

    def test_abbreviation(self) -> None:
        """Abbreviation should be properly generated"""

        self.assertEqual("BLER", self.evaluator.abbreviation)

    def test_title(self) -> None:
        """Title should be properly generated"""

        self.assertEqual("Block Error Rate", self.evaluator.title)

    def test_scalar_cdf(self) -> None:
        """CDF should be properly generated"""

        self.assertEqual(0.0, self.evaluator._scalar_cdf(0.0))
        self.assertEqual(1.0, self.evaluator._scalar_cdf(1.0))


class TestFrameErrorEvaluation(TestCase):
    """Test frame error evaluation"""

    def setUp(self) -> None:
        data = np.random.randint(0, 2, 10)
        self.evaluation = FrameErrorEvaluation(data)

    def test_title(self) -> None:
        """Title should be properly generated"""

        self.assertEqual("Frame Error Evaluation", self.evaluation.title)

    def test_plot(self) -> None:
        """Plotting should generate a valid plot"""

        figure_mock = Mock(spec=plt.Figure)
        axes_mock = Mock(spec=plt.Axes)
        axes_collection = np.array([[axes_mock]], dtype=np.object_)

        with patch("matplotlib.pyplot.subplots") as subplots_mock:
            subplots_mock.return_value = (figure_mock, axes_collection)

            self.evaluation.visualize()
            subplots_mock.assert_called_once()


class TestFrameErrorEvaluator(TestCase):
    """Test frame error evaluator"""

    def setUp(self) -> None:
        self.waveform = RootRaisedCosineWaveform(symbol_rate=1e9, num_preamble_symbols=0, num_data_symbols=10)
        self.transmitter = TransmittingModem()
        self.transmitter.waveform = self.waveform
        self.receiver = ReceivingModem()
        self.receiver.waveform = self.waveform

        self.transmit_device = SimulatedDevice()
        self.transmit_device.transmitters.add(self.transmitter)
        self.receive_device = SimulatedDevice()
        self.transmit_device.receivers.add(self.receiver)

        self.evaluator = FrameErrorEvaluator(self.transmitter, self.receiver)

    def test_evaluate(self) -> None:
        """Evaluator should compute the proper frame error rate"""

        transmission = self.transmitter.transmit(self.transmit_device.state(), 5 * self.waveform.frame_duration)
        self.receiver.receive(transmission.signal, self.receive_device.state())

        evaluation = self.evaluator.evaluate()
        self.assertEqual(0.0, evaluation.artifact().to_scalar())

    def test_evaluate_frame_count_mismatch(self) -> None:
        """A mismatch in number of frames should count as a frame error"""

        # Generate two frames receive only one
        frame_duration = self.waveform.frame_duration
        transmission = self.transmitter.transmit(self.transmit_device.state(), 2 * frame_duration)
        
        received_samples = transmission.signal.getitem()[:, :int(frame_duration * self.waveform.sampling_rate)]
        self.receiver.receive(transmission.signal.from_ndarray(received_samples), self.receive_device.state())

        evaluation = self.evaluator.evaluate()
        self.assertLess(0, evaluation.artifact().to_scalar())

    def test_abbreviation(self) -> None:
        """Abbreviation should be properly generated"""

        self.assertEqual("FER", self.evaluator.abbreviation)

    def test_title(self) -> None:
        """Title should be properly generated"""

        self.assertEqual("Frame Error Rate", self.evaluator.title)

    def test_scalar_cdf(self) -> None:
        """CDF should be properly generated"""

        self.assertEqual(0.0, self.evaluator._scalar_cdf(0.0))
        self.assertEqual(1.0, self.evaluator._scalar_cdf(1.0))


class TestThroughputEvaluation(TestCase):
    """Test throughput evaluation"""

    def setUp(self) -> None:
        self.bps = 100
        self.duration = 1e-3
        data = np.random.randint(0, 2, 10)
        self.evaluation = ThroughputEvaluation(self.bps, self.duration, data)

    def test_title(self) -> None:
        """Title should be properly generated"""

        self.assertEqual("Data Throughput", self.evaluation.title)

    def test_plot(self) -> None:
        """Plotting should generate a valid plot"""

        figure_mock = Mock(spec=plt.Figure)
        axes_mock = Mock(spec=plt.Axes)
        axes_collection = np.array([[axes_mock]], dtype=np.object_)

        with patch("matplotlib.pyplot.subplots") as subplots_mock:
            subplots_mock.return_value = (figure_mock, axes_collection)

            self.evaluation.visualize()
            subplots_mock.assert_called_once()


class TestThroughputEvaluator(TestCase):
    """Test throughput evaluation"""

    def setUp(self) -> None:
        self.rng = default_rng(42)
        self.num_frames = 10

        self.waveform = RootRaisedCosineWaveform(symbol_rate=1e9, num_preamble_symbols=0, num_data_symbols=100)
        self.receiver = ReceivingModem(waveform=self.waveform)
        self.transmitter = TransmittingModem(waveform=self.waveform)
        self.device = SimulatedDevice()
        self.device.transmitters.add(self.transmitter)
        self.device.receivers.add(self.receiver)

        self.evaluator = ThroughputEvaluator(self.transmitter, self.receiver)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""

        self.assertIs(self.transmitter, self.evaluator.transmitting_modem)
        self.assertIs(self.receiver, self.evaluator.receiving_modem)

    def test_evaluate(self) -> None:
        """Evaluator should compute the proper throughput rate"""

        duration = self.num_frames * self.waveform.frame_duration
        transmission = self.transmitter.transmit(self.device.state(), duration)
        self.receiver.receive(transmission.signal, self.device.state())

        # Assert throughput without any frame errors
        expected_throughput = self.waveform.bits_per_frame(None) / self.waveform.frame_duration
        throughput = self.evaluator.evaluate()
        self.assertAlmostEqual(expected_throughput, throughput.artifact().to_scalar())

    def test_title(self) -> None:
        """Title should be properly generated"""

        self.assertEqual("Data Throughput", self.evaluator.title)

    def test_abbreviation(self) -> None:
        """Abbreviation should be properly generated"""

        self.assertEqual("DRX", self.evaluator.abbreviation)


class TestEVMEvaluation(TestCase):
    """Test EVM evaluation"""

    def setUp(self) -> None:

        transmitted_symbols = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j])
        received_symbols = transmitted_symbols

        self.evaluation = EVMEvaluation(transmitted_symbols, received_symbols)

    def test_title(self) -> None:
        """Title should be properly generated"""

        self.assertEqual("Error Vector Magnitude", self.evaluation.title)

    def test_plot(self) -> None:
        """Plotting should generate a valid plot"""

        figure_mock = Mock(spec=plt.Figure)
        axes_mock = Mock(spec=plt.Axes)
        axes_collection = np.array([[axes_mock]], dtype=np.object_)

        with patch("matplotlib.pyplot.subplots") as subplots_mock:
            subplots_mock.return_value = (figure_mock, axes_collection)

            self.evaluation.visualize()
            subplots_mock.assert_called_once()

    def test_artifact(self) -> None:
        """Artifacts should be properly generated"""

        artifact = self.evaluation.artifact()
        self.assertEqual(0.0, artifact.to_scalar())


class TestConstellationEVM(TestCase):
    """Test the constellation diagram EVM evaluator"""

    def setUp(self) -> None:
        self.waveform = RootRaisedCosineWaveform(symbol_rate=1e9, num_preamble_symbols=0, num_data_symbols=10, roll_off=.9)
        self.transmitter = TransmittingModem()
        self.transmitter.seed = 42
        self.transmitter.waveform = self.waveform
        self.receiver = ReceivingModem()
        self.receiver.waveform = self.waveform

        self.transmit_device = SimulatedDevice()
        self.transmit_device.transmitters.add(self.transmitter)
        self.receive_device = SimulatedDevice()
        self.transmit_device.receivers.add(self.receiver)

        self.evaluator = ConstellationEVM(self.transmitter, self.receiver)

    def test_evaluate(self) -> None:
        """Evaluator should compute the proper frame error rate"""

        transmission = self.transmitter.transmit(self.transmit_device.state())
        self.receiver.receive(transmission.signal, self.receive_device.state())

        rolled_off_evaluation = self.evaluator.evaluate()
        self.assertAlmostEqual(0.0, rolled_off_evaluation.artifact().to_scalar(), 3)

        # Lower roll-off should result in higher EVM
        self.waveform.roll_off = 0.1
        transmission = self.transmitter.transmit(self.transmit_device.state())
        self.receiver.receive(transmission.signal, self.receive_device.state())

        evaluation = self.evaluator.evaluate()
        self.assertGreater(evaluation.artifact().to_scalar(), rolled_off_evaluation.artifact().to_scalar())

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""

        self.assertIs(self.transmitter, self.evaluator.transmitting_modem)
        self.assertIs(self.receiver, self.evaluator.receiving_modem)

    def test_title(self) -> None:
        """Title should be properly generated"""

        self.assertEqual("Error Vector Magnitude", self.evaluator.title)

    def test_abbreviation(self) -> None:
        """Abbreviation should be properly generated"""

        self.assertEqual("EVM", self.evaluator.abbreviation)
