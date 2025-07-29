# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

from hermespy.core import AntennaMode, ScalarEvaluationResult, Signal, SignalTransmitter, SignalReceiver, GridDimensionInfo, SamplePoint, ValueType
from hermespy.core.evaluators import PAPRArtifact, PowerArtifact, PowerEvaluation, ReceivePowerEvaluator, TransmitPowerEvaluator, PAPR, PAPREvaluation, SignalExtractor, SignalExtraction, ExtractedSignals
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
        result = self.evaluator.initialize_result([GridDimensionInfo([SamplePoint(0)], 'x', 'linear', ValueType.LIN)])
        
        expected_mean_power = np.sum(self.transmitted_signal.power * np.sum(signal_scales**2)) / num_drops

        # Collect drop artifacts
        grid = []
        artifacts = np.empty(1, dtype=np.object_)
        artifacts[0] = list()

        for signal_scale in signal_scales:
            signal = self.transmitted_signal.copy()
            for block in signal:
                block *= signal_scale

            _ = self.device.receive(signal)
            
            evaluation = self.evaluator.evaluate()
            result.add_artifact((0,), evaluation.artifact())
            expected_powers = self.transmitted_signal.power * signal_scale**2
            assert_almost_equal(evaluation.power, expected_powers)

        # Generate result
        assert_almost_equal(result.to_array()[0], expected_mean_power)


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
        result = self.evaluator.initialize_result([GridDimensionInfo([SamplePoint(0)], 'x', 'linear', ValueType.LIN)])
        
        expected_mean_power = np.sum(self.transmitted_signal.power * np.sum(signal_scales**2)) / num_drops

        for signal_scale in signal_scales:
            expected_power = self.transmitted_signal.power * signal_scale**2
            
            
            signal = self.transmitted_signal.copy()
            for block in signal:
                block *= signal_scale

            _ = self.transmitter.signal = signal
            self.device.transmit()
            
            evaluation = self.evaluator.evaluate()
            result.add_artifact((0,), self.evaluator.evaluate().artifact())

            assert_almost_equal(evaluation.power, expected_power)
    
        assert_almost_equal(result.to_array()[0], expected_mean_power)

class TestPAPREvaluation(TestCase):
    """Test PAPR evaluation"""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.instantaneous_powers = self.rng.rayleigh(1.0, (3, 10))
        self.evaluation = PAPREvaluation(self.instantaneous_powers)

    def test_properties(self) -> None:
        """Test static properties"""
        
        self.assertIs(self.instantaneous_powers, self.evaluation.instantaneous_power)
        self.assertEqual("Instantaneous Power", self.evaluation.title)

    def test_artifact(self) -> None:
        """Artifact function should return a correct artifact"""

        artifact = self.evaluation.artifact()
        self.assertIsInstance(artifact, PAPRArtifact)

    def test_plot(self) -> None:
        """Plot function should return a correct plot"""

        axes = Mock()
        axes.plot.return_value = [Mock()]
        axes_array = np.empty((1, 1), dtype=np.object_)
        axes_array[0, 0] = axes
        self.evaluation.visualize(axes_array)
        axes.plot.assert_called()


class TestPAPR(TestCase):
    """Test the PAPR evaluator"""
    
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.num_samples = 100
        self.sampling_rate = 1e6
        self.num_antennas = 2

        self.transmitted_signal = Signal.Create(self.rng.standard_normal((self.num_antennas, self.num_samples)) + 1j * self.rng.standard_normal((self.num_antennas, self.num_samples)), self.sampling_rate, 0, 0, 0)

        self.device = SimulatedDevice(antennas=SimulatedUniformArray(SimulatedIdealAntenna, 1.0, [self.num_antennas, 1, 1]))
        
        self.transmitter = SignalTransmitter(self.transmitted_signal)
        self.device.transmitters.add(self.transmitter)

        self.tx_papr = PAPR(self.device, AntennaMode.TX)
        self.rx_papr = PAPR(self.device, AntennaMode.RX)

        # Cache a signal
        self.device.receive(self.device.transmit())
        
    def test_init_validation(self) -> None:
        """Test initialization validation"""
        
        with self.assertRaises(ValueError):
            PAPR(Mock(), AntennaMode.DUPLEX)

    def test_properties(self) -> None:
        """Test static properties"""
        
        self.assertEqual("PAPR", self.tx_papr.abbreviation)
        self.assertEqual("Peak-to-Average Power Ratio", self.tx_papr.title)

    def test_evaluation(self) -> None:
        """Test evaluation and result generation"""

        tx_artifact = self.tx_papr.evaluate().artifact()
        rx_artifact = self.rx_papr.evaluate().artifact()

        # Collect drop artifacts
        tx_result = self.tx_papr.initialize_result([GridDimensionInfo([SamplePoint(0)], 'x', 'linear', ValueType.LIN)])
        rx_result = self.rx_papr.initialize_result([GridDimensionInfo([SamplePoint(0)], 'x', 'linear', ValueType.LIN)])
        
        tx_result.add_artifact((0,), tx_artifact)
        rx_result.add_artifact((0,), rx_artifact)


class TestSignalExtraction(TestCase):
    """Test signal extraction from simulation runtimes"""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.signal = Signal.Create(self.rng.standard_normal((1, 10)), 1e6, 1e8)
        self.device_state = SimulatedDevice().state()
        
        self.transmit_target = SignalTransmitter(self.signal)
        self.receive_target = SignalReceiver(self.signal.num_samples, self.signal.sampling_rate)
        
        self.transmit_extractor = SignalExtractor(self.transmit_target)
        self.receive_extractor = SignalExtractor(self.receive_target)

        self.grid = [GridDimensionInfo([0], 'random_dimensions', 'linear', ValueType.LIN)]
        self.transmit_result = self.transmit_extractor.initialize_result(self.grid)
        self.receive_result = self.receive_extractor.initialize_result(self.grid)

    def test_init_validation(self) -> None:
        """Test initialization validation"""

        with self.assertRaises(TypeError):
            _ = SignalExtractor(Mock())

    def test_evaluation(self) -> None:
        """Test the evaluation workflow"""

        # Generate transmission and reception
        transmission = self.transmit_target.transmit(self.device_state)
        reception = self.receive_target.receive(transmission.signal, self.device_state)

        # Generate extractions
        transamit_extraction = self.transmit_extractor.evaluate()
        receive_extraction = self.receive_extractor.evaluate()
    
        # Add extractions to results
        num_artifacts = 3
        for i in range(num_artifacts):
            self.transmit_result.add_artifact((0,), transamit_extraction.artifact())
            self.receive_result.add_artifact((0,), receive_extraction.artifact())

        # Compile results to array
        transmit_array = self.transmit_result.to_array()
        receive_array = self.receive_result.to_array()

        # Assert the arrays are not empty
        expected_array_dimensions = (1, num_artifacts, self.signal.num_streams, self.signal.num_samples)
        self.assertCountEqual(transmit_array.shape, expected_array_dimensions)
        self.assertCountEqual(receive_array.shape, expected_array_dimensions)
