# -*- coding: utf-8 -*-
"""Test communication evaluators"""

from os.path import join
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import PropertyMock, patch, Mock

import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_equal

from hermespy.channel import SingleTargetRadarChannel
from hermespy.core.monte_carlo import Evaluation, GridDimension, ScalarEvaluationResult
from hermespy.core.scenario import ScenarioMode, Scenario
from hermespy.radar import DetectionProbEvaluator, FMCW, PointDetection, Radar, RadarPointCloud, ReceiverOperatingCharacteristic, ThresholdDetector
from hermespy.radar.evaluators import RadarEvaluator, RocArtifact, RocEvaluation, RocEvaluationResult, RootMeanSquareArtifact, RootMeanSquareError, RootMeanSquareErrorResult, RootMeanSquareEvaluation
from hermespy.simulation import SimulatedDevice, SimulationScenario

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RadarEvaluatorMock(RadarEvaluator):
    """Mock for the abstract radar evaluator"""

    def evaluate(self) -> Evaluation:
        return Mock()

    @property
    def abbreviation(self) -> str:
        return "Mock"

    @property
    def title(self) -> str:
        return "RadarEvaluator"


class TestRadarEvaluator(TestCase):
    """Test radar evaluator"""

    def setUp(self) -> None:
        device = SimulatedDevice()
        radar = Radar()
        radar.device = device

        channel = SingleTargetRadarChannel(1.0, 1.0)
        channel.alpha_device = device
        channel.beta_device = device

        self.evaluator = RadarEvaluatorMock(radar, channel)

    def test_init_validation(self) -> None:
        """Initialization parameters should be properly validated"""

        channel = SingleTargetRadarChannel(1.0, 1.0)
        radar = Radar()

        with self.assertRaises(ValueError):
            RadarEvaluatorMock(radar, channel)

        channel.alpha_device = SimulatedDevice()
        channel.beta_device = SimulatedDevice()

        with self.assertRaises(ValueError):
            RadarEvaluatorMock(radar, channel)

    def test_device_inference(self) -> None:
        alpha_device = SimulatedDevice()
        beta_device = SimulatedDevice()

        channel = SingleTargetRadarChannel(1.0, 1.0, alpha_device=alpha_device, beta_device=beta_device)

        radar = Radar()
        radar.device = beta_device
        evaluator = RadarEvaluatorMock(radar, channel)
        self.assertIs(beta_device, evaluator.receiving_device)
        self.assertIs(alpha_device, evaluator.transmitting_device)

    def test_device_inference_validation(self) -> None:
        alpha_device = SimulatedDevice()
        beta_device = SimulatedDevice()

        channel = SingleTargetRadarChannel(1.0, 1.0, alpha_device=alpha_device, beta_device=beta_device)

        radar = Radar()
        radar.device = SimulatedDevice()

        with self.assertRaises(ValueError):
            RadarEvaluatorMock(radar, channel)

    def test_generate_result(self) -> None:
        """Result generation should be properly handled"""

        grid = []
        artifacts = np.empty(0, dtype=np.object_)

        result = self.evaluator.generate_result(grid, artifacts)

        self.assertIsInstance(result, ScalarEvaluationResult)


class TestDetectionProbEvaluator(TestCase):
    """Test detection probability evaluation"""

    def setUp(self) -> None:
        self.rng = default_rng(42)
        self.num_frames = 10

        self.threshold = 2.0

        self.radar = Mock()
        self.radar.reception = PropertyMock()

        self.evaluator = DetectionProbEvaluator(self.radar)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""

        self.assertIs(self.radar, self.evaluator.radar)

    def test_properties(self) -> None:
        """Properties should be properly handled"""

        self.assertEqual("PD", self.evaluator.abbreviation)
        self.assertEqual("Probability of Detection Evaluation", self.evaluator.title)

    def test_scalar_cdf(self) -> None:
        """Scalara CDF should be properly computed"""  #

        self.assertEqual(0.0, self.evaluator._scalar_cdf(0.0))
        self.assertEqual(1.0, self.evaluator._scalar_cdf(1.0))

    def test_evaluate_validation(self) -> None:
        """Evaluation routine should raise RuntimeError if no point cloud is available"""

        self.radar.reception.cloud = None

        with self.assertRaises(RuntimeError):
            self.evaluator.evaluate()

    def test_generate_result(self) -> None:
        """Result generation should be properly handled"""

        grid = []
        artifacts = np.empty(0, dtype=np.object_)

        result = self.evaluator.generate_result(grid, artifacts)

        self.assertIsInstance(result, ScalarEvaluationResult)

    def test_evaluate(self) -> None:
        """Evaluator should compute the proper detection evaluation"""

        self.radar.reception.cloud.num_points = 0
        evaluation = self.evaluator.evaluate()
        self.assertEqual(0.0, evaluation.artifact().to_scalar())

        self.radar.reception.cloud.num_points = 1
        evaluation = self.evaluator.evaluate()
        self.assertEqual(1.0, evaluation.artifact().to_scalar())


class TestRocArtifact(TestCase):
    """Test ROC artifact"""

    def setUp(self) -> None:
        self.artifact = RocArtifact(0.2, 0.5)

    def test_properties(self) -> None:
        """Properties should be properly handled"""

        self.assertEqual(0.2, self.artifact.h0_value)
        self.assertEqual(0.5, self.artifact.h1_value)
        self.assertIsNone(self.artifact.to_scalar())

    def test_string_conversion(self) -> None:
        """String conversion should be properly handled"""

        self.assertEqual("(0.2, 0.5)", str(self.artifact))


class TestRocEvaluationResult(TestCase):
    """Test ROC evaluation result"""

    def setUp(self) -> None:
        scenario = SimulationScenario()
        grid = [GridDimension(scenario, "snr", [1, 2, 3], "Custom Title")]
        evaluator = Mock(spec=ReceiverOperatingCharacteristic)

        self.detection_probabilities = np.empty(3, dtype=np.object_)
        self.false_alarm_probabilities = np.empty(3, dtype=np.object_)
        for i in range(3):
            self.false_alarm_probabilities[i] = np.arange(10) / 10
            self.detection_probabilities[i] = np.arange(10) / 10 / (i + 1)

        self.result = RocEvaluationResult(grid, evaluator, self.detection_probabilities, self.false_alarm_probabilities)

    def test_plot(self) -> None:
        """ROC plotting should be properly handled"""

        figure_mock = Mock()
        axes_mock = Mock()
        axes_mock.plot.return_value = [Mock()]
        axes_collection = np.array([[axes_mock]], dtype=np.object_)

        with patch("matplotlib.pyplot.subplots") as subplots_mock:
            subplots_mock.return_value = (figure_mock, axes_collection)
            self.result.visualize()
            axes_mock.plot.assert_called()

    def test_to_array(self) -> None:
        """Conversion to array should be properly handled"""

        expected_array = np.stack((self.detection_probabilities, self.false_alarm_probabilities), axis=-1)
        result_array = self.result.to_array()

        assert_array_equal(expected_array.shape, result_array.shape)


class TestReciverOperatingCharacteristics(TestCase):
    """Test receiver operating characteristics evaluation"""

    def setUp(self) -> None:
        self.scenario = SimulationScenario()
        self.device = self.scenario.new_device(carrier_frequency=1e9)
        self.channel = SingleTargetRadarChannel(1.0, 1.0)
        self.scenario.set_channel(self.device, self.device, self.channel)

        self.radar = Radar()
        self.radar.waveform = FMCW()
        self.radar.device = self.device

        self.evaluator = ReceiverOperatingCharacteristic(self.radar, self.channel)

    def test_init_validation(self) -> None:
        """Initialization parameters should be properly validated"""

        channel = SingleTargetRadarChannel(1.0, 1.0)

        with self.assertRaises(ValueError):
            ReceiverOperatingCharacteristic(self.radar, channel)

    def _generate_evaluation(self) -> RocEvaluation:
        """Helper class to generate an evaluation.

        Returns: The evaluation.
        """

        propagation = self.channel.propagate(self.device.transmit())
        self.device.receive(propagation)

        return self.evaluator.evaluate()

    def test_evaluate_validation(self) -> None:
        """Evaluation routine should raise RuntimeError if radar channel is not specified"""

        evaluator = ReceiverOperatingCharacteristic(self.radar, self.channel)

        with self.assertRaises(RuntimeError):
            evaluator.evaluate()

        # Prepare channel states
        propagation = self.channel.propagate(self.device.transmit())
        self.device.receive(propagation)

        with patch("hermespy.channel.radar_channel.SingleTargetRadarChannel.realization", new_callable=PropertyMock) as realization_mock:
            realization_mock.return_value = None
            with self.assertRaises(RuntimeError):
                self.evaluator.evaluate()

        with patch("hermespy.simulation.simulated_device.SimulatedDevice.output", new_callable=PropertyMock) as output_mock:
            output_mock.return_value = None
            with self.assertRaises(RuntimeError):
                self.evaluator.evaluate()

    def test_evaluate(self) -> None:
        """Test evaluation extraction"""

        evaluation = self._generate_evaluation()
        self.assertCountEqual(evaluation.cube_h0.data.shape, evaluation.cube_h1.data.shape)

    def test_generate_result_empty_grid(self) -> None:
        """Test result generation over an empty grid"""

        artifacts = np.empty(1, dtype=np.object_)
        artifacts[0] = [self._generate_evaluation().artifact() for _ in range(3)]

        result = self.evaluator.generate_result([], artifacts)
        self.assertIsInstance(result, RocEvaluationResult)

    def test_generate_result_full_grid(self) -> None:
        """Test result generation over a full grid"""

        grid = []
        artifacts = np.empty((1, 2, 3), dtype=np.object_)
        for g in range(3):
            dimension = Mock(spec=GridDimension)
            dimension.num_sample_points = g
            grid.append(dimension)

        for x in np.ndindex(artifacts.shape):
            artifacts[x] = [self._generate_evaluation().artifact() for _ in range(3)]

        result = self.evaluator.generate_result(grid, artifacts)
        self.assertIsInstance(result, RocEvaluationResult)

    def test_from_scenarios_validation(self) -> None:
        """Recall ROC from scenarios should raise ValueError for invalid configurations"""

        mock_h0_scenario = Mock(spec=Scenario)
        mock_h1_scenario = Mock(spec=Scenario)
        mock_h0_scenario.operators = []
        mock_h1_scenario.operators = []
        mock_h0_scenario.num_operators = 0
        mock_h1_scenario.num_opeators = 0

        mock_h0_scenario.mode = ScenarioMode.RECORD
        mock_h1_scenario.mode = ScenarioMode.REPLAY

        with self.assertRaises(ValueError):
            ReceiverOperatingCharacteristic.From_Scenarios(mock_h0_scenario, mock_h1_scenario)

        mock_h0_scenario.mode = ScenarioMode.REPLAY
        mock_h1_scenario.mode = ScenarioMode.RECORD

        with self.assertRaises(ValueError):
            ReceiverOperatingCharacteristic.From_Scenarios(mock_h0_scenario, mock_h1_scenario)

        mock_h0_scenario.mode = ScenarioMode.REPLAY
        mock_h1_scenario.mode = ScenarioMode.REPLAY

        mock_h0_scenario.num_drops = 0
        mock_h1_scenario.num_drops = 1

        with self.assertRaises(ValueError):
            ReceiverOperatingCharacteristic.From_Scenarios(mock_h0_scenario, mock_h1_scenario)

        mock_h0_scenario.num_drops = 1
        mock_h1_scenario.num_drops = 0

        with self.assertRaises(ValueError):
            ReceiverOperatingCharacteristic.From_Scenarios(mock_h0_scenario, mock_h1_scenario)

        mock_h0_scenario.num_drops = 1
        mock_h1_scenario.num_drops = 1

        with self.assertRaises(ValueError):
            ReceiverOperatingCharacteristic.From_Scenarios(mock_h0_scenario, mock_h1_scenario, h0_operator=Mock())

        mock_h0_scenario.num_operators = 0

        with self.assertRaises(ValueError):
            ReceiverOperatingCharacteristic.From_Scenarios(mock_h0_scenario, mock_h1_scenario)

        mock_h0_scenario.num_operators = 1
        mock_h0_scenario.operators = [Mock()]

        with self.assertRaises(ValueError):
            ReceiverOperatingCharacteristic.From_Scenarios(mock_h0_scenario, mock_h1_scenario)

        mock_h0_scenario.operators = [Mock(spec=Radar)]

        with self.assertRaises(ValueError):
            ReceiverOperatingCharacteristic.From_Scenarios(mock_h0_scenario, mock_h1_scenario, h1_operator=Mock())

        mock_h1_scenario.num_operators = 0

        with self.assertRaises(ValueError):
            ReceiverOperatingCharacteristic.From_Scenarios(mock_h0_scenario, mock_h1_scenario)

        mock_h1_scenario.num_operators = 1
        mock_h1_scenario.operators = [Mock()]

        with self.assertRaises(ValueError):
            ReceiverOperatingCharacteristic.From_Scenarios(mock_h0_scenario, mock_h1_scenario)

    def test_from_scenarios(self) -> None:
        """Recall ROC from scenarios should be properly handled"""

        mock_h0_scenario = Mock(spec=Scenario)
        mock_h1_scenario = Mock(spec=Scenario)
        mock_h0_scenario.operators = [Mock(spec=Radar)]
        mock_h1_scenario.operators = [Mock(spec=Radar)]
        mock_h0_scenario.num_operators = 1
        mock_h1_scenario.num_operators = 1
        mock_h0_scenario.mode = ScenarioMode.REPLAY
        mock_h1_scenario.mode = ScenarioMode.REPLAY
        mock_h0_scenario.num_drops = 1
        mock_h1_scenario.num_drops = 1

        forwards_propagation = self.channel.propagate(self.device.transmit())
        self.device.process_input(forwards_propagation)
        reception = self.radar.receive()

        mock_h0_scenario.operators[0].reception = reception
        mock_h1_scenario.operators[0].reception = reception

        result = ReceiverOperatingCharacteristic.From_Scenarios(mock_h0_scenario, mock_h1_scenario)
        self.assertIsInstance(result, RocEvaluationResult)

    def test_from_hdf(self) -> None:
        """Recall ROC from HDF should be properly handled"""

        with TemporaryDirectory() as tempdir:
            file_path = join(tempdir, "test.hdf")

            self.scenario.record(file_path, campaign="h0_measurements")
            self.scenario.drop()
            self.scenario.stop()

            self.scenario.record(file_path, campaign="h1_measurements")
            self.scenario.drop()
            self.scenario.stop()

            result = ReceiverOperatingCharacteristic.From_HDF(file_path, "h0_measurements", "h1_measurements")

        self.assertIsInstance(result, RocEvaluationResult)


class TestRootMeanSquareArtifact(TestCase):
    """Test root mean square artifact"""

    def setUp(self) -> None:
        self.num_errors = 5
        self.cummulation = 10

        self.artifact = RootMeanSquareArtifact(self.num_errors, self.cummulation)

    def test_properties(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""

        self.assertEqual(self.num_errors, self.artifact.num_errors)
        self.assertEqual(self.cummulation, self.artifact.cummulation)

    def test_to_scalar(self) -> None:
        """Scalar conversion should be properly handled"""

        self.assertEqual(2 * (self.num_errors / self.cummulation) ** 0.5, self.artifact.to_scalar())

    def test_string_conversion(self) -> None:
        """String conversion should be properly handled"""

        self.assertIsInstance(str(self.artifact), str)


class TestRootMeanSquareEvaluation(TestCase):
    """Test root mean square evaluation"""

    def setUp(self) -> None:
        self.ground_truth = np.array([[1, 2, 3]], dtype=np.float_)
        self.pcl = RadarPointCloud(3)
        for point in self.ground_truth:
            self.pcl.add_point(PointDetection(point, np.zeros(3), 1.0))

        self.evaluation = RootMeanSquareEvaluation(self.pcl, self.ground_truth)

    def test_artifact(self) -> None:
        """Artifact should be properly computed"""

        artifact = self.evaluation.artifact()
        self.assertEqual(0, artifact.to_scalar())


class TestRootMeanSquareError(TestCase):
    """Test root mean square error"""

    def setUp(self) -> None:
        self.scenario = SimulationScenario()
        self.device = self.scenario.new_device(carrier_frequency=1e9)
        self.channel = SingleTargetRadarChannel(1.0, 1.0)
        self.scenario.set_channel(self.device, self.device, self.channel)

        self.radar = Radar()
        self.radar.waveform = FMCW()
        self.radar.device = self.device
        self.radar.detector = ThresholdDetector(0.1)

        self.evaluator = RootMeanSquareError(self.radar, self.channel)

    def test_properties(self) -> None:
        """Properties should be properly handled"""

        self.assertEqual("RMSE", self.evaluator.abbreviation)
        self.assertEqual("Root Mean Square Error", self.evaluator.title)

    def test_evaluate_validation(self) -> None:
        """Evaluation routine should raise RuntimeError for invalid internal states"""

        with patch("hermespy.core.device.Receiver.reception", new_callable=PropertyMock) as reception_mock:
            reception_mock.return_value = None
            with self.assertRaises(RuntimeError):
                self.evaluator.evaluate()

            reception = Mock()
            reception.cloud = None
            reception_mock.return_value = reception

            with self.assertRaises(RuntimeError):
                self.evaluator.evaluate()

            reception.cloud = Mock()

            with self.assertRaises(RuntimeError):
                self.evaluator.evaluate()

    def test_evaluate(self) -> None:
        """Evaluate routine should generate the corret evaluation"""

        # Prepare the scenario state for evaluation
        propagation = self.channel.propagate(self.device.transmit())
        self.device.receive(propagation)

        evaluation = self.evaluator.evaluate()
        self.assertIsInstance(evaluation, RootMeanSquareEvaluation)

    def test_generate_result(self) -> None:
        """Result generation should be properly handled"""

        propagation = self.channel.propagate(self.device.transmit())
        self.device.receive(propagation)

        artifact = self.evaluator.evaluate().artifact()

        artifacts = np.empty(1, dtype=object)
        artifacts[0] = [artifact for _ in range(3)]
        grid = []
        result = self.evaluator.generate_result(grid, artifacts)

        self.assertIsInstance(result, RootMeanSquareErrorResult)
