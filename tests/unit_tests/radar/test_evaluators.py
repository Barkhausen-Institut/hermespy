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
from hermespy.core import ValueType, GridDimensionInfo, ScenarioMode, Scenario, SamplePoint
from hermespy.core.pymonte import Evaluation, GridDimension,ScalarEvaluationResult
from hermespy.radar import DetectionProbEvaluator, FMCW, PointDetection, Radar, RadarPointCloud, ReceiverOperatingCharacteristic, ThresholdDetector
from hermespy.radar.evaluators import RadarEvaluator, RocArtifact, RocEvaluation, RocEvaluationResult, RootMeanSquareArtifact, RootMeanSquareError, RootMeanSquareErrorResult, RootMeanSquareEvaluation
from hermespy.simulation import SimulatedDevice, SimulationScenario, N0
from unit_tests.utils import SimulationTestContext

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RadarEvaluatorMock(RadarEvaluator):
    """Mock for the abstract radar evaluator"""

    def evaluate(self) -> Evaluation:
        return Mock()

    def generate_result(self, grid, artifacts):
        return Mock()

    def initialize_result(self, grid):
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
        self.device = SimulatedDevice()
        radar = Radar()
        self.device.transmitters.add(radar)
        self.device.receivers.add(radar)

        self.channel = SingleTargetRadarChannel(1.0, 1.0)
        self.evaluator = RadarEvaluatorMock(radar, self.device, self.device, self.channel)

    def test_init_validation(self) -> None:
        """Radar evelauator should raise ValuError if receiving radar is not registered at receiving device"""

        radar = Radar()
        with self.assertRaises(ValueError):
            RadarEvaluatorMock(radar, self.device, self.device, self.channel)

    def test_properties(self) -> None:
        """Properties should be properly initialized"""

        self.assertIs(self.device, self.evaluator.transmitting_device)
        self.assertIs(self.device, self.evaluator.receiving_device)

    def test_fetch_reception_validation(self) -> None:
        """Reception fetching should raise RuntimeError if no reception is available"""

        with self.assertRaises(RuntimeError):
            self.evaluator._fetch_reception()

    def test_fetch_pcl_validation(self) -> None:
        """Point cloud fetching should raise RuntimeError if no point cloud is available"""

        with patch("hermespy.radar.evaluators.RadarEvaluator._fetch_reception") as fetch_reception_mock:
            reception = Mock()
            reception.cloud = None
            fetch_reception_mock.return_value = reception

            with self.assertRaises(RuntimeError):
                self.evaluator._fetch_pcl()

    def test_fetch_channel_validation(self) -> None:
        """Channel fetching should raise RuntimeError if no channel is available"""

        with self.assertRaises(RuntimeError):
            self.evaluator._fetch_channel()


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

    def test_evaluate(self) -> None:
        """Evaluator should compute the proper detection evaluation"""

        with patch.object(self.evaluator, "_DetectionProbEvaluator__cloud") as cloud_mock:
            cloud_mock.num_points = 0
            evaluation = self.evaluator.evaluate()
            self.assertEqual(0.0, evaluation.artifact().to_scalar())

            cloud_mock.num_points = 5
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
        self.result = RocEvaluationResult(
            [GridDimensionInfo([SamplePoint(0)], "Custom Title", "linear", ValueType.LIN)],
            Mock(spec=ReceiverOperatingCharacteristic),
        )
        self.result.add_artifact((0,), RocArtifact(0.2, 0.5))

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

        result_array = self.result.to_array()
        assert_array_equal([1, 101, 2], result_array.shape)


class TestReciverOperatingCharacteristics(TestCase):
    """Test receiver operating characteristics evaluation"""

    def setUp(self) -> None:
        self.scenario = SimulationScenario()
        self.device = self.scenario.new_device(carrier_frequency=1e9)
        self.channel = SingleTargetRadarChannel(1.0, 1.0)
        self.scenario.set_channel(self.device, self.device, self.channel)

        self.radar = Radar()
        self.radar.waveform = FMCW()
        self.device.transmitters.add(self.radar)
        self.device.receivers.add(self.radar)

        self.evaluator = ReceiverOperatingCharacteristic(
            self.radar,
            self.device,
            self.device,
            self.channel,
        )
        
    def _generate_evaluation(self) -> RocEvaluation:
        """Helper class to generate an evaluation.

        Returns: The evaluation.
        """

        propagation = self.channel.propagate(self.device.transmit(), self.device, self.device)
        self.device.receive(propagation)

        return self.evaluator.evaluate()

    def test_evaluate_validation(self) -> None:
        """Evalzuate should raise runtime errors if cached information is not available"""

        with (
            patch("hermespy.radar.evaluators.RadarEvaluator._fetch_reception") as fetch_reception_mock,
            patch("hermespy.radar.evaluators.RadarEvaluator._fetch_channel") as fetch_channel_mock
        ):
            # No device output
            self.evaluator._ReceiverOperatingCharacteristic__output = Mock()
            self.evaluator._ReceiverOperatingCharacteristic__input = None
            with self.assertRaises(RuntimeError):
                self.evaluator.evaluate()

            # No device input
            self.evaluator._ReceiverOperatingCharacteristic__output = None
            self.evaluator._ReceiverOperatingCharacteristic__input = Mock()
            with self.assertRaises(RuntimeError):
                self.evaluator.evaluate()

    def test_evaluate(self) -> None:
        """Test evaluation extraction"""

        evaluation = self._generate_evaluation()
        self.assertCountEqual(evaluation.cube_h0.data.shape, evaluation.cube_h1.data.shape)

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
            ReceiverOperatingCharacteristic.FromScenarios(mock_h0_scenario, mock_h1_scenario, 1)

        mock_h0_scenario.mode = ScenarioMode.REPLAY
        mock_h1_scenario.mode = ScenarioMode.RECORD

        with self.assertRaises(ValueError):
            ReceiverOperatingCharacteristic.FromScenarios(mock_h0_scenario, mock_h1_scenario, 1)

        mock_h0_scenario.mode = ScenarioMode.REPLAY
        mock_h1_scenario.mode = ScenarioMode.REPLAY

        mock_h0_scenario.num_drops = 0
        mock_h1_scenario.num_drops = 1

        with self.assertRaises(ValueError):
            ReceiverOperatingCharacteristic.FromScenarios(mock_h0_scenario, mock_h1_scenario, 1)

        mock_h0_scenario.num_drops = 1
        mock_h1_scenario.num_drops = 0

        with self.assertRaises(ValueError):
            ReceiverOperatingCharacteristic.FromScenarios(mock_h0_scenario, mock_h1_scenario, 1)

        mock_h0_scenario.num_drops = 1
        mock_h1_scenario.num_drops = 1

        with self.assertRaises(ValueError):
            ReceiverOperatingCharacteristic.FromScenarios(mock_h0_scenario, mock_h1_scenario, 1, h0_operator=Mock())

        mock_h0_scenario.num_operators = 0

        with self.assertRaises(ValueError):
            ReceiverOperatingCharacteristic.FromScenarios(mock_h0_scenario, mock_h1_scenario, 1)

        mock_h0_scenario.num_operators = 1
        mock_h0_scenario.operators = [Mock()]

        with self.assertRaises(ValueError):
            ReceiverOperatingCharacteristic.FromScenarios(mock_h0_scenario, mock_h1_scenario, 1)

        mock_h0_scenario.operators = [Mock(spec=Radar)]

        with self.assertRaises(ValueError):
            ReceiverOperatingCharacteristic.FromScenarios(mock_h0_scenario, mock_h1_scenario, 1, h1_operator=Mock())

        mock_h1_scenario.num_operators = 0

        with self.assertRaises(ValueError):
            ReceiverOperatingCharacteristic.FromScenarios(mock_h0_scenario, mock_h1_scenario, 1)

        mock_h1_scenario.num_operators = 1
        mock_h1_scenario.operators = [Mock()]

        with self.assertRaises(ValueError):
            ReceiverOperatingCharacteristic.FromScenarios(mock_h0_scenario, mock_h1_scenario, 1)

    @patch('hermespy.simulation.scenario.SimulationScenario.mode', ScenarioMode.REPLAY)
    @patch('hermespy.simulation.scenario.SimulationScenario.num_drops', 2)
    def test_from_scenarios(self) -> None:
        """Recall ROC from scenarios should be properly handled"""

        mock_h0_scenario = self.scenario
        mock_h1_scenario = self.scenario

        with patch.object(self.scenario, 'drop') as drop_mock:
            drop_mock.side_effect = self.scenario._drop
            result = ReceiverOperatingCharacteristic.FromScenarios(mock_h0_scenario, mock_h1_scenario, 1)
    
        self.assertIsInstance(result, RocEvaluationResult)

    def test_from_file(self) -> None:
        """Recalling ROCs from the filesystem should be properly handled"""

        with TemporaryDirectory() as tempdir:
            file_path = join(tempdir, "test.hdf")

            self.scenario.record(file_path, campaign="h0_measurements")
            for _ in range(3):
                self.scenario.drop()
            self.scenario.stop()

            self.scenario.record(file_path, campaign="h1_measurements")
            for _ in range(3):
                self.scenario.drop()
            self.scenario.stop()

            result = ReceiverOperatingCharacteristic.FromFile(file_path, "h0_measurements", "h1_measurements")

        self.assertIsInstance(result, RocEvaluationResult)

    def test_evaluation_plotting(self) -> None:
        """Test plotting of an ROC evaluation"""

        evaluation = self._generate_evaluation()

        with SimulationTestContext():
            evaluation.visualize()


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
        self.ground_truth = np.array([[1, 2, 3]], dtype=np.float64)
        self.pcl = RadarPointCloud(3)
        for point in self.ground_truth:
            self.pcl.add_point(PointDetection(point, np.zeros(3), 1.0))

        self.evaluation = RootMeanSquareEvaluation(self.pcl, self.ground_truth)

    def test_artifact(self) -> None:
        """Artifact should be properly computed"""

        artifact = self.evaluation.artifact()
        self.assertEqual(0, artifact.to_scalar())

    def test_visualize(self) -> None:
        """Visualize should not raise any exceptions"""

        with SimulationTestContext():
            self.evaluation.visualize()


class TestRootMeanSquareError(TestCase):
    """Test root mean square error"""

    def setUp(self) -> None:
        self.scenario = SimulationScenario()
        self.device = self.scenario.new_device(carrier_frequency=1e9)
        self.channel = SingleTargetRadarChannel(1.0, 1.0)
        self.scenario.set_channel(self.device, self.device, self.channel)

        self.radar = Radar()
        self.radar.waveform = FMCW()
        self.radar.detector = ThresholdDetector(0.1)
        self.device.transmitters.add(self.radar)
        self.device.receivers.add(self.radar)

        self.evaluator = RootMeanSquareError(self.radar, self.device, self.device, self.channel)

    def test_properties(self) -> None:
        """Properties should be properly handled"""

        self.assertEqual("RMSE", self.evaluator.abbreviation)
        self.assertEqual("Root Mean Square Error", self.evaluator.title)

    def test_evaluate(self) -> None:
        """Evaluate routine should generate the corret evaluation"""

        # Prepare the scenario state for evaluation
        propagation = self.channel.propagate(self.device.transmit(), self.device, self.device)
        self.device.receive(propagation)

        evaluation = self.evaluator.evaluate()
        self.assertIsInstance(evaluation, RootMeanSquareEvaluation)

