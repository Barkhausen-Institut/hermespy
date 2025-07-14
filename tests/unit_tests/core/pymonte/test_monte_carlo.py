# -*- coding: utf-8 -*-

from io import StringIO
from os import getenv
from unittest import TestCase
from unittest.mock import Mock, patch

import ray as ray
from rich.console import Console

from hermespy.core import ConsoleMode
from hermespy.core.pymonte.grid import GridDimension, LogarithmicSequence
from hermespy.core.pymonte.monte_carlo import MonteCarlo, MonteCarloResult
from ...utils import SimulationTestContext
from .object import TestObjectMock
from .test_actors import MonteCarloActorMock
from .test_evaluation import SumEvaluator, ProductEvaluator

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


GENERATE_OUTPUT = getenv("HERMES_TEST_PLOT", "False").lower() == "true"


class TestMonteCarlo(TestCase):
    """Test the simulation grid"""

    @classmethod
    def setUpClass(cls) -> None:
        ray.init(local_mode=True, num_cpus=1, ignore_reinit_error=True, logging_level=logging.ERROR)

    @classmethod
    def tearDownClass(cls) -> None:
        # Shut down ray
        ray.shutdown()

    def setUp(self) -> None:
        self.investigated_object = TestObjectMock()
        self.evaluators = [ProductEvaluator(self.investigated_object), SumEvaluator(self.investigated_object)]
        self.num_samples = 3
        self.num_actors = 2

        self.io = StringIO()
        self.console = Console(file=None if GENERATE_OUTPUT else self.io)

        self.monte_carlo = MonteCarlo(investigated_object=self.investigated_object, evaluators=self.evaluators, num_samples=self.num_samples, num_actors=self.num_actors, console=self.console, console_mode=ConsoleMode.INTERACTIVE, progress_log_interval=-1.0)

    def test_init(self) -> None:
        """Initialization arguments should be properly stored as class attributes"""

        self.assertIs(self.investigated_object, self.monte_carlo.investigated_object)
        self.assertEqual(self.num_samples, self.monte_carlo.num_samples)
        self.assertEqual(self.num_actors, self.monte_carlo.num_actors)
        self.assertIs(self.console, self.monte_carlo.console)

    def test_ray_init(self) -> None:
        """Ray should be initialized if not already initialized"""

        with patch("hermespy.core.monte_carlo.ray.is_initialized") as is_initialized_mock, patch("hermespy.core.monte_carlo.ray.init") as init_mock:
            is_initialized_mock.return_value = False
            _ = MonteCarlo(self.investigated_object, 1)

            init_mock.assert_called_once()

    def test_new_dimension(self) -> None:
        """Test adding a new grid dimension"""

        dimension_str = "property_a"
        sample_points = [1, 2, 3, 4]

        dimension = self.monte_carlo.new_dimension(dimension_str, sample_points)
        self.assertIs(self.investigated_object, dimension.considered_objects[0])

    def test_add_dimension(self) -> None:
        """Test adding a grid dimension"""

        dimension = Mock()
        self.monte_carlo.add_dimension(dimension)
        self.assertIn(dimension, self.monte_carlo.dimensions)

    def test_add_dimension_validation(self) -> None:
        """Adding an already existing dimension should raise a ValueError"""

        dimension = Mock()
        self.monte_carlo.add_dimension(dimension)

        with self.assertRaises(ValueError):
            self.monte_carlo.add_dimension(dimension)

    def test_remove_dimension_validation(self) -> None:
        """Removing a non-existing dimension should raise a ValueError"""

        with self.assertRaises(ValueError):
            self.monte_carlo.remove_dimension(Mock())

    def test_remove_dimension(self) -> None:

        dimension = Mock()
        self.monte_carlo.add_dimension(dimension)
        self.monte_carlo.remove_dimension(dimension)

        self.assertNotIn(dimension, self.monte_carlo.dimensions)

    def test_num_samples_setget(self) -> None:
        """Number of samples property getter should return setter argument"""

        num_samples = 20
        self.monte_carlo.num_samples = num_samples

        self.assertEqual(num_samples, self.monte_carlo.num_samples)

    def test_num_samples_validation(self) -> None:
        """Number of samples property setter should raise ValueError on arguments smaller than one"""

        with self.assertRaises(ValueError):
            self.monte_carlo.num_samples = 0

        with self.assertRaises(ValueError):
            self.monte_carlo.num_samples = -1

    def test_num_actors_setget(self) -> None:
        """Number of actors property getter should return setter argument"""

        num_actors = 10
        self.monte_carlo.num_actors = num_actors

        self.assertEqual(num_actors, self.monte_carlo.num_actors)

    def test_num_actors_validation(self) -> None:
        """Number of actors property setter should raise ValueError on arguments smaller than one"""

        with self.assertRaises(ValueError):
            self.monte_carlo.num_actors = -1

        with self.assertRaises(ValueError):
            self.monte_carlo.num_actors = 0

    def test_simulate(self) -> None:
        """Test the simulation routine"""

        dimensions = {"property_a": [1, Mock()], "property_b": [1, 1e2], "property_c": LogarithmicSequence([1, 2, 3])}

        dimensions = [self.monte_carlo.new_dimension(dimension, parameters) for dimension, parameters in dimensions.items()]
        result = self.monte_carlo.simulate(MonteCarloActorMock)

        self.assertEqual(2, len(result.evaluation_results))

    def test_simulate_silent(self) -> None:
        """The simulation routine should not print anything if in silent mode"""

        self.monte_carlo.console_mode = ConsoleMode.SILENT

        _ = self.monte_carlo.new_dimension("property_a", [1, 2])
        _ = self.monte_carlo.simulate(MonteCarloActorMock)

        if not GENERATE_OUTPUT:
            self.assertEqual("", self.io.getvalue())

    def test_simulate_linear(self) -> None:
        """Tes the linear printing simulation routine"""

        self.monte_carlo.console_mode = ConsoleMode.LINEAR

        _ = self.monte_carlo.new_dimension("property_a", [1, 2])
        result = self.monte_carlo.simulate(MonteCarloActorMock)

        self.assertEqual(2, len(result.evaluation_results))

    def test_simulate_strict_confidence(self) -> None:
        """Test simulation with strict confidence criteria"""

        for evaluator in self.evaluators:
            evaluator.tolerance = 0.0

        _ = self.monte_carlo.new_dimension("property_a", [1, 2])
        _ = self.monte_carlo.simulate(MonteCarloActorMock)

    def test_add_evaluator(self) -> None:
        """Test adding an evaluator"""

        evaluator = Mock()
        self.monte_carlo.add_evaluator(evaluator)

        self.assertIn(evaluator, self.monte_carlo.evaluators)

    def test_min_num_samples_validation(self) -> None:
        """Minimum number of samples property setter should raise ValueError on negative arguments"""

        with self.assertRaises(ValueError):
            self.monte_carlo.min_num_samples = -1

    def test_min_num_samples_setget(self) -> None:
        """Minimum number of samples property getter should return setter argument"""

        min_num_samples = 10
        self.monte_carlo.min_num_samples = min_num_samples

        self.assertEqual(min_num_samples, self.monte_carlo.min_num_samples)

    def test_max_num_samples(self) -> None:
        """Maximum number of samples should return the correct value"""

        _ = self.monte_carlo.new_dimension("property_a", [1, 2])
        self.assertEqual(6, self.monte_carlo.max_num_samples)

    def test_section_block_size_setget(self) -> None:
        """Section block size property getter should return setter argument"""

        section_block_size = 10
        self.monte_carlo.section_block_size = section_block_size

        self.assertEqual(section_block_size, self.monte_carlo.section_block_size)

    def test_num_actors(self) -> None:
        """Number of actors property should return the correct number of actors"""

        self.assertEqual(self.num_actors, self.monte_carlo.num_actors)

        self.monte_carlo.num_actors = None

        with patch("hermespy.core.monte_carlo.ray.available_resources") as resources_mock:
            get_mock = Mock()
            get_mock.side_effect = 1
            resources_mock.available_resources.side_effect = get_mock

            self.assertEqual(1, self.monte_carlo.num_actors)

    def test_console_setget(self) -> None:
        """Console property getter should return setter argument"""

        console = Mock()
        self.monte_carlo.console = console

        self.assertIs(console, self.monte_carlo.console)

    def test_section_block_size_validation(self) -> None:
        """Section block size property setter should raise ValueError on arguments smaller than one"""

        with self.assertRaises(ValueError):
            self.monte_carlo.section_block_size = -1

        with self.assertRaises(ValueError):
            self.monte_carlo.section_block_size = 0

    def test_cpus_per_actor_validation(self) -> None:
        """CPUs per actor property setter should raise ValueError on arguments smaller than one"""

        with self.assertRaises(ValueError):
            self.monte_carlo.cpus_per_actor = -1

        with self.assertRaises(ValueError):
            self.monte_carlo.cpus_per_actor = 0

    def test_cpus_per_actor_setget(self) -> None:
        """CPUs per actor property getter should return setter argument"""

        cpus_per_actor = 10
        self.monte_carlo.cpus_per_actor = cpus_per_actor

        self.assertEqual(cpus_per_actor, self.monte_carlo.cpus_per_actor)

    def test_console_mode_setget(self) -> None:
        """Console mode property getter should return setter argument"""

        console_mode = ConsoleMode.SILENT
        self.monte_carlo.console_mode = console_mode

        self.assertEqual(console_mode, self.monte_carlo.console_mode)

        self.monte_carlo.console_mode = "INTERACTIVE"
        self.assertEqual(ConsoleMode.INTERACTIVE, self.monte_carlo.console_mode)


class TestMonteCarloResult(TestCase):
    """Test the result class"""

    def setUp(self) -> None:
        self.investigated_object = TestObjectMock()
        self.dimensions = [GridDimension(self.investigated_object, "property_a", [1, 2, 6, 7, 8])]
        self.evaluators = [SumEvaluator(self.investigated_object), ProductEvaluator(self.investigated_object)]
        self.grid = SampleGrid(self.dimensions, self.evaluators)
        self.performance = 1.2345

        self.result = MonteCarloResult(self.dimensions, self.evaluators, self.grid, self.performance)

    def test_properties(self) -> None:
        """Properties should return the correct values"""

        self.assertEqual(self.performance, self.result.performance_time)

    def test_plot(self) -> None:
        """Plotting should call the correct plotting routine"""

        with SimulationTestContext():
            visualizations = self.result.plot()

            for visualization in visualizations:
                visualization.axes[0, 0].plot.assert_called()

    def test_save_to_matlab(self) -> None:
        """Saving to Matlab should call the correct routine"""

        with patch("hermespy.core.monte_carlo.savemat") as savemat_mock:
            self.result.save_to_matlab("test.mat")
            savemat_mock.assert_called()

    def test_evaluation_results(self) -> None:
        """Evaluation results should return the correct results"""

        self.assertEqual(len(self.evaluators), len(self.result.evaluation_results))
