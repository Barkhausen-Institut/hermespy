# -*- coding: utf-8 -*-

from __future__ import annotations
import logging
from io import StringIO
from os import getenv
from typing import Callable, List
from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

import ray
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_equal
from rich.console import Console

from hermespy.core import ConsoleMode, LogarithmicSequence
from hermespy.core.monte_carlo import ActorRunResult, Evaluation, EvaluationTemplate, EvaluationResult, ScalarEvaluationResult, GridSection, MonteCarlo, MonteCarloActor, MonteCarloSample, \
    SampleGrid, Evaluator, ArtifactTemplate, GridDimension, register, RegisteredDimension, ScalarEvaluationResult, ValueType, MonteCarloResult

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


GENERATE_OUTPUT = getenv('HERMES_TEST_PLOT', 'False').lower() == 'true'


class EvaluationMock(EvaluationTemplate[float]):
    
    def artifact(self) -> ArtifactTemplate[float]:
        
        return ArtifactTemplate[float](self.evaluation)


class TestArtifactTemplate(TestCase):
    """Test the template for scalar artifacts"""
    
    def setUp(self) -> None:
        
        self.artifact_value = 1.2345
        self.artifact = ArtifactTemplate[float](self.artifact_value)
        
    def test_init(self) -> None:
        """Initialization parameter should be properly stored as class attributes"""
        
        self.assertEqual(self.artifact_value, self.artifact.artifact)
        
    def test_artifact(self) -> None:
        """Artifact property should return the represented scalar artifact"""
        
        self.assertEqual(self.artifact_value, self.artifact.artifact)
        
    def test_str(self) -> None:
        """String representation should return a string"""
        
        self.assertIsInstance(self.artifact.__str__(), str)
        
    def test_to_scalar(self) -> None:
        """Scalar conversion routine should return the represented artifact"""

        self.assertEqual(self.artifact_value, self.artifact.to_scalar())


class EvaluatorMock(Evaluator):

    def evaluate(self) -> Evaluation:
        return Mock()

    @property
    def abbreviation(self) -> str:
        return "?"

    @property
    def title(self) -> str:
        return "??"

    def generate_result(self, grid: List[GridDimension], artifacts: np.ndarray) -> EvaluationResult:
        return Mock()

class TestEvaluator(TestCase):
    """Test base class for all evaluators"""

    def setUp(self) -> None:

        self.rng = default_rng(42)
        self.evaluator = EvaluatorMock()
        self.evaluator.tolerance = .2

    def test_init(self) -> None:
        """Initialization should set the proper default attributes"""

        self.assertEqual(1., self.evaluator.confidence)
        self.assertEqual(.2, self.evaluator.tolerance)

    def test_confidence_setget(self) -> None:
        """Confidence property getter should return setter argument"""

        confidence = .5
        self.evaluator.confidence = confidence

        self.assertEqual(confidence, self.evaluator.confidence)

    def test_confidence_validation(self) -> None:
        """Confidence property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.evaluator.confidence = -1.

        with self.assertRaises(ValueError):
            self.evaluator.confidence = 1.5

        try:

            self.evaluator.confidence = 0.
            self.evaluator.confidence = 1.

        except ValueError:
            self.fail()

    def test_tolerance_setget(self) -> None:
        """Tolerance property getter should return setter argument"""

        tolerance = .5
        self.evaluator.tolerance = tolerance

        self.assertEqual(tolerance, self.evaluator.tolerance)

    def test_tolerance_validation(self) -> None:
        """Confidence margin property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.evaluator.tolerance = -1.

        try:

            self.evaluator.tolerance = 0.
            self.evaluator.tolerance = 1.

        except ValueError:
            self.fail()
            
    def test_str(self) -> None:
        """Evaluator string representation should return a string"""
        
        self.assertEqual(self.evaluator.abbreviation, self.evaluator.__str__())
            
    def test_scalar_cdf(self) -> None:
        """Scalar cumulitive distribution function should return the cumulative probability"""

        cdf_low = self.evaluator._scalar_cdf(0.)
        cdf_high = self.evaluator._scalar_cdf(1.)
        
        self.assertTrue(cdf_low < cdf_high)

    def test_plot_scale_setget(self) -> None:
        """Plot scale property getter should return setter argument"""

        plot_scale = 'abce'
        self.evaluator.plot_scale = plot_scale

        self.assertEqual(plot_scale, self.evaluator.plot_scale)

# ToDo: Move confidence testing to GridSection     
#    def test_confidence_level_validation(self) -> None:
#        """Confidence level should raise a ValueError on invalid arguments"""
#        
#        with self.assertRaises(ValueError):
#            _ = self.evaluator.confidence_level(self.rng.random((5, 4)))
#
#    def test_confidence_level_no_tolerance(self) -> None:
#        """With a tolerance of zero the confidence should always be zero as well"""
#
#        scalars = self.rng.normal(size=10)
#        self.evaluator.tolerance = 0.
#        
#        self.assertEqual(0., self.evaluator.confidence_level(scalars))
#        
#    def test_confidence_level_no_variance(self) -> None:
#        """The confidence without sample variance should always be one"""
#
#        scalars = np.ones(5)   
#        self.assertEqual(1., self.evaluator.confidence_level(scalars))
#
#    def test_confidence_level_normal(self) -> None:
#        """Test the confidence level estimation for a normal distribution prior assumption"""
#        
#        scalars = self.rng.normal(10, size=100)
#        
#        with patch('hermespy.core.Evaluator._scalar_cdf') as cdf:
#            cdf.side_effect = lambda x: norm.cdf(x)
#            
#            self.evaluator.tolerance = .1
#            confidence_level_low_tolerance = np.array([self.evaluator.confidence_level(scalars[:s]) for s in range(0, len(scalars))], dtype=float)
#            
#            self.evaluator.tolerance = .5
#            confidence_level_high_tolerance = np.array([self.evaluator.confidence_level(scalars[:s]) for s in range(0, len(scalars))], dtype=float)
#            
#            cdf.assert_called()
#            self.assertTrue(np.any(confidence_level_low_tolerance <= confidence_level_high_tolerance))
#
#    def test_confidence_level_bounded_normal(self) -> None:
#        """Test the confidence level estimation for a bounded normal distribution prior assumption"""
#        
#        mu = .5
#        lower = 0.
#        upper = 1.
#        scale = .1
#        dist = truncnorm((lower - mu) / scale, (upper - mu) / scale, loc=mu, scale=scale)
#        prior_dist = truncnorm(lower, upper, loc=0., scale=1.)
#        
#        scalars = dist.rvs(200)
#        
#        with patch('hermespy.core.Evaluator._scalar_cdf') as cdf:
#            cdf.side_effect = lambda x: prior_dist.cdf(x)
#            
#            self.evaluator.tolerance = .01
#            confidence_level_low_tolerance = np.array([self.evaluator.confidence_level(scalars[:s]) for s in range(0, len(scalars))], dtype=float)
#            
#            self.evaluator.tolerance = .5
#            confidence_level_high_tolerance = np.array([self.evaluator.confidence_level(scalars[:s]) for s in range(0, len(scalars))], dtype=float)
#            
#            cdf.assert_called()
#            self.assertTrue(np.any(confidence_level_low_tolerance <= confidence_level_high_tolerance))


class MockEvaluationResult(EvaluationResult):
    """Mock of an evaluation result"""

    def to_array(self) -> np.ndarray:
        return np.empty(0, dtype=np.float_)


class TestEvaluationResult(TestCase):
    """Test evaluation result base class"""

    def setUp(self) -> None:
        
        self.result = MockEvaluationResult()
        
    def test_linear_plotting(self) -> None:
        """Linear plotting should call the correct plotting routine"""
        
        grid = [GridDimension(TestObjectMock(), 'property_b', np.arange(10), tick_format=ValueType.DB)]
        sample_points = np.arange(10)
        scalar_data = np.arange(10)
        evaluator = Mock()
        axes = Mock()
        axes_collection = np.array([[axes]], dtype=np.object_)
        
        self.result._plot_linear(grid, sample_points, scalar_data, evaluator, axes_collection)
        axes.plot.assert_called_once()

    def test_surface_plotting(self) -> None:
        """Surface plotting should call the correct plotting routine"""
        
        grid = [GridDimension(TestObjectMock(), 'property_b', np.arange(10)) for _ in range(2)]
        sample_points = np.arange(10)
        scalar_data = np.random.uniform(size=(10, 10))
        evaluator = Mock()
        axes = Mock()
        axes_collection = np.array([[axes]], dtype=np.object_)
        
        self.result._plot_surface(grid, sample_points, scalar_data, evaluator, axes_collection)
        axes.plot_surface.assert_called_once()

    def test_multidim_plotting(self) -> None:
        """Multidimensional plotting should call the correct plotting routine"""

        grid = [GridDimension(TestObjectMock(), 'property_b', np.arange(10)) for _ in range(3)]
        scalar_data = np.random.uniform(size=(10, 10, 10))
        axes = Mock()
        axes_collection = np.array([[axes]], dtype=np.object_)
        
        self.result._plot_multidim(grid, scalar_data, 0, 'lin', 'lin', 'lin', axes_collection)
        axes.plot.assert_called()

    def test_multidim_plotting_no_labels(self) -> None:
        """Multidimensional plotting should call the correct plotting routine"""
        
        grid = [GridDimension(TestObjectMock(), 'property_b', np.arange(10))]
        scalar_data = np.random.uniform(size=(10))
        axes = Mock()
        axes_collection = np.array([[axes]], dtype=np.object_)
        
        self.result._plot_multidim(grid, scalar_data, 0, 'lin', 'lin', 'lin', axes_collection)
        axes.plot.assert_called()
        
    def test_empty_plotting(self) -> None:
        """Empty plotting should call the correct plotting routine"""
        
        axes = Mock()
        axes_collection = np.array([[axes]], dtype=np.object_)
        
        self.result._plot_empty(axes_collection)
        axes.text.assert_called_once()


class TestScalarEvaluationResult(TestCase):
    """Test processed scalar evaluation result class"""
    
    def test_linear_plotting(self) -> None:
        """Linear plotting should call the correct plotting routine"""
        
        grid = [GridDimension(TestObjectMock(), 'property_b', np.arange(10)) for _ in range(1)]
        scalar_data = np.random.uniform(size=(10, 10))
        evaluator = Mock()
        
        result = ScalarEvaluationResult(grid, scalar_data, evaluator)
            
        axes = MagicMock(spec=np.ndarray)
        figure = MagicMock(spec=plt.Figure)
        
        with patch('matplotlib.pyplot.subplots') as subplots_mock:
            
            subplots_mock.return_value = (figure, axes)
            _ = result.plot()
    
        axes.flat[0].plot.assert_called_once()
                
    def test_surface_plotting(self) -> None:
        """Surface plotting should call the correct plotting routine"""
        
        grid = [GridDimension(TestObjectMock(), 'property_b', np.arange(10)) for _ in range(2)]
        scalar_data = np.random.uniform(size=(10, 10))
        evaluator = Mock()
        
        result = ScalarEvaluationResult(grid, scalar_data, evaluator)
        
        axes = MagicMock(spec=np.ndarray)
        figure = MagicMock(spec=plt.Figure)
        with patch('matplotlib.pyplot.subplots') as subplots_mock:
            subplots_mock.return_value = (figure, axes)
            _ = result.plot()
            
        axes.flat[0].plot_surface.assert_called_once()
        
    def test_multidim_plotting(self) -> None:
        """Multidimensional plotting should call the correct plotting routine"""
        
        grid = [GridDimension(TestObjectMock(), 'property_b', np.arange(10)) for _ in range(3)]
        scalar_data = np.random.uniform(size=(10, 10))
        evaluator = Mock()
        
        result = ScalarEvaluationResult(grid, scalar_data, evaluator)
            
        axes = MagicMock(spec=np.ndarray)
        figure = MagicMock(spec=plt.Figure)
        with patch('matplotlib.pyplot.subplots') as subplots_mock:
            
            subplots_mock.return_value = (figure, axes)
            _ = result.plot()
            
        axes.flat[0].plot.assert_called()
        
    def test_plot_no_data(self) -> None:
        """Even without grid dimensions an empty figure should be generated"""
        
        with patch('matplotlib.pyplot.figure'):
    
            evaluation_result = ScalarEvaluationResult([], np.empty(0, dtype=object), EvaluatorMock())
            figure = evaluation_result.plot()
            self.assertIsInstance(figure, Mock)   
        
    def test_to_array(self) -> None:
        """Array conversion should return the correct array"""
        
        grid = [GridDimension(TestObjectMock(), 'property_b', np.arange(10)) for _ in range(1)]
        scalar_data = np.random.uniform(size=(10, 10))
        evaluator = Mock()
        
        result = ScalarEvaluationResult(grid, scalar_data, evaluator)
        
        assert_array_equal(scalar_data, result.to_array())
    

class TestObjectMock(object):
    """Mock of a tested object"""

    def __init__(self) -> None:

        self.property_a = None
        self.property_b = 0
        self.property_c = 0

    @register(first_impact='init_stage', last_impact='exit_stage')
    @property
    def property_a(self):
        return self.__property_a

    @property_a.setter
    def property_a(self, value) -> None:
        self.__property_a = value

    @property
    def property_b(self):
        return self.__property_b

    @property_b.setter
    def property_b(self, value):
        self.__property_b = value

    @property
    def property_c(self):
        return self.__property_c

    @property_c.setter
    def property_c(self, value):
        self.__property_c = value

    def some_operation(self):
        return 2 * self.__property_a + self.__property_b + self.__property_c


class SumEvaluator(Evaluator):
    """An evaluator summing up object properties"""
    
    __investigated_object: TestObjectMock
    
    def __init__(self, investigated_object: TestObjectMock) -> None:
        
        self.__investigated_object = investigated_object
        Evaluator.__init__(self)

    def evaluate(self) -> EvaluationMock:

        summed = self.__investigated_object.property_a + self.__investigated_object.property_b + self.__investigated_object.property_c
        return EvaluationMock(summed)

    @property
    def abbreviation(self) -> str:
        return "SUM"

    @property
    def title(self) -> str:
        return "Sum Evaluator"
    
    def generate_result(self, grid: List[GridDimension], artifacts: np.ndarray) -> EvaluationResult:
        
        return ScalarEvaluationResult.From_Artifacts(grid, artifacts, self)


class ProductEvaluator(Evaluator):
    """An evaluator multiplying object properties"""
    
    __investigated_object: TestObjectMock
    
    def __init__(self, investigated_object: TestObjectMock) -> None:
        
        self.__investigated_object = investigated_object
        Evaluator.__init__(self)
        
    def evaluate(self) -> EvaluationMock:

        product = self.__investigated_object.property_a * self.__investigated_object.property_b * self.__investigated_object.property_c
        return EvaluationMock(product)

    @property
    def abbreviation(self) -> str:
        return "Product"

    @property
    def title(self) -> str:
        return "Product Evaluator"
    
    def generate_result(self, grid: List[GridDimension], artifacts: np.ndarray) -> EvaluationResult:
        
        return ScalarEvaluationResult.From_Artifacts(grid, artifacts, self)

class TestMonteCarloSample(TestCase):
    """Test the Monte Carlo sample class"""

    def setUp(self) -> None:

        self.grid_section = (0, 1, 2, 3)
        self.sample_index = 5
        
        self.evaluation_artifacts = []
        for _ in range(5):
            
            artifact = Mock()
            artifact.to_scalar.return_value = 1.
            self.evaluation_artifacts.append(artifact)

        self.sample = MonteCarloSample(self.grid_section, self.sample_index, self.evaluation_artifacts)

    def test_init(self) -> None:
        """Initialization arguments should be properly stored as object attributes"""

        self.assertCountEqual(self.grid_section, self.sample.grid_section)
        self.assertEqual(self.sample_index, self.sample.sample_index)
        self.assertCountEqual(self.evaluation_artifacts, self.sample.artifacts)
        self.assertEqual(len(self.evaluation_artifacts), self.sample.num_artifacts)
        
    def test_artifact_scalars(self) -> None:
        """Artifact scalars property should call the artifact conversion routine for each scalar"""
        
        scalars = self.sample.artifact_scalars
        
        self.assertEqual(len(self.evaluation_artifacts), len(scalars))
        
        for artifact in self.evaluation_artifacts:
            artifact.to_scalar.assert_called()


class TestGridSection(TestCase):
    """Test the grid section class representation"""
    
    def setUp(self) -> None:
        
        self.coordiantes = (0, 4, 2)
        self.num_evaluators = 2
        self.investigated_object = TestObjectMock()
        self.evaluators = [SumEvaluator(self.investigated_object), ProductEvaluator(self.investigated_object)]

        self.section = GridSection(self.coordiantes, self.num_evaluators)
    
    def test_init(self) -> None:
        """Initialization parameters shuld be properly stored as class attributes"""
        
        self.assertEqual(self.coordiantes, self.section.coordinates)
        
    def test_num_samples(self) -> None:
        """Number of samples property should return the correct amount of samples"""

        self.assertEqual(0, self.section.num_samples)
        
    def test_add_samples_validation(self) -> None:
        """Adding samples with an non-matching number of artifacts and evaluators should raise a ValueError"""
        
        artifacts = []
        for _ in range(self.num_evaluators - 1):
            
            artifact = Mock()
            artifact.to_scalar.return_value = 0.
            artifacts.append(artifact)
            
        sample = MonteCarloSample((0, 4, 2), 0, artifacts)
        
        with self.assertRaises(ValueError):
            self.section.add_samples(sample, self.evaluators)
              
    def test_add_samples(self) -> None:
        """Adding samples should correctly update the confidences"""
        
        artifacts = []
        for _ in range(self.num_evaluators):
            
            artifact = Mock()
            artifact.to_scalar.return_value = 0.
            artifacts.append(artifact)
        
        sample = MonteCarloSample((0, 4, 2), 0, artifacts)
        self.section.add_samples(sample, self.evaluators)
        
        self.assertCountEqual([False, False], self.section.confidences)
        self.assertEqual(1, self.section.num_samples)
        self.assertSequenceEqual([sample], self.section.samples)
        
    def test_confidence_status(self) -> None:
        """Confidence status should return the correct status"""
   
        confidence = self.section.confidence_status(self.evaluators)
        self.assertFalse(confidence)
        
        for evaluator in self.evaluators:
            evaluator.confidence = 0.
            
        self.assertTrue(self.section.confidence_status(self.evaluators))

    def test_update_confidences_validation(self) -> None:
        """Updating confidences with an non-matching number of artifacts and evaluators should raise a ValueError"""
        
        with self.assertRaises(ValueError):
            self.section._GridSection__update_confidences(np.zeros((1, 2)), self.evaluators)
        
        with self.assertRaises(ValueError):
            self.section._GridSection__update_confidences(np.zeros(1 + len(self.evaluators)), self.evaluators)
        
    def test_update_confidences(self) -> None:
        """Updating confidences should correctly update the confidences"""
        
        self.evaluators[0].tolerance = 0.
        self.evaluators[1].tolerance = .1

        samples = []
        recent_confidence = 0.
        for i in range(10):
            
            artifacts = []
            for _ in range(self.num_evaluators):
                
                artifact = Mock()
                artifact.to_scalar.return_value = np.random.standard_normal()
                artifacts.append(artifact)
                
            samples.append(MonteCarloSample([], i, artifacts))
            self.assertGreaterEqual(self.section.confidences[1], recent_confidence)
            recent_confidence = self.section.confidences[1]

        self.section.add_samples(samples, self.evaluators)
        self.assertFalse(self.section.confidence_status(self.evaluators))


class TestSampleGrid(TestCase):
    """Test the sample grid class representation"""
    
    def setUp(self) -> None:
        
        self.investigated_object = TestObjectMock()
        self.dimensions = [GridDimension(self.investigated_object, 'property_a', [1, 2, 6, 7, 8])]
        self.evaluators = [SumEvaluator(self.investigated_object), ProductEvaluator(self.investigated_object)]
        
        self.grid = SampleGrid(self.dimensions, self.evaluators)
        
    def test_getitem(self) -> None:
        """Getting an item should return the correct grid section"""
        
        section = self.grid[(0,)]
        
        self.assertIsInstance(section, GridSection)
        self.assertSequenceEqual((0,), section.coordinates)

    def test_iteration(self) -> None:
        """Iteration should return the correct grid sections"""
        
        for s, section in enumerate(self.grid):
            
            self.assertIsInstance(section, GridSection)
            self.assertSequenceEqual((s,), section.coordinates)


@ray.remote(num_cpus=1)
class MonteCarloActorMock(MonteCarloActor[TestObjectMock]):
    """Mock of a Monte Carlo Actor"""
    
    def init_stage(self) -> None:
        return
    
    def exit_stage(self) -> None:
        return

    @staticmethod
    def stage_identifiers() -> List[str]:
        return ['init_stage', 'exit_stage']
    
    def stage_executors(self) -> List[Callable]:
        return [self.init_stage, self.exit_stage]

         
class TestMonteCarloActor(TestCase):
    """Test the Monte Carlo actor"""
    
    @classmethod
    def setUpClass(cls) -> None:

        ray.init(local_mode=True, num_cpus=1, logging_level=logging.ERROR)
            
    @classmethod
    def tearDownClass(cls):
        
        ray.shutdown()


@ray.remote(num_cpus=1)
class MonteCarloActorMock(MonteCarloActor[TestObjectMock]):
    """Mock of a Monte Carlo Actor"""
    
    def init_stage(self) -> None:
        return
    
    def exit_stage(self) -> None:
        return

    @staticmethod
    def stage_identifiers() -> List[str]:
        return ['init_stage', 'exit_stage']
    
    def stage_executors(self) -> List[Callable]:
        return [self.init_stage, self.exit_stage]

         
class TestMonteCarloActor(TestCase):
    """Test the Monte Carlo actor"""
    
    @classmethod
    def setUpClass(cls) -> None:

        ray.init(local_mode=True, num_cpus=1, ignore_reinit_error=True, logging_level=logging.ERROR)
            
    @classmethod
    def tearDownClass(cls) -> None:

        # Shut down ray 
        ray.shutdown()

    def setUp(self) -> None:
        
        self.investigated_object = TestObjectMock()
        self.investigated_object.property = 1
        self.dimensions = [GridDimension(self.investigated_object, 'property_a', [1, 2, 6, 7, 8])]
        self.evaluators = [SumEvaluator(self.investigated_object), ProductEvaluator(self.investigated_object)]

    def test_run(self) -> None:
        """Running the actor should produce the expected result"""

        actor = MonteCarloActorMock.remote((self.investigated_object, self.dimensions, self.evaluators), 0)

        for sample_idx in range(self.dimensions[0].num_sample_points - 1):

            program = [(sample_idx,), (1 + sample_idx,)]

            result: ActorRunResult = ray.get(actor.run.remote(program))
            self.assertEqual(2, len(result.samples))

    def test_run_exepction_handling(self) -> None:
        """Running the actor should produce the expected result when throwing exepctions"""
        
        def throw_exepction(*args):
            raise RuntimeError('test')
        
        with patch.object(self.dimensions[0], 'configure_point') as configure_point_mock:
            
            configure_point_mock.side_effect = throw_exepction  
            patched_actor = MonteCarloActorMock.remote((self.investigated_object, self.dimensions, self.evaluators), 0)

            result: ActorRunResult = ray.get(patched_actor.run.remote([(0,)]))
            self.assertEqual('test', result.message)


class TestMonteCarloResult(TestCase):
    """Test the result class"""
    
    def setUp(self) -> None:
        
        self.investigated_object = TestObjectMock()
        self.dimensions = [GridDimension(self.investigated_object, 'property_a', [1, 2, 6, 7, 8])]
        self.evaluators = [SumEvaluator(self.investigated_object), ProductEvaluator(self.investigated_object)]
        self.grid = SampleGrid(self.dimensions, self.evaluators)
        self.performance = 1.2345
        
        self.result = MonteCarloResult(self.dimensions, self.evaluators, self.grid, self.performance)
        
    def test_properties(self) -> None:
        """Properties should return the correct values"""

        self.assertEqual(self.performance, self.result.performance_time)
        
    def test_plot(self) -> None:
        """Plotting should call the correct plotting routine"""
        
        figure_mock = MagicMock(spec=plt.Figure)
        axes_mock = MagicMock(spec=np.ndarray)
        with patch('matplotlib.pyplot.subplots') as subplots_mock:
            
            subplots_mock.return_value = figure_mock, axes_mock
            _ = self.result.plot()
            
        subplots_mock.assert_called()
        
    def test_save_to_matlab(self) -> None:
        """Saving to Matlab should call the correct routine"""
        
        with patch('hermespy.core.monte_carlo.savemat') as savemat_mock:
            
            self.result.save_to_matlab('test.mat')
            savemat_mock.assert_called()
            
    def test_evaluation_results(self) -> None:
        """Evaluation results should return the correct results"""
        
        self.assertEqual(len(self.evaluators), len(self.result.evaluation_results))

        
class TestGridDimension(TestCase):
    """Test the simulation grid dimension class"""

    def setUp(self) -> None:
        
        class MockObject(object):
            
            def __init__(self) -> None:
                
                self.__dimension = 1234
                
            @register(first_impact='a', last_impact='b', title='testtitle')
            @property
            def dimension(self) -> int:
                
                return self.__dimension
            
            @dimension.setter
            def dimension(self, value: float) -> None:
                """Setter should raise a ValueError on invalid arguments"""
                
                self.__dimension = value
            
            def set_dimension(self, value: float) -> None:
                self.dimension = value

        self.considered_object = MockObject()
        self.sample_points = [1, 2, 3, 4]

        self.dimension = GridDimension(self.considered_object, 'dimension', self.sample_points)

    def test_logarithmic_init(self) -> None:
        """Initialization with logarithmic sample points should configure the plot scale to logarithmic"""
        
        sample_points = LogarithmicSequence(self.sample_points)
        dimension = GridDimension(self.considered_object, 'dimension', sample_points)

        self.assertEqual('log', dimension.plot_scale)
        
    def test_plot_scale_init(self) -> None:
        """Initialization with a plot scale should set the plot scale"""
        
        dimension = GridDimension(self.considered_object, 'dimension', self.sample_points, plot_scale='log', tick_format=ValueType.DB)
        
        self.assertEqual('log', dimension.plot_scale)
        self.assertEqual(ValueType.DB, dimension.tick_format)
        
    def test_init_validation(self) -> None:
        """Initialization should ra ise ValueErrors on invalid arguments"""
        
        with self.assertRaises(ValueError):
            GridDimension(self.considered_object, 'nonexistentdimension', self.sample_points)

        with self.assertRaises(ValueError):
            GridDimension(self.considered_object, 'dimension', [])
            
    def test_registered_dimension_validation(self) -> None:
        """Initialization should raise ValueError if multiple registered dimesions don't share impacts"""
        
        class MockObjetB(object):
            
            def __init__(self) -> None:
                
                self.__dimension = 1234
                
            @register(first_impact='c', last_impact='b')
            @property
            def dimension(self) -> int:
                
                return self.__dimension
            
            @dimension.setter
            def dimension(self, value: float) -> None:
                """Setter should raise a ValueError on invalid arguments"""
                
                self.__dimension = value
            
            def set_dimension(self, value: float) -> None:
                self.dimension = value
                
        class MockObjetC(object):
            
            def __init__(self) -> None:
                
                self.__dimension = 1234
                
            @register(first_impact='a', last_impact='d')
            @property
            def dimension(self) -> int:
                
                return self.__dimension
            
            @dimension.setter
            def dimension(self, value: float) -> None:
                """Setter should raise a ValueError on invalid arguments"""
                
                self.__dimension = value
            
            def set_dimension(self, value: float) -> None:
                self.dimension = value
                
        mock_b = MockObjetB()
        mock_c = MockObjetC()
        
        with self.assertRaises(ValueError):
            _ = GridDimension((self.considered_object, mock_b), 'dimension', self.sample_points)
            
        with self.assertRaises(ValueError):
            _ = GridDimension((self.considered_object, mock_c), 'dimension', self.sample_points)
            
    def test_function_dimension(self) -> None:
        """Test pointing to a function instead of a property"""
        
        self.dimension = GridDimension(self.considered_object, 'set_dimension', self.sample_points)

        self.dimension.configure_point(0)
        self.assertEqual(self.sample_points[0], self.considered_object.dimension)

    def test_considered_object(self) -> None:
        """Considered object property should return considered object"""

        self.assertIs(self.considered_object, self.dimension.considered_objects[0])

    def test_dimension(self) -> None:
        """Dimension propertsy should return the correct dimension"""
        
        self.assertEqual('dimension', self.dimension.dimension)

    def test_sample_points(self) -> None:
        """Sample points property should return sample points"""

        self.assertIs(self.sample_points, self.dimension.sample_points)

    def test_num_sample_points(self) -> None:
        """Number of sample points property should return the correct amount of sample points"""

        self.assertEqual(4, self.dimension.num_sample_points)
        
    def test_configure_point_validation(self) -> None:
        """Configuring a point with an invalid index should raise a ValueError"""

        with self.assertRaises(ValueError):
            self.dimension.configure_point(4)

    def test_configure_point(self) -> None:
        """Configuring a point should set the property correctly"""

        expected_value = self.sample_points[3]
        self.dimension.configure_point(3)

        self.assertEqual(expected_value, self.considered_object.dimension)

    def test_title(self) -> None:
        """Title property should infer the correct title"""

        self.dimension.title = None
        self.assertEqual("dimension", self.dimension.title)

        self.dimension.title = "xyz"
        self.assertEqual("xyz", self.dimension.title)

    def test_plot_scale_setget(self) -> None:
        """Plot scale property getter should return setter argument"""

        scale = 'loglin'
        self.dimension.plot_scale = scale

        self.assertEqual(scale, self.dimension.plot_scale)
        

class TestRegisteredDimension(TestCase):
    """Test the registered dimension"""
    
    def setUp(self) -> None:
        
        self.property = property(lambda: 10)
        self.first_impact = 'a'
        self.last_impact = 'b'
        self.title = 'c'
        
        self.dimension = RegisteredDimension(self.property, self.first_impact, self.last_impact, self.title)
    
    def test_is_registered(self) -> None:
        """Is registered should return the correct result"""
        
        self.assertTrue(RegisteredDimension.is_registered(self.dimension))
        
    def test_properties(self) -> None:
        """Properties should return the correct values"""
        
        self.assertEqual(self.first_impact, self.dimension.first_impact)
        self.assertEqual(self.last_impact, self.dimension.last_impact)
        self.assertEqual(self.title, self.dimension.title)
        
    def test_getter(self) -> None:
        """Getter should return the correct value"""
        
        self.assertIsInstance(self.dimension.getter(Mock()), RegisteredDimension)
    
    def test_setter(self) -> None:
        """Setter should return the correct value"""
        
        self.assertIsInstance(self.dimension.setter(Mock()), RegisteredDimension)
    
    def test_deleter(self) -> None:
        """Deleter should return the correct value"""
        
        self.assertIsInstance(self.dimension.deleter(Mock()), RegisteredDimension)
    
    def test_decoration(self) -> None:
        """The decorator should return a property registered within the simulation registry"""
        
        expected_first_impact_a = '123124'
        expected_last_impact_a = '21341312'
        
        expected_first_impact_b = '1231223234'
        expected_last_impact_b = '213413123232'
        
        class TestClassA():
            
            def __init__(self) -> None:
                self.__value_a = 1.2345
            
            @register(first_impact=expected_first_impact_a, last_impact=expected_last_impact_a)
            @property
            def test_dimension(self) -> float:
                return self.__value_a
            
            @test_dimension.setter
            def test_dimension(self, value: float) -> None:
                self.__value_a = value
                
        class TestClassB():
            
            def __init__(self) -> None:
                self.test_dimension = 6.789
                
            @register(first_impact=expected_first_impact_b, last_impact=expected_last_impact_b)
            @property
            def test_dimension(self) -> float:
                return self.__value_b
            
            @test_dimension.setter
            def test_dimension(self, value: float) -> None:
                self.__value_b = value
            
        self.assertTrue(RegisteredDimension.is_registered(TestClassA.test_dimension))
        self.assertEqual(expected_first_impact_a, TestClassA.test_dimension.first_impact)
        self.assertEqual(expected_last_impact_a, TestClassA.test_dimension.last_impact)

        self.assertTrue(RegisteredDimension.is_registered(TestClassB.test_dimension))
        self.assertEqual(expected_first_impact_b, TestClassB.test_dimension.first_impact)
        self.assertEqual(expected_last_impact_b, TestClassB.test_dimension.last_impact)

        expected_value_a = 1.2345
        expected_value_b = 6.7890
        
        class_a = TestClassA()
        class_b = TestClassB()
        class_a.test_dimension = expected_value_a
        class_b.test_dimension = expected_value_b
        
        self.assertEqual(expected_value_a, class_a.test_dimension)
        self.assertEqual(expected_value_b, class_b.test_dimension)


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

        self.monte_carlo = MonteCarlo(investigated_object=self.investigated_object,
                                      evaluators=self.evaluators,
                                      num_samples=self.num_samples,
                                      num_actors=self.num_actors,
                                      console=self.console,
                                      console_mode=ConsoleMode.INTERACTIVE,
                                      database_caching=True,
                                      progress_log_interval=-1.)

    def test_init(self) -> None:
        """Initialization arguments should be properly stored as class attributes"""

        self.assertIs(self.investigated_object, self.monte_carlo.investigated_object)
        self.assertEqual(self.num_samples, self.monte_carlo.num_samples)
        self.assertEqual(self.num_actors, self.monte_carlo.num_actors)
        self.assertIs(self.console, self.monte_carlo.console)
        
    def test_ray_init(self) -> None:
        """Ray should be initialized if not already initialized"""
        
        with patch('hermespy.core.monte_carlo.ray.is_initialized') as is_initialized_mock, \
            patch('hermespy.core.monte_carlo.ray.init') as init_mock:
                
            is_initialized_mock.return_value = False
            _ = MonteCarlo(self.investigated_object, 1)
            
            init_mock.assert_called_once()

    def test_new_dimension(self) -> None:
        """Test adding a new grid dimension"""

        dimension_str = 'property_a'
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
        
        dimensions = {
                'property_a': [1, Mock()],
                'property_b': [1, 1e2],
                'property_c': LogarithmicSequence([1, 2, 3]),
            }
        
        dimensions = [self.monte_carlo.new_dimension(dimension, parameters) for dimension, parameters in dimensions.items()]
        result = self.monte_carlo.simulate(MonteCarloActorMock)
        
        self.assertEqual(2, len(result.evaluation_results))
        
    def test_simulate_silent(self) -> None:
        """The simulation routine should not print anything if in silent mode"""
        
        self.monte_carlo.console_mode = ConsoleMode.SILENT

        _ = self.monte_carlo.new_dimension('property_a', [1, 2])
        _ = self.monte_carlo.simulate(MonteCarloActorMock)
        
        if not GENERATE_OUTPUT:
            self.assertEqual('', self.io.getvalue())
            
    def test_simulate_linear(self) -> None:
        """Tes the linear printing simulation routine"""

        self.monte_carlo.console_mode = ConsoleMode.LINEAR

        _ = self.monte_carlo.new_dimension('property_a', [1, 2])
        result = self.monte_carlo.simulate(MonteCarloActorMock)

        self.assertEqual(2, len(result.evaluation_results))
        
    def test_simulate_strict_confidence(self) -> None:
        """Test simulation with strict confidence criteria"""
        
        for evaluator in self.evaluators:
            evaluator.tolerance = 0.    

        _ = self.monte_carlo.new_dimension('property_a', [1, 2])
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
        
        _ = self.monte_carlo.new_dimension('property_a', [1, 2])
        self.assertEqual(6, self.monte_carlo.max_num_samples)

    def test_section_block_size_setget(self) -> None:
        """Section block size property getter should return setter argument"""

        section_block_size = 10
        self.monte_carlo.section_block_size = section_block_size

        self.assertEqual(section_block_size, self.monte_carlo.section_block_size)

    def test_num_actors_validation(self) -> None:
        """Number of actors property should raise ValueError on arguments smaller than one"""

        with self.assertRaises(ValueError):
            self.monte_carlo.num_actors = -1

        with self.assertRaises(ValueError):
            self.monte_carlo.num_actors = 0

    def test_num_actors(self) -> None:
        """Number of actors property should return the correct number of actors"""
        
        self.assertEqual(self.num_actors, self.monte_carlo.num_actors)
        
        self.monte_carlo.num_actors = None
        
        with patch('hermespy.core.monte_carlo.ray.available_resources') as resources_mock:
            
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
        
        self.monte_carlo.console_mode = 'INTERACTIVE'
        self.assertEqual(ConsoleMode.INTERACTIVE, self.monte_carlo.console_mode)
    