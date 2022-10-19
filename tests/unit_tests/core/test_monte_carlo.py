# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Callable, List
from unittest import TestCase
from unittest.mock import Mock, patch

import ray
import numpy as np
from numpy.random import default_rng
from scipy.stats import norm, truncnorm

from hermespy.core.monte_carlo import ActorRunResult, Evaluation, EvaluationTemplate, EvaluationResult, ScalarEvaluationResult, GridSection, MonteCarlo, MonteCarloActor, MonteCarloSample, \
    Evaluator, ArtifactTemplate, MO, Artifact, GridDimension, RegisteredDimension, dimension

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class EvaluationMock(EvaluationTemplate[float]):
    
    def artifact(self) -> ArtifactTemplate[float]:
        
        return ArtifactTemplate[float](self.evaluation)


class TestArtifactTemplate(TestCase):
    """Test the template for scalar artifacts."""
    
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
    """Test base class for all evaluators."""

    def setUp(self) -> None:

        self.rng = default_rng(42)
        self.evaluator = EvaluatorMock()
        self.evaluator.tolerance = .2

    def test_init(self) -> None:
        """Initialization should set the proper default attributes."""

        self.assertEqual(1., self.evaluator.confidence)
        self.assertEqual(.2, self.evaluator.tolerance)

    def test_confidence_setget(self) -> None:
        """Confidence property getter should return setter argument."""

        confidence = .5
        self.evaluator.confidence = confidence

        self.assertEqual(confidence, self.evaluator.confidence)

    def test_confidence_validation(self) -> None:
        """Confidence property setter should raise ValueError on invalid arguments."""

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
        """Tolerance property getter should return setter argument."""

        tolerance = .5
        self.evaluator.tolerance = tolerance

        self.assertEqual(tolerance, self.evaluator.tolerance)

    def test_tolerance_validation(self) -> None:
        """Confidence margin property setter should raise ValueError on invalid arguments."""

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
        """Plot scale property getter should return setter argument."""

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


class TestObjectMock(object):
    """Mock of a tested object."""

    def __init__(self) -> None:

        self.property_a = None
        self.property_b = 0
        self.property_c = 0

    @dimension
    def property_a(self):
        return self.__property_a

    @property_a.setter(first_impact='init_stage', last_impact='exit_stage')
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
    """An evaluator summing up object properties."""
    
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
        
        return ScalarEvaluationResult(grid, artifacts)


class ProductEvaluator(Evaluator):
    """An evaluator multiplying object properties."""
    
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
        
        return ScalarEvaluationResult(grid, artifacts)

class TestMonteCarloSample(TestCase):
    """Test the Monte Carlo sample class."""

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
        """Initialization arguments should be properly stored as object attributes."""

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


@ray.remote(num_cpus=1)
class MonteCarloActorMock(MonteCarloActor[TestObjectMock]):
    """Mock of a Monte Carlo Actor."""
    
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
    """Test the Monte Carlo actor."""
    
    @classmethod
    def setUpClass(cls) -> None:

        ray.init(local_mode=True, num_cpus=1)
            
    @classmethod
    def tearDownClass(cls):
        
        ray.shutdown()


@ray.remote(num_cpus=1)
class MonteCarloActorMock(MonteCarloActor[TestObjectMock]):
    """Mock of a Monte Carlo Actor."""
    
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
    """Test the Monte Carlo actor."""
    
    @classmethod
    def setUpClass(cls) -> None:

        ray.init(local_mode=True, num_cpus=1, ignore_reinit_error=True)
            
    @classmethod
    def tearDownClass(cls) -> None:

        # Shut down ray 
        ray.shutdown()

    def setUp(self) -> None:
        
        self.investigated_object = TestObjectMock()
        self.investigated_object.property = 1
        self.dimensions = [GridDimension(self.investigated_object, 'property_a', [1, 2, 6, 7, 8])]
        self.evaluators = [SumEvaluator(self.investigated_object), ProductEvaluator(self.investigated_object)]

        self.actor = MonteCarloActorMock.remote((self.investigated_object, self.dimensions, self.evaluators), 0)

    def test_run(self) -> None:
        """Running the actor should produce the expected result"""

        for sample_idx in range(self.dimensions[0].num_sample_points - 1):

            program = [(sample_idx,), (1 + sample_idx,)]

            result: ActorRunResult = ray.get(self.actor.run.remote(program))
            self.assertEqual(2, len(result.samples))


class TestMonteCarloResult(TestCase):
    """Test the result class."""
    
    def setUp(self) -> None:
        ...
        
        
class TestGridDimension(TestCase):
    """Test the simulation grid dimension class."""

    def setUp(self) -> None:
        
        class MockObject(object):
            
            dimension = 11234

        self.considered_object = MockObject()
        self.sample_points = [1, 2, 3, 4]

        self.dimension = GridDimension(self.considered_object, 'dimension', self.sample_points)

    def test_considered_object(self) -> None:
        """Considered object property should return considered object."""

        self.assertIs(self.considered_object, self.dimension.considered_objects[0])

    def test_sample_points(self) -> None:
        """Sample points property should return sample points."""

        self.assertIs(self.sample_points, self.dimension.sample_points)

    def test_num_sample_points(self) -> None:
        """Number of sample points property should return the correct amount of sample points."""

        self.assertEqual(4, self.dimension.num_sample_points)

    def test_configure_point(self) -> None:
        """Configuring a point should set the property correctly."""

        expected_value = self.sample_points[3]
        self.dimension.configure_point(3)

        self.assertEqual(expected_value, self.considered_object.dimension)

    def test_title(self) -> None:
        """Title property should infer the correct title."""

        self.dimension.title = None
        self.assertEqual("dimension", self.dimension.title)

        self.dimension.title = "xyz"
        self.assertEqual("xyz", self.dimension.title)

    def test_plot_scale_setget(self) -> None:
        """Plot scale property getter should return setter argument."""

        scale = 'loglin'
        self.dimension.plot_scale = scale

        self.assertEqual(scale, self.dimension.plot_scale)
        

class TestRegisteredDimension(TestCase):
    """Test the registered dimension"""
    
    def test_decoration(self) -> None:
        """The decorator should return a property registered within the simulation registry"""
        
        expected_first_impact_a = '123124'
        expected_last_impact_a = '21341312'
        
        expected_first_impact_b = '1231223234'
        expected_last_impact_b = '213413123232'
        
        class TestClassA():
            
            def __init__(self) -> None:
                self.__value_a = 1.2345
            
            @dimension
            def test_dimension(self) -> float:
                return self.__value_a
            
            @test_dimension.setter(first_impact=expected_first_impact_a, last_impact=expected_last_impact_a)
            def test_dimension(self, value: float) -> None:
                self.__value_a = value
                
        class TestClassB():
            
            def __init__(self) -> None:
                self.test_dimension = 6.789
                
            @dimension
            def test_dimension(self) -> float:
                return self.__value_b
            
            @test_dimension.setter(first_impact=expected_first_impact_b, last_impact=expected_last_impact_b)
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
    """Test the simulation grid."""
    
    @classmethod
    def setUpClass(cls) -> None:

        ray.init(local_mode=True, num_cpus=1, ignore_reinit_error=True)

    @classmethod
    def tearDownClass(cls) -> None:

        # Shut down ray 
        ray.shutdown()

    def setUp(self) -> None:

        self.investigated_object = TestObjectMock()
        self.evaluators = [ProductEvaluator(self.investigated_object), SumEvaluator(self.investigated_object)]
        self.num_samples = 3
        self.num_actors = 2

        self.monte_carlo = MonteCarlo(investigated_object=self.investigated_object,
                                        evaluators=self.evaluators,
                                        num_samples=self.num_samples,
                                        num_actors=self.num_actors)

    def test_init(self) -> None:
        """Initialization arguments should be properly stored as class attributes."""

        self.assertIs(self.investigated_object, self.monte_carlo.investigated_object)
        self.assertEqual(self.num_samples, self.monte_carlo.num_samples)
        self.assertEqual(self.num_actors, self.monte_carlo.num_actors)

    def test_new_dimension(self) -> None:
        """Test adding a new grid dimension."""

        dimension_str = 'property_a'
        sample_points = [1, 2, 3, 4]

        dimension = self.monte_carlo.new_dimension(dimension_str, sample_points)
        self.assertIs(self.investigated_object, dimension.considered_objects[0])

    def test_add_dimension(self) -> None:
        """Test adding a grid dimension."""

        dimension = Mock()
        self.monte_carlo.add_dimension(dimension)

    def test_add_dimension_validation(self) -> None:
        """Adding an already existing dimension should raise a ValueError."""

        dimension = Mock()
        self.monte_carlo.add_dimension(dimension)

        with self.assertRaises(ValueError):
            self.monte_carlo.add_dimension(dimension)

    def test_num_samples_setget(self) -> None:
        """Number of samples property getter should return setter argument."""

        num_samples = 20
        self.monte_carlo.num_samples = num_samples

        self.assertEqual(num_samples, self.monte_carlo.num_samples)

    def test_num_samples_validation(self) -> None:
        """Number of samples property setter should raise ValueError on arguments smaller than one."""

        with self.assertRaises(ValueError):
            self.monte_carlo.num_samples = 0

        with self.assertRaises(ValueError):
            self.monte_carlo.num_samples = -1

    def test_num_actors_setget(self) -> None:
        """Number of actors property getter should return setter argument."""

        num_actors = 10
        self.monte_carlo.num_actors = num_actors

        self.assertEqual(num_actors, self.monte_carlo.num_actors)

    def test_num_actors_validation(self) -> None:
        """Number of actors property setter should raise ValueError on arguments smaller than one."""

        with self.assertRaises(ValueError):
            self.monte_carlo.num_actors = -1

        with self.assertRaises(ValueError):
            self.monte_carlo.num_actors = 0

#    def test_simulate(self) -> None:
#        """Test the simulation routine."""
#
#        dimensions = {
#            'property_a': [1],
#            'property_b': [1, 2],
#            'property_c': [1, 2, 3],
#        }
#        for dimension, parameters in dimensions.items():
#            self.monte_carlo.new_dimension(dimension, parameters)
#
#        with self.monte_carlo.console.capture():
#            self.monte_carlo.simulate(MonteCarloActorMock)

    def test_section_block_size_setget(self) -> None:
        """Section block size property getter should return setter argument."""

        section_block_size = 10
        self.monte_carlo.section_block_size = section_block_size

        self.assertEqual(section_block_size, self.monte_carlo.section_block_size)

    def test_section_block_size_validation(self) -> None:
        """Section block size property setter should raise ValueError on arguments smaller than one."""

        with self.assertRaises(ValueError):
            self.monte_carlo.section_block_size = -1

        with self.assertRaises(ValueError):
            self.monte_carlo.section_block_size = 0
