# -*- coding: utf-8 -*-
"""Monte Carlo Simulation on Python Ray."""

from __future__ import annotations
import unittest
import warnings
from unittest.mock import Mock
from ray.util import inspect_serializability

import ray

from hermespy.core.monte_carlo import MonteCarlo, MonteCarloActor, MonteCarloSample, \
    Evaluator, ArtifactTemplate, MO, Artifact, GridDimension

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "3.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class EvaluatorMock(Evaluator[Mock]):

    def evaluate(self, investigated_object: MO) -> Artifact:
        return Mock()

    @property
    def abbreviation(self) -> str:
        return "?"

    @property
    def title(self) -> str:
        return "??"


class TestEvaluator(unittest.TestCase):
    """Test base class for all evaluators."""

    def setUp(self) -> None:

        self.evaluator = EvaluatorMock()

    def test_init(self) -> None:
        """Initialization should set the proper default attributes."""

        self.assertEqual(1., self.evaluator.confidence)
        self.assertEqual(0., self.evaluator.tolerance)

    def test_confidence_setget(self) -> None:
        """Confidence property getter should return setter argument."""

        confidence = .5
        self.evaluator.confidence = confidence

        self.assertEqual(confidence, self.evaluator.confidence)

    def test_confidence_level_validation(self) -> None:
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

    def test_plot_scale_setget(self) -> None:
        """Plot scale property getter should return setter argument."""

        plot_scale = 'abce'
        self.evaluator.plot_scale = plot_scale

        self.assertEqual(plot_scale, self.evaluator.plot_scale)


class TestObjectMock(object):
    """Mock of a tested object."""

    def __init__(self) -> None:

        self.property_a = None
        self.property_b = 0
        self.property_c = 0

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


class SumEvaluator(Evaluator[TestObjectMock]):
    """An evaluator summing up object properties."""

    def evaluate(self, investigated_object: TestObjectMock) -> ArtifactTemplate[float]:

        summed = investigated_object.property_a + investigated_object.property_b + investigated_object.property_c
        return ArtifactTemplate[float](summed)

    @property
    def abbreviation(self) -> str:
        return "SUM"

    @property
    def title(self) -> str:
        return "Sum Evaluator"


class ProductEvaluator(Evaluator[TestObjectMock]):
    """An evaluator multiplying object properties."""

    def evaluate(self, investigated_object: TestObjectMock) -> ArtifactTemplate[float]:

        product = investigated_object.property_a * investigated_object.property_b * investigated_object.property_c
        return ArtifactTemplate[float](product)

    @property
    def abbreviation(self) -> str:
        return "Product"

    @property
    def title(self) -> str:
        return "Product Evaluator"


@ray.remote(num_cpus=1)
class MonteCarloActorMock(MonteCarloActor[TestObjectMock]):
    """Mock of a Monte Carlo Actor."""

    def sample(self) -> TestObjectMock:
        return self._investigated_object


class TestMonteCarloSample(unittest.TestCase):
    """Test the Monte Carlo sample class."""

    def setUp(self) -> None:

        self.grid_section = (0, 1, 2, 3)
        self.sample_index = 5
        self.evaluation_artifacts = []

        self.sample = MonteCarloSample(self.grid_section, self.sample_index, self.evaluation_artifacts)

    def test_init(self) -> None:
        """Initialization arguments should be properly stored as object attributes."""

        self.assertCountEqual(self.grid_section, self.sample.grid_section)
        self.assertEqual(self.sample_index, self.sample.sample_index)


# class TestMonteCarloActor(unittest.TestCase):
#     """Test the Monte Carlo actor."""
#
#     def setUp(self) -> None:
#
#         self.investigated_object = TestObjectMock()
#         self.investigated_object.property = 1
#         self.dimensions = [GridDimension(self.investigated_object, 'property_a', [1, 2, 6, 7, 8])]
#         self.evaluators = [SumEvaluator(), ProductEvaluator()]
#
#         self.actor = MonteCarloActorMock.remote((self.investigated_object, self.dimensions, self.evaluators),
#                                                 section_block_size=1)
#
#     def test_run(self) -> None:
#         """Running the actor should produce the expected result."""
#
#
#         for sample_idx in range(self.dimensions[0].num_sample_points):
#
#             expected_grid_section = [sample_idx]
#
#             samples = ray.get(self.actor.run.remote((sample_idx,)))
#             self.assertCountEqual(expected_grid_section, samples[0].grid_section)


class TestGridDimension(unittest.TestCase):
    """Test the simulation grid dimension class."""

    def setUp(self) -> None:

        self.considered_object = Mock()
        self.considered_object.dimension = 11234
        self.sample_points = [1, 2, 3, 4]

        self.dimension = GridDimension(self.considered_object, 'dimension', self.sample_points)

    def test_considered_object(self) -> None:
        """Considered object property should return considered object."""

        self.assertIs(self.considered_object, self.dimension.considered_object)

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


class TestMonteCarlo(unittest.TestCase):
    """Test the simulation grid."""

    def setUp(self) -> None:

        self.investigated_object = TestObjectMock()
        self.evaluators = [ProductEvaluator(), SumEvaluator()]
        self.num_samples = 3
        self.num_actors = 2

        # Required to suppress weird redis warnings
        with warnings.catch_warnings():

            warnings.simplefilter("ignore")
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
        self.assertIs(self.investigated_object, dimension.considered_object)

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
