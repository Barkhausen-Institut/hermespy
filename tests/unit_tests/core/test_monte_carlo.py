# -*- coding: utf-8 -*-
"""Monte Carlo Simulation on Python Ray."""

import unittest
import warnings

from hermespy.core.monte_carlo import MonteCarlo, MonteCarloActor, MonteCarloSample,\
    Evaluator, ArtifactTemplate

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.3"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


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

        sum = investigated_object.property_a + investigated_object.property_b + investigated_object.property_c
        return ArtifactTemplate[float](sum)

    def __str__(self) -> str:

        return "Sum"


class ProductEvaluator(Evaluator[TestObjectMock]):
    """An evaluator multiplying object properties."""

    def evaluate(self, investigated_object: TestObjectMock) -> ArtifactTemplate[float]:

        product = investigated_object.property_a * investigated_object.property_b * investigated_object.property_c
        return ArtifactTemplate[float](product)

    def __str__(self) -> str:

        return "Product"


class MonteCarloActorMock(MonteCarloActor[TestObjectMock]):
    """Mock of a Monte Carlo Actor."""

    @staticmethod
    def sample(investigated_object: TestObjectMock):
        return investigated_object.some_operation()


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


class TestMonteCarloActor(unittest.TestCase):
    """Test the Monte Carlo actor."""

    def setUp(self) -> None:

        self.investigated_object = TestObjectMock()
        self.investigated_object.property = 1
        self.dimensions = {'property_a': [1, 2, 6, 7, 8]}
        self.evaluators = set()

        self.actor = MonteCarloActorMock(investigated_object=self.investigated_object,
                                         dimensions=self.dimensions, evaluators=self.evaluators)

    def test_run(self) -> None:
        """Running the actor should produce the expected result."""

        for sample_idx, sample_value in enumerate(self.dimensions['property_a']):

            expected_grid_section = [sample_idx]
            expected_sample = 2 * sample_value

            sample = self.actor.run([sample_idx], sample_idx)
            self.assertCountEqual(expected_grid_section, sample.grid_section)
            #self.assertEqual(expected_sample, sample)


class TestMonteCarlo(unittest.TestCase):
    """Test the simulation grid."""

    def setUp(self) -> None:

        self.investigated_object = TestObjectMock()
        self.evaluators = {ProductEvaluator(), SumEvaluator()}
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

    def test_add_dimension(self) -> None:
        """Test adding a simulation grid dimension by string representation."""

        dimension_str = 'property_a'
        sample_points = [1, 2, 3, 4]

        self.monte_carlo.add_dimension(dimension_str, sample_points)

    def test_add_dimension_validation(self) -> None:
        """Adding a non-existent simulation dimension should raise a ValueError."""

        with self.assertRaises(ValueError):
            self.monte_carlo.add_dimension('xxx', [1, 2, 3])

        with self.assertRaises(ValueError):
            self.monte_carlo.add_dimension('property', [])

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

    def test_simulate(self) -> None:
        """Test the simulation routine."""

        dimensions = {
            'property_a': [1],
            'property_b': [1, 2],
            'property_c': [1, 2, 3],
        }
        for dimension, parameters in dimensions.items():
            self.monte_carlo.add_dimension(dimension, parameters)

        self.monte_carlo.simulate(MonteCarloActorMock)
