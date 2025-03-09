# -*- coding utf-8 -*-

from __future__ import annotations
from unittest import TestCase

import numpy as np
from h5py import File
from numpy.testing import assert_array_almost_equal

from hermespy.channel import ConsistentGenerator
from hermespy.channel.consistent import ConsistentRealization, DualConsistentRealization, StaticConsistentRealization, StaticConsistentSample
from unit_tests.core.test_factory import test_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestDualConsistent(TestCase):
    """Test dual consistent random processes."""

    def setUp(self) -> None:

        self.rng = np.random.default_rng(42)
        self.decorrelation_distance = 30.
        self.generator = ConsistentGenerator(self.rng)

        self.variable_dimensions = (512, 2, 2)
        self.gaussian_variable = self.generator.gaussian(self.variable_dimensions)
        self.uniform_variable = self.generator.uniform(self.variable_dimensions)
        self.boolean_variable = self.generator.boolean(self.variable_dimensions)

        self.realiztion = self.generator.realize(self.decorrelation_distance)

    def test_consistency(self) -> None:
        """Sampling a variable at the same position should yield identical random realizations"""

        position_a = np.array([1, 2, 3])
        position_b = np.array([10, 11, 12])
        sample = self.realiztion.sample(position_a, position_b)

        gaussian_a = self.gaussian_variable.sample(sample, 2.34, 5.67)
        gaussian_b = self.gaussian_variable.sample(sample, 2.34, 5.67)
        assert_array_almost_equal(gaussian_a, gaussian_b)

        uniform_a = self.uniform_variable.sample(sample)
        uniform_b = self.uniform_variable.sample(sample)
        assert_array_almost_equal(uniform_a, uniform_b)

        boolean_a = self.boolean_variable.sample(sample)
        boolean_b = self.boolean_variable.sample(sample)
        assert_array_almost_equal(boolean_a, boolean_b)

    def test_spatial_correlation(self) -> None:
        """Test the spatial correlation of the Gaussian process"""

        position_a_a = np.array([1, 2, 3])
        position_a_b = np.array([10, 11, 12])
        position_b_a = np.array([2, 3, 4])
        position_b_b = np.array([11, 12, 13])
        displacement_a = position_a_a - position_b_a
        displacement_b = position_a_b - position_b_b

        # Equation 1 in " spatially consistent Gaussian process for dual
        # mobility in the three-dimensional space"
        expected_cross_correlation = (
            np.exp(-np.linalg.norm(displacement_a) / self.decorrelation_distance) *
            np.exp(-np.linalg.norm(displacement_b) / self.decorrelation_distance)
        )

        sample_a = self.realiztion.sample(position_a_a, position_a_b)
        sample_b = self.realiztion.sample(position_b_a, position_b_b)
        gaussian_a = self.gaussian_variable.sample(sample_a, 2.34, 5.67)
        gaussian_b = self.gaussian_variable.sample(sample_b, 2.34, 5.67)

        correlation = np.corrcoef(gaussian_a.flatten(), gaussian_b.flatten())[0, 1]
        self.assertAlmostEqual(expected_cross_correlation, correlation, delta=0.1)

    def test_gaussian_statistics(self) -> None:
        """Test the statistics of the represented Gaussian process"""

        position_a = np.array([1, 2, 3])
        position_b = np.array([10, 11, 12])
        sample = self.realiztion.sample(position_a, position_b)

        expected_mean = 2.34
        expected_standard_deviation = 5.67
        gaussian = self.gaussian_variable.sample(sample, expected_mean, expected_standard_deviation)

        self.assertAlmostEqual(expected_mean, gaussian.mean(), delta=0.1)
        self.assertAlmostEqual(expected_standard_deviation, gaussian.var()**.5, delta=0.2)

    def test_uniform_statistics(self) -> None:
        """Test the statistics of the represented uniform process"""

        position_a = np.array([1, 2, 3])
        position_b = np.array([10, 11, 12])
        sample = self.realiztion.sample(position_a, position_b)

        uniform = self.uniform_variable.sample(sample)

        self.assertAlmostEqual(0.5, uniform.mean(), delta=0.1)
        self.assertAlmostEqual(1./12, uniform.var(), delta=0.1)

    def test_boolean_statistics(self) -> None:
        """Test the statistics of the represented boolean process"""

        position_a = np.array([1, 2, 3])
        position_b = np.array([10, 11, 12])
        sample = self.realiztion.sample(position_a, position_b)

        boolean = self.boolean_variable.sample(sample)

        self.assertAlmostEqual(0.5, boolean.mean(), delta=0.1)
        self.assertAlmostEqual(0.25, boolean.var(), delta=0.1)

    def test_variable_serialization(self) -> None:
        """Test serialization of consistent variables"""
        
        for var in [self.gaussian_variable, self.uniform_variable, self.boolean_variable]:
            with self.subTest(var.__class__.__name__):
                test_roundtrip_serialization(self, var)


class _TestConsistentRealization(object):
    """Test the consistent realization base class"""
    
    realization: ConsistentRealization
    
    def test_serialization(self) -> None:
        """Test serialization of consistent realizations"""
        
        test_roundtrip_serialization(self, self.realization)


class TestDualConsistentRealization(TestCase, _TestConsistentRealization):
    """Test realization of dual consistent random processes."""

    def setUp(self) -> None:

        rng = np.random.default_rng(42)
        num_variables = 15
        self.phases = rng.normal(size=num_variables)
        self.frequencies = rng.normal(size=(3, num_variables, 2))

        self.realization = DualConsistentRealization(self.frequencies, self.phases)


class TestStaticConstentSample(TestCase):
    """Test the static realization of a spatially invariant random process."""
    
    def setUp(self) -> None:
        
        self.scalar_samples = np.random.normal(size=15)
        self.sample = StaticConsistentSample(self.scalar_samples)

    def test_fetch_scalars(self) -> None:
        """Fetching scalars should return the correct values"""
        
        scalars = self.sample.fetch_scalars(0, 15)
        assert_array_almost_equal(self.scalar_samples, scalars)


class TestStaticConsistentRealization(TestCase, _TestConsistentRealization):
    """Test the realization of a spatially invariant random process."""

    def setUp(self) -> None:

        rng = np.random.default_rng(42)
        num_variables = 15
        self.scalar_samples = rng.normal(size=num_variables)

        self.realization = StaticConsistentRealization(self.scalar_samples)
