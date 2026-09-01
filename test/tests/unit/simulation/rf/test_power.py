# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from h5py import File
from hermespy.simulation import NoDCPowerModel, ConstantDCPowerModel, SampledDCPowerModel
from hermespy.core import Factory
from ...core.test_factory import test_roundtrip_serialization

__author__ = "Emre Can Atas"
__copyright__ = "Copyright 2026, Barkhausen Institut gGmbH"
__credits__ = ["Emre Can Atas", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.6.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestNoDCPowerModel(TestCase):
    """Test class for the placeholder power consumption model."""

    def setUp(self) -> None:
        self.model = NoDCPowerModel()

    def test_get_power(self) -> None:
        """No power should be consumed independently of the input signal"""

        signal = np.array([0.0 + 0j, 1.0 + 0j, 10.0 + 0j])
        assert_array_equal(np.zeros(3), self.model.get_power(signal))

    def test_get_power_shape(self) -> None:
        """Consumed power should be returned with the shape of the input signal"""

        signal = np.ones((2, 3), dtype=np.complex128)
        self.assertSequenceEqual((2, 3), self.model.get_power(signal).shape)

    def test_serialization(self) -> None:
        """Test power model serialization"""

        test_roundtrip_serialization(self, self.model)


class TestConstantDCPowerModel(TestCase):
    """Test class for the constant power consumption model."""

    def setUp(self) -> None:
        self.model = ConstantDCPowerModel(2.5)

    def test_power_setget(self) -> None:
        """Power property getter should return setter argument"""

        expected_power = 1.2345
        self.model.power = expected_power
        self.assertEqual(expected_power, self.model.power)

    def test_power_validation(self) -> None:
        """Power should be non-negative"""

        with self.assertRaises(ValueError):
            self.model.power = -1.0

    def test_get_power(self) -> None:
        """Consumed power should be constant independently of the input signal"""

        signal = np.array([0.0 + 0j, 1.0 + 0j, 10.0 + 0j])
        assert_array_equal(np.full(3, 2.5), self.model.get_power(signal))

    def test_get_power_shape(self) -> None:
        """Consumed power should be returned with the shape of the input signal"""

        signal = np.ones((2, 3), dtype=np.complex128)
        self.assertSequenceEqual((2, 3), self.model.get_power(signal).shape)

    def test_serialization(self) -> None:
        """Test power model serialization"""

        test_roundtrip_serialization(self, self.model)


class TestSampledDCPowerModel(TestCase):
    """Test class for the sampled power consumption model."""

    def setUp(self) -> None:
        self.model = SampledDCPowerModel([0.0, 1.0, 2.0], [1.0, 3.0, 4.0])

    def test_init_validation(self) -> None:
        """Initialization should raise ValueErrors on invalid arguments"""

        with self.assertRaises(ValueError):
            SampledDCPowerModel([0.0, 1.0], [1.0])

        with self.assertRaises(ValueError):
            SampledDCPowerModel([0.0], [1.0])

        with self.assertRaises(ValueError):
            SampledDCPowerModel([1.0, 0.0], [1.0, 2.0])

        with self.assertRaises(ValueError):
            SampledDCPowerModel([1.0, 1.0], [1.0, 2.0])

        with self.assertRaises(ValueError):
            SampledDCPowerModel([0.0, 1.0], [1.0, -2.0])

    def test_init_flatten(self) -> None:
        """Initialization should accept column vectors"""

        model = SampledDCPowerModel(np.array([[0.0], [1.0]]), np.array([[1.0], [3.0]]))
        assert_array_equal(np.array([0.0, 1.0]), model.input_powers)

    def test_sampling_points(self) -> None:
        """Sampling point properties should return the initialization arguments"""
        assert_array_equal(np.array([0.0, 1.0, 2.0]), self.model.input_powers)
        assert_array_equal(np.array([1.0, 3.0, 4.0]), self.model.consumed_powers)

    def test_get_power(self) -> None:
        """Consumed power should be interpolated between the sampling points"""

        signal = np.array([0.5 + 0j, 1.0 + 0j, 10.0 + 0j])
        assert_array_almost_equal(np.array([1.5, 3.0, 4.0]), self.model.get_power(signal))

    def test_get_power_shape(self) -> None:
        """Consumed power should be returned with the shape of the input signal"""

        signal = np.ones((2, 3), dtype=np.complex128)
        self.assertSequenceEqual((2, 3), self.model.get_power(signal).shape)

    def test_serialization(self) -> None:
        """Sampling points should survive a serialization roundtrip"""

        file = File("test.h5", "w", driver="core", backing_store=False)
        Factory().to_HDF(file, self.model)
        deserialization = Factory().from_HDF(file, SampledDCPowerModel)
        file.close()

        assert_array_equal(self.model.input_powers, deserialization.input_powers)
        assert_array_equal(self.model.consumed_powers, deserialization.consumed_powers)
