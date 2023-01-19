# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np
import ray as ray
from numpy.testing import assert_array_almost_equal

from hermespy.core import dB, Logarithmic, LogarithmicSequence, ValueType
from hermespy.tools import db2lin, lin2db


class TestLogarithmic(TestCase):

    def test_add_logarithmic(self) -> None:
        """Test addition of logarithmic number"""

        summand_a = Logarithmic(10, ValueType.LIN)
        summand_b = Logarithmic(20, ValueType.LIN)

        sum = summand_a + summand_b

        self.assertIsInstance(sum, Logarithmic)
        self.assertEqual(30, sum)
        self.assertAlmostEqual(lin2db(30), sum.value_db)

    def test_add_float(self) -> None:
        """Test addition of floating pint number"""

        summand_a = Logarithmic(10, ValueType.LIN)
        summand_b = 20

        sum_left = summand_a + summand_b
        sum_right = summand_b + summand_a

        self.assertIsInstance(sum_right, float)
        self.assertIsInstance(sum_left, float)
        self.assertEqual(30, sum_right)
        self.assertEqual(30, sum_left)

    def test_substract_logarithmic(self) -> None:
        """Test susbtractopm of logarithmic number"""

        summand_a = Logarithmic(20, ValueType.LIN)
        summand_b = Logarithmic(9, ValueType.LIN)

        sum = summand_a - summand_b

        self.assertIsInstance(sum, Logarithmic)
        self.assertEqual(11, sum)
        self.assertAlmostEqual(lin2db(11), sum.value_db)

    def test_susbtract_float(self) -> None:
        """Test susbtraction of floating point number"""

        summand_a = Logarithmic(30, ValueType.LIN)
        summand_b = 20

        sum_left = summand_a - summand_b
        sum_right = summand_b - summand_a

        self.assertIsInstance(sum_right, float)
        self.assertIsInstance(sum_left, float)
        self.assertEqual(-10, sum_right)
        self.assertEqual(10, sum_left)

    def test_multiplication_logarithmic(self) -> None:
        """Test multiplication with logarithmic number"""

        a = Logarithmic(10., ValueType.LIN)
        b = Logarithmic(20., ValueType.LIN)

        product = a * b
        
        self.assertIsInstance(product, Logarithmic)
        self.assertEqual(200, product)
        self.assertAlmostEqual(lin2db(200), product.value_db)

    def test_multiplication_float(self) -> None:
        """Test multiplication with floating point number"""

        a = Logarithmic(10., ValueType.LIN)
        b = 20.

        product_left = b * a
        product_right = a * b
        
        self.assertIsInstance(product_left, float)
        self.assertIsInstance(product_right, float)
        self.assertEqual(200, product_left)
        self.assertEqual(200, product_right)

    def test_division_logarithmic(self) -> None:
        """Test division with logarithmic number"""

        a = Logarithmic(10., ValueType.LIN)
        b = Logarithmic(20., ValueType.LIN)

        product = a / b
        
        self.assertIsInstance(product, Logarithmic)
        self.assertEqual(.5, product)
        self.assertAlmostEqual(lin2db(.5), product.value_db)

    def test_division_float(self) -> None:
        """Test division with floating point number"""

        a = Logarithmic(10., ValueType.LIN)
        b = 20.

        product_left = b / a
        product_right = a / b
        
        self.assertIsInstance(product_left, float)
        self.assertIsInstance(product_right, float)
        self.assertEqual(2, product_left)
        self.assertEqual(.5, product_right)
        
    def test_string_conversion_integer(self) -> None:
        """Test conversion of integer logarithmics to text"""
        
        i = Logarithmic(10, ValueType.DB)
        
        expected_str = "10dB"
        generated_str = str(i)
        
        self.assertEqual(expected_str, generated_str)
        
    def test_string_conversion_float(self) -> None:
        """Test conversion of floating point logarithmics to text"""
        
        i = Logarithmic(0.12, ValueType.DB)
        
        expected_str = "0.12dB"
        generated_str = str(i)
        
        self.assertEqual(expected_str, generated_str)
        
    def test_representation(self) -> None:
        """Test text representation of object"""
        
        i = Logarithmic(10, ValueType.DB)
        
        expected_str = "<Log 10dB>"
        generated_str = repr(i)
        
        self.assertEqual(expected_str, generated_str)

    def test_shorthand_init(self) -> None:
        """Test dB initialization for scalar logarithmic number"""

        expected_value = 1.234
        log = dB(expected_value)

        self.assertEqual(db2lin(expected_value), log)
        self.assertEqual(expected_value, log.value_db)
        
    def test_tuple_init(self) -> None:
        """Test initialization from tuple"""
        
        linear_expectation = 20.
        logarithmic_expectation = lin2db(20.)
        logarithmic = Logarithmic.From_Tuple(linear_expectation, logarithmic_expectation)
        
        self.assertIsInstance(logarithmic, Logarithmic)
        self.assertEqual(linear_expectation, logarithmic)
        self.assertEqual(logarithmic_expectation, logarithmic.value_db)


class TestLogarithmicSequence(TestCase):

    def setUp(self) -> None:

        self.expected_values = [20, 10, 30]
        self.sequence = LogarithmicSequence(self.expected_values, ValueType.LIN)

    def test_len(self) -> None:
        """Test length calculation"""

        self.assertEqual(len(self.expected_values), len(self.sequence))
        
    def test_tolist(self) -> None:
        """Text explicit list conversion"""
        
        self.assertSequenceEqual(self.expected_values, self.sequence.tolist())
        
    def test_setitem(self) -> None:
        """Test updating a single item within the sequence"""
        
        expected_value = 1234
        
        self.sequence[1] = expected_value
        self.expected_values[1] = expected_value
        
        self.assertEqual(expected_value, self.sequence[1])
        self.assertSequenceEqual(self.expected_values, self.sequence.tolist())

    def test_interaction_scalar(self) -> None:
        """Test interaction with scalar numbers"""
        
        expected_sum = np.array(self.expected_values, dtype=float) + 5.
        left_sum = self.sequence + 5.
        right_sum = 5. + self.sequence
        
        assert_array_almost_equal(expected_sum, left_sum)
        assert_array_almost_equal(expected_sum, right_sum)
        
        expected_product = np.array(self.expected_values, dtype=float) * 5.
        left_product = self.sequence * 5.
        right_product = 5. * self.sequence
        
        assert_array_almost_equal(expected_product, left_product)
        assert_array_almost_equal(expected_product, right_product)
        
    def test_interaction_array(self) -> None:
        """Test interaction with arrays"""
        
        interacting_array = np.arange(len(self.expected_values))
        
        expected_dot_product = np.array(self.expected_values) @ interacting_array
        dot_product = self.sequence @ interacting_array
        
        assert_array_almost_equal(expected_dot_product, dot_product)

    def test_shorthand_init(self) -> None:
        """Test dB initialization for sequence of logarithmic numbers"""

        expected_values = [10., 20., 30.]
        logs_list = dB(expected_values)
        logs_sequence = dB(*expected_values)

        self.assertSequenceEqual(expected_values, [log.value_db for log in logs_list])
        self.assertSequenceEqual(expected_values, [log.value_db for log in logs_sequence])

    def test_cast(self) -> None:
        """Test casting behaviour"""
        
        casted_sequence = np.array(self.expected_values, dtype=float).view(LogarithmicSequence)
        
        assert_array_almost_equal(self.sequence, casted_sequence)
        self.assertEqual(self.expected_values[1], casted_sequence[1])

    def test_ray_serialization(self) -> None:
        """Test serializatin within the Ray framework."""
        
        ray.init(local_mode=True)
        
        sequence_reference = ray.put(self.sequence)
        sequence_copy  = ray.get(sequence_reference)
        
        assert_array_almost_equal(self.sequence, sequence_copy)
        self.assertEqual(self.sequence[1], sequence_copy[1])
