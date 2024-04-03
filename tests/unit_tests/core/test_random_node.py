# -*- coding: utf-8 -*-

from sys import maxsize
from unittest import TestCase
from unittest.mock import Mock

from numpy.random import Generator

from hermespy.core.random_node import RandomNode, RandomRealization


__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestRandomRealization(TestCase):
    """Test a single randim realization"""

    def setUp(self) -> None:
        self.node = RandomNode(seed=42)
        self.realization = RandomRealization(self.node)

    def test_seed(self) -> None:
        """The seed property should return the correct seed"""

        self.assertEqual(RandomRealization(RandomNode(seed=42)).seed, self.realization.seed)

    def test_generator(self) -> None:
        """The generator property should provide a reproducible generator"""

        test_generator = RandomRealization(RandomNode(seed=42)).generator()
        generator = self.realization.generator()

        self.assertEqual(test_generator.integers(maxsize), generator.integers(maxsize))


class TestRandomNode(TestCase):
    """Test a single Random Node."""

    def setUp(self) -> None:
        self.seed = 1234
        self.node = RandomNode(seed=self.seed)

    def test_rng(self) -> None:
        """The Random Number Generator property should point to the correct generator."""

        self.assertIs(self.node._RandomNode__generator, self.node._rng)

        mother_node = RandomNode()
        self.node.random_mother = mother_node
        self.assertIs(mother_node._RandomNode__generator, self.node._rng)

    def test_rng_setget(self) -> None:
        """The Random Number Generator property getter should return the setter argument"""

        expected_rng = Mock()
        self.node._rng = expected_rng

        self.assertIs(expected_rng, self.node._rng)

    def test_random_root(self) -> None:
        """Random root property should correctly report if the nood is a root."""

        self.assertTrue(self.node.is_random_root)

        self.node.random_mother = RandomNode()
        self.assertFalse(self.node.is_random_root)

    def test_set_seed(self) -> None:
        """Setting a seed should result in reproductible random number generation."""

        first_number = self.node._rng.normal()

        self.node.seed = self.seed
        second_number = self.node._rng.normal()

        self.assertEqual(first_number, second_number)

    def test_random_mother_setget(self) -> None:
        """Random mother property getter should return setter argument."""

        self.assertIsNone(self.node.random_mother)

        random_mother = Mock()
        self.node.random_mother = random_mother

        self.assertIs(random_mother, self.node.random_mother)


class TestRandomTree(TestCase):
    """Test random nodes within a tree configuration."""

    def setUp(self) -> None:
        self.seed = 12345643123
        self.root_node = RandomNode(seed=self.seed)

        self.child_nodes = [RandomNode() for _ in range(10)]

        self.child_nodes[0].random_mother = self.root_node
        self.child_nodes[1].random_mother = self.root_node

        for n in range(2, 10):
            self.child_nodes[n].random_mother = self.child_nodes[n - 2]

    def test_set_seed(self) -> None:
        """Setting a seed should result in reproductible random number generation."""

        first_numbers = [child._rng.normal() for child in self.child_nodes]

        self.root_node.seed = self.seed
        second_numbers = [child._rng.normal() for child in self.child_nodes]

        self.assertCountEqual(first_numbers, second_numbers)
