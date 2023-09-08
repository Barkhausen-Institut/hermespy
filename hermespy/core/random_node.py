# -*- coding: utf-8 -*-
"""
============
Random Graph
============
"""

from __future__ import annotations
from sys import maxsize
from typing import Optional

from numpy.random import default_rng, Generator

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RandomRealization(object):
    """Realization of a random node."""

    __seed: int

    def __init__(self, random_node: RandomNode) -> None:
        """
        Args:
            random_node (RandomNode): Random node from which to generate a realization.
        """

        # Draw a random signed integer from the node's random number generator
        self.__seed = random_node._rng.integers(0, maxsize)

    @property
    def seed(self) -> int:
        """Seed of the random realization.

        Returns: A signed integer representing the random seed.
        """

        return self.__seed

    def generator(self) -> Generator:
        """Initialize a new generator from the realized random seed.

        Returns: A new numpy generator object.
        """

        return default_rng(self.__seed)


class RandomNode(object):
    """Random Node within a random dependency graph."""

    __mother_node: Optional[RandomNode]  # Mother node of this node
    __generator: Optional[Generator]  # Numpy generator object
    __seed: Optional[int]

    def __init__(self, mother_node: Optional[RandomNode] = None, seed: Optional[int] = None) -> None:
        """
        Args:

            mother_node (RandomNode, optional):
                Mother node of this random node.
                By default, nodes are considered to be roots.

            seed (int, optional):
                Random seed used to initialize the pseudo-random number generator.
        """

        self.seed = seed
        self.__mother_node = mother_node

    @property
    def _rng(self) -> Generator:
        """Random number generator.

        If this random node is a root node, it will return this node's generator.
        Otherwise, the generator of the mother's node will be returned.

        Returns:
            numpy.random.Generator: The random number generator of this node.
        """

        if self.is_random_root:
            return self.__generator

        return self.random_mother._rng

    @_rng.setter
    def _rng(self, value: Generator) -> None:
        self.__generator = value

    @property
    def is_random_root(self) -> bool:
        """Is this random node a root node?

        :meta private:
        """

        return self.__generator is not None

    @property
    def seed(self) -> int | None:
        """Random seed of this node.

        :obj:`None` if no seed has been set.

        :meta private:
        """

        return self.__seed

    @seed.setter
    def seed(self, value: int) -> None:
        self.__seed = value
        self.__generator = default_rng(value)

    @property
    def random_mother(self) -> RandomNode | None:
        """The mother node of this random number generator.

        :obj:`None` if this random node is considered a root node.
        Note that setting the mother node will convert any random node to a child node!

        :meta private:
        """

        return self.__mother_node

    @random_mother.setter
    def random_mother(self, value: RandomNode) -> None:
        """Set the mother node of this random number generator."""

        self.__generator = default_rng(self.seed) if value is None else None
        self.__mother_node = value
