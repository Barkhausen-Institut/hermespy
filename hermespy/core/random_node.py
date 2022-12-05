# -*- coding: utf-8 -*-
"""
============
Random Graph
============
"""

from __future__ import annotations
from typing import Optional

from numpy.random import default_rng, Generator

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


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

        return self.__mother_node._rng

    @_rng.setter
    def _rng(self, value: Generator) -> None:

        self.__generator = value

    @property
    def is_random_root(self) -> bool:
        """Is this random node a root node?

        Return:
            is_root (bool): Boolean root node indicator.
        """

        return self.__generator is not None

    @property
    def seed(self) -> Optional[int]:
        """Random seed of this node.

        Returns: Random seed. `None` if no seed was specified.
        """

        return self.__seed

    @seed.setter
    def seed(self, value: Optional[int]) -> None:

        self.__seed = value
        self.__generator = default_rng(value)

    @property
    def random_mother(self) -> Optional[RandomNode]:
        """The mother node of this random number generator.

        Note that setting the mother node will convert any random node to a child node!

        Returns:

            mother_node (Optional[RandomNode]):
                The mother node. `None` if this node is a root.
        """

        return self.__mother_node

    @random_mother.setter
    def random_mother(self, value: RandomNode) -> None:
        """Set the mother node of this random number generator."""

        self.__generator = None
        self.__mother_node = value
