# -*- coding: utf-8 -*-
"""
===========
Random Node
===========
"""

from __future__ import annotations
from typing import Optional, Type, Union

from numpy.random import default_rng, Generator
from ruamel.yaml import SafeConstructor, SafeRepresenter, ScalarNode, MappingNode

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.3"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RandomNode(object):
    """Random Node within a random dependency graph."""

    # __slots__ = ['__mother_node', '__generator', '__seed']

    yaml_tag = u'RandomNode'
    """YAML serialization tag."""

    __mother_node: Optional[RandomNode]     # Mother node of this node
    __generator: Optional[Generator]        # Numpy generator object
    __seed: Optional[int]                   # Seed used to initialize the pseud-random number generator

    def __init__(self,
                 mother_node: Optional[RandomNode] = None,
                 seed: Optional[int] = None) -> None:
        """
        Args:

            mother_node (RandomNode, optional):
                Mother node of this random node.
                By default, nodes are considered to be roots.

            seed (int, optional):
                Random seed used to initialize the pseudo-random number generator.

        """

        self.__generator = default_rng(seed)
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

    @property
    def is_random_root(self) -> bool:
        """Is this random node a root node?

        Return:
            is_root (bool): Boolean root node indicator.
        """

        return self.__generator is not None

    def set_seed(self, seed: int) -> None:
        """Set an initialization seed for the pseudo-random number generator.

        Note that setting a seed will convert any random node to a base node!

        Args:
            seed (int): Random number seed.
        """

        self.__seed = seed
        self.__generator = default_rng(seed)

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

    @classmethod
    def to_yaml(cls: Type[RandomNode], representer: SafeRepresenter, node: RandomNode) -> ScalarNode:
        """Serialize a `RandomNode` object to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (RandomNode):
                The `RandomNode` instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
                None if the object state is default.
        """

        if node.__seed is not None:

            state = {'seed': node.__seed}
            return representer.represent_mapping(cls.yaml_tag, state)

        return representer.represent_none(None)

    @classmethod
    def from_yaml(cls: Type[RandomNode],
                  constructor: SafeConstructor,
                  node: Union[ScalarNode, MappingNode]) -> RandomNode:
        """Recall a new `RandomNode` instance from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `RandomNode` serialization.

        Returns:
            RandomNode:
                Newly created `RandomNode` instance.
        """

        if isinstance(node, ScalarNode):
            return cls()

        state = constructor.construct_mapping(node)

        # Just mask the seed state if provided
        state['random_generator'] = state.pop('seed', None)
