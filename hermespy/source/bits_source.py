# -*- coding: utf-8 -*-
"""Source of bit streams to be transmitted."""

from __future__ import annotations
from typing import TYPE_CHECKING, Type, Optional, Union
from collections import namedtuple
from ruamel.yaml import SafeConstructor, SafeRepresenter, ScalarNode, MappingNode
from numpy.typing import ArrayLike
import numpy as np
import numpy.random as rnd

if TYPE_CHECKING:
    from hermespy.modem import Transmitter

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.4"
__maintainer__ = "Tobias Kronauer"
__email__ = "tobias.kronauer@barkhauseninstitut.org"
__status__ = "Prototype"


ErrorStats = namedtuple(
    'ErrorStats',
    'number_of_bits number_of_bit_errors number_of_blocks number_of_block_errors')


class BitsSource:
    """Implements a random bit source, with calculation of error statistics."""

    yaml_tag = "Bits"
    __random_generator: Optional[rnd.Generator]

    def __init__(self, 
                 transmitter: Optional[Transmitter] = None,
                 random_generator: Union[rnd.Generator, ArrayLike, None] = None) -> None:
        """BitsSource object initialization.

        Args:
        
            transmitter (Scenario, optional):
                The transmitter this bits source belongs to.
        
            random_generator (Union[numpy.random.Generator, ArrayLike], optional):
                State of the underlying random generator.
        """

        self.__transmitter = None

        self.transmitter = transmitter
        self.random_generator = random_generator

    @property
    def transmitter(self) -> Transmitter:
        """Access the transmitter this bit source is attached to.

        Returns:
            Transmitter:
                The referenced transmitter.

        Raises:
            RuntimeError: If the bit source is currently floating.
        """

        if self.__transmitter is None:
            raise RuntimeError("Error trying to access the transmitter of a floating bit source")

        return self.__transmitter

    @transmitter.setter
    def transmitter(self, transmitter: Transmitter) -> None:
        """Attach the bit source to a specific transmitter.

        This can only be done once to a floating bit source.

        Args:
            transmitter (Transmitter): The transmitter this bit source should be attached to.

        Raises:
            RuntimeError: If the bit source is already attached to a transmitter.
        """

        if self.__transmitter is not None:
            raise RuntimeError("Error trying to modify the transmitter of an already attached bit source")

        self.__transmitter = transmitter
        
    @property
    def random_generator(self) -> rnd.Generator:
        """Access the random number generator assigned to this bit source.

        This property will return the scenarios random generator if no random generator has been specifically set.

        Returns:
            numpy.random.Generator: The random generator.

        Raises:
            RuntimeError: If trying to access the random generator of a floating bit source.
        """

        if self.__transmitter is None:
            raise RuntimeError("Trying to access the random generator of a floating bit source")

        if self.__random_generator is None:
            return self.__transmitter.random_generator

        return self.__random_generator

    @random_generator.setter
    def random_generator(self, generator: Union[rnd.Generator, ArrayLike, None]) -> None:
        """Modify the configured random number generator assigned to this bit source.

        Args:
            generator (Union[rnd.Generator, ArrayLike, None]):
                The random generator. None if not specified.
        """

        if generator is None:
            self.__random_generator = None

        elif isinstance(generator, rnd.Generator):
            self.__random_generator = generator

        # Try to initialize the random generator from an array-like seed
        else:

            # Convert strings to integer sequences by default, using them as generator seeds
            if isinstance(generator, str):
                self.__random_generator = rnd.default_rng([ord(c) for c in list(generator)])

            else:
                self.__random_generator = rnd.default_rng(generator)

    def get_bits(self, number_of_bits: int) -> np.ndarray:
        """Returns a vector of generated bits.

        Args:
            number_of_bits (int): Number of bits to be generated.

        Returns:
            np.ndarray: A vector containing `number_of_bits` generated bits.
        """

        return self.random_generator.integers(0, 2, size=number_of_bits, dtype=int)

    @classmethod
    def to_yaml(cls: Type[BitsSource], representer: SafeRepresenter, node: BitsSource) -> ScalarNode:
        """Serialize a `BitsSource` object to YAML.

        Currently a stub.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (BitsSource):
                The `BitsSource` instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
                None if the object state is default.
        """

        return representer.represent_none(None)

    @classmethod
    def from_yaml(cls: Type[BitsSource],
                  constructor: SafeConstructor,
                  node: Union[ScalarNode, MappingNode]) -> BitsSource:
        """Recall a new `BitsSource` instance from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `BitsSource` serialization.

        Returns:
            BitsSource:
                Newly created `BitsSource` instance.
        """

        if isinstance(node, ScalarNode):
            return cls()

        state = constructor.construct_mapping(node)

        # Just mask the seed state if provided
        state['random_generator'] = state.pop('seed', None)
        return cls(**state)
