# -*- coding: utf-8 -*-
"""Noise model base class."""

from __future__ import annotations
import numpy as np
import numpy.random as rnd
from typing import TYPE_CHECKING, Type, Optional, Union
from ruamel.yaml import ScalarNode, MappingNode, SafeRepresenter, SafeConstructor

if TYPE_CHECKING:
    from hermespy.modem import Receiver

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.0"
__maintainer__ = "Tobias Kronauer"
__email__ = "tobias.kronaue@barkhauseninstitut.org"
__status__ = "Prototype"


class Noise:
    """Implements a complex additive white gaussian noise at the receiver.

    Attributes:

        __receiver (Optional[Receiver]):
            Receiver modem this noise model is attached to.
            None if the noise model is currently floating.

        __random_generator (Optional[numpy.random.Generator]):
            Random number generator.
            If not set (i.e. None), the random generator instance of the attached receiver will be called.
    """

    yaml_tag = u'Noise'
    __receiver: Optional[Receiver]
    __random_generator: Optional[rnd.Generator]

    def __init__(self,
                 random_generator: Optional[rnd.Generator] = None) -> None:
        """Noise model object initialization.

        Args:
            random_generator (rnd.Generator, optional): Random number generator.
        """

        self.__receiver = None
        self.__random_generator = random_generator

    @property
    def receiver(self) -> Receiver:
        """Access the receiver modem this noise model is attached to.

        Returns:
            Receiver: Handle to the receiver modem.

        Raises:
            RuntimeError: If the noise model is currently floating.
        """

        if self.__receiver is None:
            raise RuntimeError("Error trying to access the receiver of a floating noise model")

        return self.__receiver

    @receiver.setter
    def receiver(self, new_receiver: Optional[Receiver]) -> None:
        """Configure the receiver modem this noise model is attached to.

        Args:
            new_receiver (Optional[Receiver]): New receiver modem.

        Raises:
            RuntimeError: If the noise model is already attached to a receiver
        """

        if self.__receiver is not new_receiver:

            if self.__receiver is not None:
                raise RuntimeError("Error trying to re-configure the receiver of an attached noise model")

            self.__receiver = new_receiver
            new_receiver.noise = self
            
    @property
    def random_generator(self) -> rnd.Generator:
        """Access the random number generator assigned to this noise model.

        This property will return the scenarios random generator if no random generator has been specifically set.

        Returns:
            numpy.random.Generator: The random generator.

        Raises:
            RuntimeError: If trying to access the random generator of a floating noise model.
        """

        if self.__random_generator is not None:
            return self.__random_generator

        if self.__receiver is None:
            raise RuntimeError("Trying to access the random generator of a floating noise model")

        return self.__receiver.random_generator

    @random_generator.setter
    def random_generator(self, generator: Optional[rnd.Generator]) -> None:
        """Modify the configured random number generator assigned to this noise model.

        Args:
            generator (Optional[numpy.random.generator]): The random generator. None if not specified.
        """

        self.__random_generator = generator

    def add_noise(self, signal: np.ndarray, noise_power: float) -> np.ndarray:
        """Adds noise to a signal.

        Args:
            signal (np.ndarray):
                Input signal, rows denoting antenna, columns denoting samples.

            noise_power: Power of the additive noise.

        Returns:
            np.ndarray: Noisy signal.

        Raises:
            ValueError: If noise power is negative.
        """

        if noise_power < 0.0:
            raise ValueError("Noise power must be non-negative")

        if noise_power == 0.0:
            return signal

        noise = (self.random_generator.standard_normal(signal.shape) + 1j *
                 self.random_generator.standard_normal(signal.shape)) / np.sqrt(2) * np.sqrt(noise_power)
        return signal + noise

    @classmethod
    def to_yaml(cls: Type[Noise],
                representer: SafeRepresenter,
                node: Noise) -> ScalarNode:
        """Serialize an `Noise` object to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (Noise):
                The `Noise` instance to be serialized.

        Returns:
            Node:
                The serialized YAML node
        """

        return representer.represent_scalar(cls.yaml_tag, None)

    @classmethod
    def from_yaml(cls: Type[Noise],
                  constructor: SafeConstructor,
                  node: Union[ScalarNode, MappingNode]) -> Noise:
        """Recall a new `Noise` instance from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Union[ScalarNode, MappingNode]):
                YAML node representing the `Noise` serialization.

        Returns:
            WaveformGenerator:
                Newly created `Noise` instance.
        """

        if isinstance(node, ScalarNode):
            return cls()

        state = constructor.construct_mapping(node)

        # Convert the random seed to a new random generator object if its specified within the config
        random_seed = state.pop('random_seed', None)
        if random_seed is not None:
            state['random_generator'] = rnd.default_rng(random_seed)

        return cls(**state)
