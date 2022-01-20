# -*- coding: utf-8 -*-
"""Noise model base class."""

from __future__ import annotations
from abc import abstractmethod
from typing import Type, Optional, Union

import numpy as np
import numpy.random as rnd
from ruamel.yaml import ScalarNode, MappingNode, SafeRepresenter, SafeConstructor

from hermespy.core import RandomNode

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.3"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Noise(RandomNode):
    """Implements a complex additive white gaussian noise at the receiver."""

    yaml_tag = u'Noise'

    def __init__(self,
                 seed: Optional[int] = None) -> None:
        """Noise model object initialization.

        Args:

            seed (int, optional):
                Seed used to initialize the pseudo-random number generator.
        """

        RandomNode.__init__(self, seed=seed)

    @abstractmethod
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

        noise = (self._rng.standard_normal(signal.shape) + 1j *
                 self._rng.standard_normal(signal.shape)) / np.sqrt(2) * np.sqrt(noise_power)
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
