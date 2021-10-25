# -*- coding: utf-8 -*-
"""HermesPy transmitting modem."""

from __future__ import annotations
from ruamel.yaml import SafeConstructor, Node, MappingNode, ScalarNode
from typing import Type, List, TYPE_CHECKING, Any
import numpy as np
import numpy.random as rnd

from modem import Modem
from source import BitsSource
from modem.waveform_generator import WaveformGenerator

if TYPE_CHECKING:
    from scenario import Scenario

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Transmitter(Modem):

    yaml_tag = 'Transmitter'

    def __init__(self, **kwargs: Any) -> None:
        """Object initialization.

        Args:
            **kwargs (Any): Transmitter configuration.
        """

        Modem.__init__(self, **kwargs)

    @classmethod
    def from_yaml(cls: Type[Transmitter], constructor: SafeConstructor, node: Node) -> Transmitter:
        """Recall a new `Transmitter` instance from YAML.

        Args:
            constructor (RoundTripConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `Transmitter` serialization.

        Returns:
            Transmitter:
                Newly created `Transmitter` instance.

        Raises:
            RuntimeError: If `node` is neither a scalar or a map.
        """

        # If the transmitter is not a map, create a default object and warn the user
        if not isinstance(node, MappingNode):

            if isinstance(node, ScalarNode):
                return Transmitter()

            else:
                raise RuntimeError("Transmitters must be configured as YAML maps")

        constructor.add_multi_constructor(WaveformGenerator.yaml_tag, WaveformGenerator.from_yaml)
        state = constructor.construct_mapping(node, deep=True)

        bits_source = state.pop(BitsSource.yaml_tag, None)

        waveform_generator = None
        for key in state.keys():
            if key.startswith(WaveformGenerator.yaml_tag):
                waveform_generator = state.pop(key)
                break

        args = dict((k.lower(), v) for k, v in state.items())

        position = args.pop('position', None)
        if position is not None:
            args['position'] = np.array(position)

        orientation = args.pop('orientation', None)
        if position is not None:
            args['orientation'] = np.array(orientation)


        # Convert the random seed to a new random generator object if its specified within the config
        random_seed = args.pop('random_seed', None)
        if random_seed is not None:
            args['random_generator'] = rnd.default_rng(random_seed)

        transmitter = Transmitter(**args)

        if bits_source is not None:
            transmitter.bits_source = bits_source

        if waveform_generator is not None:
            transmitter.waveform_generator = waveform_generator

        return transmitter

    @property
    def index(self) -> int:
        """The index of this transmitter in the scenario.

        Returns:
            int:
                The index.
        """

        return self.scenario.transmitters.index(self)

    @property
    def paired_modems(self) -> List[Modem]:
        """The modems connected to this modem over an active channel.

        Returns:
            List[Modem]:
                A list of paired modems.
        """

        return [channel.receiver for channel in self.scenario.departing_channels(self, True)]

    def generate_data_bits(self) -> np.ndarray:
        """Generate data bits required to build a single transmit data frame for this modem.

        Returns:
            numpy.ndarray: A vector of hard data bits in 0/1 format.
        """

        return self.random_generator.integers(0, 2, self.num_data_bits_per_frame)
