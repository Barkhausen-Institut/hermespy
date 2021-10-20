from __future__ import annotations
from ruamel.yaml import RoundTripConstructor, Node
from ruamel.yaml.comments import CommentedOrderedMap
from typing import Type, List, TYPE_CHECKING
import numpy as np

from source import BitsSource
from modem import Modem
from modem.waveform_generator import WaveformGenerator


class Receiver(Modem):

    yaml_tag = 'Receiver'

    def __init__(self, **kwargs) -> None:
        Modem.__init__(self, **kwargs)

    @classmethod
    def from_yaml(cls: Type[Receiver], constructor: RoundTripConstructor, node: Node) -> Receiver:

        state = constructor.construct_mapping(node, CommentedOrderedMap)

        waveform_generator = None
        bits_source = None

        for key in state.keys():
            if key.startswith(WaveformGenerator.yaml_tag):
                waveform_generator = state.pop(key)
                break

        for key in state.keys():
            if key.startswith(BitsSource.yaml_tag):
                bits_source = state.pop(key)
                break

        state[WaveformGenerator.yaml_tag] = waveform_generator
        state[BitsSource.yaml_tag] = bits_source

        args = dict((k.lower(), v) for k, v in state.items())

        position = args.pop('position', None)
        if position is not None:
            args['position'] = np.array(position)

        orientation = args.pop('orientation', None)
        if position is not None:
            args['orientation'] = np.array(orientation)

        return Receiver(**args)

    @property
    def index(self) -> int:
        """The index of this receiver in the scenario.

        Returns:
            int:
                The index.
        """

        return self.scenario.receivers.index(self)

    @property
    def paired_modems(self) -> List[Modem]:
        """The modems connected to this modem over an active channel.

        Returns:
            List[Modem]:
                A list of paired modems.
        """

        return [channel.receiver for channel in self.scenario.departing_channels(self, True)]
