from __future__ import annotations
from ruamel.yaml import SafeConstructor, Node
from ruamel.yaml.comments import CommentedOrderedMap
from typing import Type, List, TYPE_CHECKING

from modem import Modem
from source import BitsSource

if TYPE_CHECKING:
    from scenario import Scenario


class Transmitter(Modem):

    yaml_tag = 'Transmitter'

    def __init__(self, scenario: Scenario, **kwargs) -> None:

        Modem.__init__(self, scenario, **kwargs)

    @classmethod
    def from_yaml(cls: Type[Transmitter], constructor: SafeConstructor, node: Node) -> Transmitter:

        scenario = [scene for node, scene in constructor.constructed_objects.items() if node.tag == 'Scenario'][0]

        state = constructor.construct_mapping(node, deep=True)
        bits_source = state.pop(BitsSource.yaml_tag, None)

        args = dict((k.lower(), v) for k, v in state.items())
        transmitter = Transmitter(scenario, **args)
        yield transmitter

        if bits_source is not None:
            transmitter.bits_source = bits_source

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
