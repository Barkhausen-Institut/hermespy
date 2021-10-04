from __future__ import annotations
from ruamel.yaml import RoundTripConstructor, Node
from ruamel.yaml.comments import CommentedOrderedMap
from typing import Type, List, TYPE_CHECKING

from modem import Modem
from source import BitsSource

if TYPE_CHECKING:
    from scenario import Scenario


class Receiver(Modem):

    yaml_tag = 'Receiver'

    def __init__(self, scenario: Scenario, **kwargs) -> None:
        Modem.__init__(self, scenario, **kwargs)

    @classmethod
    def from_yaml(cls: Type[Receiver], constructor: RoundTripConstructor, node: Node) -> Receiver:

        state = constructor.construct_mapping(node, CommentedOrderedMap)
        bits_source = state.pop(BitsSource.yaml_tag, None)

        args = dict((k.lower(), v) for k, v in state.items())
        receiver = Receiver(**args)

        if bits_source is not None:
            receiver.bits_source = bits_source

        return receiver

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
