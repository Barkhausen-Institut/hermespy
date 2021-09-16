from __future__ import annotations
from ruamel.yaml import RoundTripRepresenter, RoundTripConstructor, Node
from typing import Type, List, TYPE_CHECKING

from modem import Modem

if TYPE_CHECKING:
    from scenario import Scenario


class Receiver(Modem):

    yaml_tag = 'Receiver'

    def __init__(self, scenario: Scenario, **kwargs) -> None:
        Modem.__init__(self, scenario, **kwargs)

    @classmethod
    def from_yaml(cls: Type[Receiver], constructor: RoundTripConstructor, node: Node) -> Receiver:

        scenario = [object for node, object in constructor.constructed_objects.items() if node.tag == 'Scenario'][0]
        return scenario.add_receiver(**constructor.construct_mapping(node))

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
