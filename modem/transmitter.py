from __future__ import annotations
from ruamel.yaml import RoundTripRepresenter, RoundTripConstructor, Node
from typing import Type, List, TYPE_CHECKING

from modem import Modem

if TYPE_CHECKING:
    from scenario import Scenario


class Transmitter(Modem):

    yaml_tag = 'Transmitter'

    def __init__(self, scenario: Scenario, **kwargs) -> None:

        Modem.__init__(self, scenario, **kwargs)

    @classmethod
    def to_yaml(cls: Type[Transmitter], representer: RoundTripRepresenter, node: Transmitter) -> Node:
        """Serialize a modem object to YAML.

        Args:
            representer (RoundTripRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (Modem):
                The modem instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
        """

        serialization = {
            "carrier_frequency": node.carrier_frequency,
            "sampling_rate": node.sampling_rate
        }

        return representer.represent_mapping(node.yaml_tag, serialization)

    @classmethod
    def from_yaml(cls: Type[Transmitter], constructor: RoundTripConstructor, node: Node) -> Transmitter:

        # Recover scenario instance which must have been created already
        # Ugly hack, there must be a better way to handle this
        constructor.constructed_objects(constructor.)
        for constructed_object in constructor.constructed_objects.values():
            if isinstance(constructed_object, type(Scenario)):
                return constructed_object.add_transmitter(**constructor.construct_mapping(node))

    @property
    def paired_modems(self) -> List[Modem]:
        """The modems connected to this modem over an active channel.

        Returns:
            List[Modem]:
                A list of paired modems.
        """

        return [channel.receiver for channel in self.scenario.departing_channels(self, True)]
