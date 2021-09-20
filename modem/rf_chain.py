from __future__ import annotations
import numpy as np
from ruamel.yaml import RoundTripConstructor, RoundTripRepresenter, Node
from typing import Type
from parameters_parser.parameters_rf_chain import ParametersRfChain
from modem.rf_chain_models.power_amplifier import PowerAmplifier


class RfChain:
    """Implements an RF chain model.

    Only PA is modelled.
    """

    yaml_tag = 'RfChain'
    __power_amplifier: PowerAmplifier

    def __init__(self, param: ParametersRfChain = None,
                 tx_power: float = 1.0) -> None:

        self.__power_amplifier = None

        self.param = param
        if self.param is not None:
            self.power_amplifier = PowerAmplifier(self.param, tx_power)

    @classmethod
    def to_yaml(cls: Type[RfChain], representer: RoundTripRepresenter, node: RfChain) -> Node:
        """Serialize an RfChain object to YAML.

        Args:
            representer (RoundTripRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (RfChain):
                The `RfChain` instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
        """

        return representer.represent_none(None)

    @classmethod
    def from_yaml(cls: Type[RfChain], constructor: RoundTripConstructor, node: Node) -> RfChain:
        """Recall a new `RfChain` instance from YAML.

        Args:
            constructor (RoundTripConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `RfChain` serialization.

        Returns:
            RfChain:
                Newly created `RfChain` instance.
        """

        return RfChain()

    def send(self, input_signal: np.ndarray) -> np.ndarray:
        """Returns the distorted version of signal in "input_signal".

        According to transmission impairments.
        """
        if self.param is not None:
            output_signal = self.power_amplifier.send(input_signal)
        else:
            output_signal = input_signal
        return output_signal

    def receive(self, input_signal: np.ndarray) -> np.ndarray:
        """Returns the distorted version of signal in "input_signal".

        According to reception impairments.
        """
        return input_signal
