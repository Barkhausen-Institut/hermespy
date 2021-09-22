from __future__ import annotations
import numpy as np
from ruamel.yaml import SafeConstructor, SafeRepresenter, Node
from typing import Type, Optional
from modem.rf_chain_models.power_amplifier import PowerAmplifier


class RfChain:
    """Implements an RF chain model.

    Only PA is modelled.
    """

    yaml_tag = 'RfChain'
    __tx_power: float
    __power_amplifier: Optional[PowerAmplifier]

    def __init__(self, tx_power: float = None) -> None:

        self.__tx_power = 1.0
        self.__power_amplifier = None

        if tx_power is not None:
            self.__tx_power = tx_power

    @classmethod
    def to_yaml(cls: Type[RfChain], representer: SafeRepresenter, node: RfChain) -> Node:
        """Serialize an RfChain object to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (RfChain):
                The `RfChain` instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
                None if the object state is default.
        """

        state = {}

        if node.__power_amplifier is not None:
            state[node.power_amplifier.yaml_tag] = node.__power_amplifier

        if len(state) < 1:
            return representer.represent_none(None)

        return representer.represent_mapping(cls.yaml_tag, state)

    @classmethod
    def from_yaml(cls: Type[RfChain], constructor: SafeConstructor, node: Node) -> RfChain:
        """Recall a new `RfChain` instance from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `RfChain` serialization.

        Returns:
            RfChain:
                Newly created `RfChain` instance.
        """

        state = constructor.construct_mapping(node)
        power_amplifier = state.pop(PowerAmplifier.yaml_tag, None)

        rf_chain = cls(**state)
        yield rf_chain

        if power_amplifier is not None:
            rf_chain.power_amplifier = power_amplifier

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

    @property
    def power_amplifier(self) -> PowerAmplifier:
        """Access the `PowerAmplifier` of the rf chain.

        Returns:
            A handle to the `PowerAmplifier`.
        """

        return self.__power_amplifier

    @power_amplifier.setter
    def power_amplifier(self, power_amplifier: PowerAmplifier) -> None:
        """Reassign the power amplifier configuration.

        Args:
            power_amplifier (PowerAmplifier):
                The new power amplifier configuration.
        """

        self.__power_amplifier = power_amplifier
