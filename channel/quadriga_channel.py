# -*- coding: utf-8 -*-
"""Single Channel of the Quadriga Channel Model Interface."""

from __future__ import annotations
from typing import Type, Tuple, TYPE_CHECKING, Optional
from ruamel.yaml import SafeRepresenter, SafeConstructor, ScalarNode, MappingNode
from math import ceil
import numpy as np

from channel import Channel, QuadrigaInterface

if TYPE_CHECKING:
    from scenario import Scenario
    from modem import Transmitter, Receiver

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Tobias Kronauer"
__email__ = "tobias.kronaue@barkhauseninstitut.org"
__status__ = "Prototype"


class QuadrigaChannel(Channel):
    """Quadriga Channel Model.

    Maps the output of the QuadrigaInterface to fit into hermes software architecture.
    """

    yaml_tag = u'Quadriga'
    yaml_matrix = True

    def __init__(self,
                 transmitter: Optional[Transmitter] = None,
                 receiver: Optional[Receiver] = None,
                 scenario: Optional[Scenario] = None,
                 active: Optional[bool] = None,
                 gain: Optional[float] = None) -> None:
        """Quadriga Channel object initialization.

        Automatically registers channel objects at the interface.

        Args:
            transmitter (Transmitter, optional):
                The modem transmitting into this channel.

            receiver (Receiver, optional):
                The modem receiving from this channel.

            scenario (Scenario, optional):
                Scenario this channel is attached to.

            active (bool, optional):
                Channel activity flag.
                Activated by default.

            gain (float, optional):
                Channel power gain.
                1.0 by default.
        """

        # Init base channel class
        Channel.__init__(self, transmitter, receiver, scenario, active, gain)

        # Register this channel at the interface
        self.__quadriga_interface.register_channel(self)

    def __del__(self) -> None:
        """Quadriga channel object destructor.

        Automatically un-registers channel objects at the interface.
        """

        self.__quadriga_interface.unregister_channel(self)

    @property
    def __quadriga_interface(self) -> QuadrigaInterface:
        """Access global Quadriga interface as property.

        Returns:
            QuadrigaInterface: Global Quadriga interface.
        """

        return QuadrigaInterface.GlobalInstance()

    def impulse_response(self, timestamps: np.ndarray) -> np.ndarray:

        # Query the quadriga interface for a new impulse response
        path_gains, path_delays = self.__quadriga_interface.get_impulse_response(self)

        max_delay_in_samples = np.around(
            np.max(path_delays) * self.scenario.sampling_rate).astype(int)

        impulse_response = np.zeros((timestamps.size,
                                     self.receiver.num_antennas,
                                     self.transmitter.num_antennas,
                                     max_delay_in_samples + 1), dtype=complex)

        for tx_antenna in range(self.transmitter.num_antennas):
            for rx_antenna in range(self.receiver.num_antennas):
                # of dimension, #paths x #snap_shots, along the third dimension are the samples
                # choose first snapshot, i.e. assume static
                cir_txa_rxa = path_gains[rx_antenna, tx_antenna, :]
                tau_txa_rxa = path_delays[rx_antenna, tx_antenna, :]

                time_delay_in_samples_vec = np.around(
                    tau_txa_rxa * self.scenario.sampling_rate).astype(int)

                for delay_idx, delay_in_samples in enumerate(
                        time_delay_in_samples_vec):

                    impulse_response[:, rx_antenna, tx_antenna, delay_in_samples] += (
                        cir_txa_rxa[delay_idx]
                    )
        return impulse_response
    
    @classmethod
    def to_yaml(cls: Type[QuadrigaChannel], representer: SafeRepresenter, node: QuadrigaChannel) -> MappingNode:
        """Serialize a QuadrigaChannel object to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (QuadrigaChannel):
                The QuadrigaChannel instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
        """

        state = {
            'active': node.__active,
            'gain': node.__gain
        }

        transmitter_index, receiver_index = node.indices

        yaml = representer.represent_mapping(u'{.yaml_tag} {} {}'.format(cls, transmitter_index, receiver_index), state)
        return yaml

    @classmethod
    def from_yaml(cls: Type[QuadrigaChannel], constructor: SafeConstructor,  node: MappingNode) -> QuadrigaChannel:
        """Recall a new `QuadrigaChannel` instance from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `QuadrigaChannel` serialization.

        Returns:
            QuadrigaChannel:
                Newly created `QuadrigaChannel` instance. The internal references to modems will be `None` and need to be
                initialized by the `scenario` YAML constructor.

        """

        # Handle empty yaml nodes
        if isinstance(node, ScalarNode):
            return cls()

        state = constructor.construct_mapping(node)
        return cls(**state)
