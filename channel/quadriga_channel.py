# -*- coding: utf-8 -*-
"""Single Channel of the Quadriga Channel Model Interface."""

from __future__ import annotations
from typing import Type, Tuple, TYPE_CHECKING, Optional
from ruamel.yaml import SafeRepresenter, SafeConstructor, ScalarNode, MappingNode
import numpy as np
import numpy.random as rnd

from channel import Channel, QuadrigaInterface

if TYPE_CHECKING:
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
                 active: Optional[bool] = None,
                 gain: Optional[float] = None,
                 sync_offset_low: Optional[float] = None,
                 sync_offset_high: Optional[float] = None,
                 random_generator: Optional[rnd.Generator] = None) -> None:
        """Quadriga Channel object initialization.

        Automatically registers channel objects at the interface.

        Args:
            transmitter (Transmitter, optional):
                The modem transmitting into this channel.

            receiver (Receiver, optional):
                The modem receiving from this channel.

            active (bool, optional):
                Channel activity flag.
                Activated by default.

            gain (float, optional):
                Channel power gain.
                1.0 by default.

            interface (QuadrigaInterface, optional):
                Interface handle to the quadriga backend.
        """

        # Init base channel class
        Channel.__init__(self, transmitter=transmitter,
                               receiver=receiver,
                               active=active,
                               gain=gain,
                               sync_offset_low=sync_offset_low,
                               sync_offset_high=sync_offset_high,
                               random_generator=random_generator)

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

    def propagate(self, transmitted_signal: np.ndarray) -> np.ndarray:

        if transmitted_signal.ndim != 2:
            raise ValueError("Transmitted signal must be a matrix (an array of two dimensions)")

        if self.transmitter is None or self.receiver is None:
            raise RuntimeError("Channel is floating, making propagation simulation impossible")

        cir, tau = self.__quadriga_interface.get_impulse_response(self)

        max_delay_in_samples = int(max(tau) * self.transmitter.sampling_rate)
        number_of_samples_out = transmitted_signal.shape[1] + max_delay_in_samples

        rx_signal = np.zeros((self.receiver.num_antennas, number_of_samples_out), dtype=complex)
        for rx_antenna in range(self.receiver.num_antennas):

            rx_signal_ant = np.zeros(number_of_samples_out, dtype=complex)

            for tx_antenna in range(self.transmitter.num_antennas):
                tx_signal_ant = transmitted_signal[tx_antenna, :]

                # of dimension, #paths x #snap_shots, along the third dimension are the samples
                # choose first snapshot, i.e. assume static
                cir_txa_rxa = cir[rx_antenna, tx_antenna, :, 0]
                tau_txa_rxa = tau[rx_antenna, tx_antenna, :, 0]

                # cir from quadriga corresponds to impulse_response_siso of
                # MultiPathFadingChannel

                time_delay_in_samples_vec = (tau_txa_rxa * self.transmitter.sampling_rate).astype(int)

                for delay_idx, delay_in_samples in enumerate(time_delay_in_samples_vec):

                    padding = self.max_delay_in_samples - delay_in_samples
                    path_response_at_delay = cir_txa_rxa[delay_idx]

                    signal_path = tx_signal_ant * path_response_at_delay
                    signal_path = np.concatenate((
                        np.zeros(delay_in_samples),
                        signal_path,
                        np.zeros(padding)))

                    rx_signal_ant += signal_path

            rx_signal[rx_antenna, :] = rx_signal_ant

        return rx_signal * self.gain

    def get_impulse_response(self, timestamps: np.array) -> np.ndarray:
        """Calculates the channel impulse response.

        This method can be used for instance by the transceivers to obtain the
        channel state information.

        Args:
            timestamps (np.array):
                Time instants with length T to calculate the response for.

        Returns:
            np.ndarray:
                Impulse response in all 'number_rx_antennas' x 'number_tx_antennas'
                channels at the time instants given in vector 'timestamps'.
                `impulse_response` is a 4D-array, with the following dimensions:
                1- sampling instants, 2 - Rx antennas, 3 - Tx antennas, 4 - delays
                (in samples)
                The shape is T x number_rx_antennas x number_tx_antennas x (L+1)
        """
        cir, tau = self.__quadriga_interface.get_impulse_response(self.transmitter, self.receiver)
        self.max_delay_in_samples = np.around(
            np.max(tau) * self.sampling_rate).astype(int)

        impulse_response = np.zeros((timestamps.size,
                                     self.number_rx_antennas,
                                     self.number_tx_antennas,
                                     self.max_delay_in_samples + 1), dtype=complex)

        for tx_antenna in range(self.number_tx_antennas):
            for rx_antenna in range(self.number_rx_antennas):
                # of dimension, #paths x #snap_shots, along the third dimension are the samples
                # choose first snapshot, i.e. assume static
                cir_txa_rxa = cir[rx_antenna, tx_antenna, :, 0]
                tau_txa_rxa = tau[rx_antenna, tx_antenna, :, 0]

                time_delay_in_samples_vec = np.around(
                    tau_txa_rxa * self.sampling_rate).astype(int)

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
            'active': node.active,
            'gain': node.gain,
            'sync_offset_low': node.sync_offset_low,
            'sync_offset_high': node.sync_offset_high
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
