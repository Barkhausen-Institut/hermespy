# -*- coding: utf-8 -*-
"""Interface prototype to the Quadriga channel model."""

from __future__ import annotations
from typing import List, Tuple, Optional, Type, TYPE_CHECKING, Any

import oct2py
from ruamel.yaml import SafeConstructor, SafeRepresenter, MappingNode
import os
import numpy as np

if TYPE_CHECKING:

    from modem import Transmitter, Receiver
    from channel import QuadrigaChannel

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Tobias Kronauer"
__email__ = "tobias.kronaue@barkhauseninstitut.org"
__status__ = "Prototype"


class QuadrigaInterface:
    """Implements the direct interface between hermes QuadrigChannel/quadriga backend.

    It is important to mention, that in the hermes implementation channels are
    independent of each other, i.e. are associated with each transmitter/receiver
    modem pair. However, this is not the case for quadriga which creates one
    channel for all transmitter/receiver modem pairs. Therefore, we need to do
    a mapping between the QuadrigaChannel objects for all transmitter/receiver
    modem pairs and the Quadriga simulation which runs in the background.

    This mapping is done in that class.
    """

    yaml_tag = u'QuadrigaInterface'
    __instance: Optional[QuadrigaInterface] = None
    __path_quadriga_src: str
    __antenna_kind: str         # TODO: Implement Enumeration for possible types of antennas
    __scenario_label: str
    __channels: List[QuadrigaChannel]
    __fetched_channels: List[QuadrigaChannel]
    __impulse_responses: List
    __delays: List

    def __init__(self,
                 path_quadriga_src: Optional[str] = None,
                 antenna_kind: Optional[str] = None,
                 scenario_label: Optional[str] = None) -> None:
        """Quadriga Interface object initialization.

        Args:
            path_quadriga_src (str, optional): Path to the Quadriga Matlab source files.
            antenna_kind (str, optional): Type of antenna considered.
            scenario_label (str, optional): Scenario label.
        """

        self.__path_quadriga_src = os.path.join(os.path.dirname(__file__), '..', '3rdparty', 'quadriga_src')
        self.__antenna_kind = 'lhcp-rhcp-dipole'
        self.__scenario_label = '3GPP_38.901_UMa_LOS'
        self.__channels = []
        self.__fetched_channels = []
        self.__impulse_responses = []
        self.__delays = []

        if path_quadriga_src is not None:
            self.path_quadriga_src = path_quadriga_src

        if antenna_kind is not None:
            self.antenna_kind = antenna_kind

        if scenario_label is not None:
            self.scenario_label = scenario_label

    @classmethod
    def GlobalInstance(cls: Type[QuadrigaInterface]) -> QuadrigaInterface:
        """Access the global Quadriga interface instance.

        Returns:
            QuadrigaInterface: Handle to the quadriga interface.
        """

        if QuadrigaInterface.__instance is None:
            QuadrigaInterface.__instance = cls()

        return QuadrigaInterface.__instance

    @classmethod
    def GlobalInstanceExists(cls: Type[QuadrigaInterface]) -> bool:
        """Checks if a global Quadriga interface instance exists.

        Returns:
            bool: If a global instance exists.
        """

        return QuadrigaInterface.__instance is None

    @classmethod
    def SetGlobalInstance(cls: Type[QuadrigaInterface], new_instance: QuadrigaInterface) -> None:
        """Set the new global quadriga instance.

        Copies registered channels to the new instance if a global instance already exists.

        Args:
            new_instance (QuadrigaInterface): The Quadriga interface instance to be made global.
        """

        if QuadrigaInterface.__instance is not None:

            for channel in QuadrigaInterface.__instance.__channels:
                new_instance.register_channel(channel)

        QuadrigaInterface.__instance = new_instance

    @property
    def path_quadriga_src(self) -> str:
        """Access the configured path to the Quadriga source files.

        Returns:
            str: Path to Quadriga sources.
        """

        return self.__path_quadriga_src

    @path_quadriga_src.setter
    def path_quadriga_src(self, path: str) -> None:
        """Modify the configured path to the Quadriga source files.

        Args:
            path (str): Path to Quadriga sources.

        Raises:
            ValueError: If the `path` does not exist within the filesystem.
        """

        if not os.path.exists(path):
            raise ValueError("Provided path to Quadriga sources does not exist within filesystem")

        self.__path_quadriga_src = path

    @property
    def antenna_kind(self) -> str:
        """Access the configured type of antenna.

        Returns:
            str: The configured antenna type.
        """

        return self.__antenna_kind

    @antenna_kind.setter
    def antenna_kind(self, antenna_type: str) -> None:
        """Modify the configured type of antenna.

        Args:
            antenna_type (str): String representation of the antenna type.
        """

        self.__antenna_kind = antenna_type

    @property
    def scenario_label(self) -> str:
        """Access the configured quadriga scenario label.

        Returns:
            str: The scenario label.
        """

        return self.__scenario_label

    @scenario_label.setter
    def scenario_label(self, label: str) -> None:
        """Modify the configured Quadriga scenario label.

        Args:
            label (str): The new label.
        """

        self.__scenario_label = label

    @property
    def channels(self) -> List[QuadrigaChannel]:
        """Access the currently registered quadriga channels.

        Returns:
            List[QuadrigaChannel]: List of channel objects.
        """

        return self.__channels

    def register_channel(self, channel: QuadrigaChannel) -> None:
        """Register a new Quadriga channel for simulation execution.

        Args:
            channel (QuadrigaChannel): The channel to be registered.

        Raises:
            ValueError: If the `channel` has already been registered.
        """

        if channel in self.__channels:
            raise ValueError("Channel has already been registered")

        self.__channels.append(channel)

    def unregister_channel(self, channel: QuadrigaChannel) -> None:
        """Unregister a Quadriga channel for simulation execution.

        Args:
            channel (QuadrigaChannel): The channel to be removed.

        Raises:
            ValueError: If the `channel` is not currently registered.
        """

        if channel not in self.__channels:
            raise ValueError("Channel is currently not registered")

        self.__channels.pop(self.__channels.index(channel))

    def get_impulse_response(self, channel: QuadrigaChannel) -> Tuple[np.ndarray, np.ndarray]:
        """Get the impulse response for a specific quadriga channel.

        Will launch the quadriga channel simulator if the channel has already been fetched.

        Args:
            channel (QuadrigaChannel): Channel for which to fetch the impulse response.

        Returns:
            (np.ndarray, np.ndarray): CIR and delay. Currently, only SISO.

        Raises:
            ValueError: If `channel` is not registered.
        """

        if channel not in self.__channels:
            raise ValueError("Channel not registered")

        # Launch the simulator if no channel has been fetched yet
        if len(self.__fetched_channels) == 0:
            self.__launch_quadriga()

        # Launch the simulator if the specific channel has already been fetched
        elif channel in self.__fetched_channels:

            self.__launch_quadriga()
            self.__fetched_channels = []

        # Mark this channel as having been fetched
        self.__fetched_channels.append(channel)

        channel_indices = self.__channel_indices[self.__channels.index(channel), :]
        channel = self.__cirs[channel_indices[0], channel_indices[1]]
        return channel.path_impulse_responses, channel.tau

    def __launch_quadriga(self) -> None:
        """Launches quadriga channel simulator.

        Raises:
            RuntimeError:
                If no channels are registered
                If transmitter sampling rates are not identical.
        """

        if len(self.__channels) < 1:
            raise RuntimeError("Attempting to launch Quadriga simulation without registered channels")

        transmitters: List[Transmitter] = []
        receivers: List[Receiver] = []

        self.__channel_indices = np.empty((len(self.__channels), 2), dtype=int)
        receiver_index = 0
        transmitter_index = 0

        for channel_idx, channel in enumerate(self.__channels):

            self.__channel_indices[channel_idx, :] = (receiver_index, transmitter_index)

            if channel.transmitter not in transmitters:

                transmitters.append(channel.transmitter)
                transmitter_index += 1

            if channel.receiver not in receivers:

                receivers.append(channel.receiver)
                receiver_index += 1

        carriers = np.empty(len(self.__channels), dtype=float)
        tx_positions = np.empty((len(transmitters), 3), dtype=float)
        rx_positions = np.empty((len(receivers), 3), dtype=float)
        tx_num_antennas = np.empty(len(transmitters), dtype=float)
        rx_num_antennas = np.empty(len(receivers), dtype=float)
        sampling_rates = np.empty(len(transmitters), dtype=float)

        for t, transmitter in enumerate(transmitters):

            position = transmitter.position
            if position is None:
                raise RuntimeError("Quadriga channel model requires transmitter position definitions")

            if np.array_equal(position, np.array([0, 0, 0])):
                raise RuntimeError("Position of transmitter must not be [0, 0, 0]")

            sampling_rates[t] = transmitter.scenario.sampling_rate
            carriers[t] = transmitter.carrier_frequency
            tx_positions[t, :] = position
            tx_num_antennas[t] = transmitter.num_antennas

        for r, receiver in enumerate(receivers):

            position = receiver.position
            if position is None:
                raise RuntimeError("Quadriga channel model requires receiver position definitions")

            rx_positions[r, :] = receiver.position
            rx_num_antennas[r] = receiver.num_antennas

        parameters = {
            "sampling_rate": sampling_rates,
            "carriers": carriers,
            "tx_position": tx_positions,
            "rx_position": rx_positions,
            "scenario_label": self.__scenario_label,
            "path_quadriga_src": self.__path_quadriga_src,
            "txs_number_antenna": tx_num_antennas,
            "rxs_number_antenna": rx_num_antennas,
            "tx_antenna_kind": self.__antenna_kind,
            "rx_antenna_kind": self.__antenna_kind,
            "number_tx": len(transmitters),
            "number_rx": len(receivers),
            "tracks_speed": np.zeros(len(receivers)),
            "tracks_length": np.zeros(len(receivers)),
            "tracks_angle": np.zeros(len(receivers)),
            "seed": np.random.rand(),
        }

        # Run quadriga for the specific interface implementation
        cirs = self._run_quadriga(**parameters)
        self.__cirs = cirs

    def _run_quadriga(self, **parameters) -> List[Any]:
        """Run the quadriga model.

        Must be realised by interface implementations.

        Args:
            **parameters: Quadriga channel parameters.
        """

        raise NotImplementedError("Neither a Matlab or Octave interface was found during Quadriga execution")

    @classmethod
    def to_yaml(cls: Type[QuadrigaInterface], representer: SafeRepresenter, node: QuadrigaInterface) -> MappingNode:
        """Serialize a QuadrigaInterface object to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (QuadrigaInterface):
                The QuadrigaInterface instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
        """

        state = {
            'path_quadriga_src': node.path_quadriga_src,
            'antenna_kind': node.antenna_kind,
            'scenario_label': node.scenario_label,
        }

        return representer.represent_mapping(cls.yaml_tag, state)

    @classmethod
    def from_yaml(cls: Type[QuadrigaInterface], constructor: SafeConstructor,  node: MappingNode) -> QuadrigaInterface:
        """Recall a new `QuadrigaInterface` instance from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `QuadrigaInterface` serialization.

        Returns:
            QuadrigaInterface:
                Newly created `QuadrigaInterface` instance. The internal references to modems will be `None` and need to be
                initialized by the `scenario` YAML constructor.

        """

        state = constructor.construct_mapping(node)
        return cls(**state)
