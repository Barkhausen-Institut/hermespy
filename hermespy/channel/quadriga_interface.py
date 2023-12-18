# -*- coding: utf-8 -*-

from __future__ import annotations
from os import getenv
import os.path as path
from typing import List, Tuple, Optional, Type, TYPE_CHECKING

import numpy as np

from hermespy.core import RandomNode

if TYPE_CHECKING:
    from hermespy.channel import QuadrigaChannel  # pragma: no cover
    from hermespy.simulation import SimulatedDevice  # pragma: no cover

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class QuadrigaInterface(RandomNode):
    """Interface between Hermes' channel model and the Quadriga channel simulator.

    It is important to mention, that in the hermes implementation channels are
    independent of each other, i.e. are associated with each transmitter/receiver
    modem pair. However, this is not the case for quadriga which creates one
    channel for all transmitter/receiver modem pairs. Therefore, we need to do
    a mapping between the QuadrigaChannel objects for all transmitter/receiver
    modem pairs and the Quadriga simulation which runs in the background.

    This mapping is done in that class.
    """

    yaml_tag = "QuadrigaInterface"
    _instance: Optional[QuadrigaInterface] = None
    __path_quadriga_src: str
    __antenna_kind: str  # TODO: Implement Enumeration for possible types of antennas
    __scenario_label: str
    __channels: List[QuadrigaChannel]
    __fetched_channels: List[QuadrigaChannel]

    def __init__(
        self,
        path_quadriga_src: Optional[str] = None,
        antenna_kind: str = "omni",
        scenario_label: str = "3GPP_38.901_UMa_LOS",
        seed: int | None = None,
    ) -> None:
        """
        Args:
            path_quadriga_src (str, optional):
                Path to the Quadriga Matlab source files.
                If not specified, the environment variable `HERMES_QUADRIGA` is assumed.

            antenna_kind (str, optional):
                Type of antenna considered.
                Defaults to "omni".

            scenario_label (str, optional):
                Scenario label.

            seed (int, optional):
                Random seed.
        """

        # Initialize base class
        RandomNode.__init__(self, seed=seed)

        # Infer the quadriga source path
        default_src_path = path.join(
            path.dirname(__file__), "..", "..", "submodules", "quadriga", "quadriga_src"
        )
        self.path_quadriga_src = (
            getenv("HERMES_QUADRIGA", default_src_path)
            if path_quadriga_src is None
            else path_quadriga_src
        )

        self.antenna_kind = antenna_kind
        self.scenario_label = scenario_label
        self.__channels = []
        self.__fetched_channels = []

    @property
    def path_launch_script(self) -> str:
        """Generate path to the launch Matlab script.

        Returns:
            Path to the launch file.
        """

        return path.join(path.split(__file__)[0], "res")

    @classmethod
    def GlobalInstance(cls: Type[QuadrigaInterface]) -> QuadrigaInterface:
        """Access the global Quadriga interface instance.

        If no global instance exists, a new one is created.

        Returns: Handle to the global Quadriga interface instance.
        """

        if QuadrigaInterface._instance is None:
            QuadrigaInterface._instance = cls()

        return QuadrigaInterface._instance

    @classmethod
    def GlobalInstanceExists(cls: Type[QuadrigaInterface]) -> bool:
        """Checks if a global Quadriga interface instance exists.

        Returns: Boolean indicating if a global instance exists.
        """

        return QuadrigaInterface._instance is not None

    @classmethod
    def SetGlobalInstance(cls: Type[QuadrigaInterface], new_instance: QuadrigaInterface) -> None:
        """Set the new global quadriga instance.

        Copies registered channels to the new instance if a global instance already exists.

        Args:

            new_instance (QuadrigaInterface):
                The Quadriga interface instance to be made global.
        """

        if QuadrigaInterface._instance is not None:
            for channel in QuadrigaInterface._instance.__channels:
                new_instance.register_channel(channel)

        QuadrigaInterface._instance = new_instance

    @property
    def path_quadriga_src(self) -> str:
        """Path to the configured Quadriga source files.

        Raises:

            ValueError: If the path does not exist within the filesystem.
        """

        return self.__path_quadriga_src

    @path_quadriga_src.setter
    def path_quadriga_src(self, location: str) -> None:
        if not path.exists(location):
            raise ValueError(
                f"Provided path to Quadriga sources {location} does not exist within filesystem"
            )

        self.__path_quadriga_src = location

    @property
    def antenna_kind(self) -> str:
        """Assumed type of antenna."""

        return self.__antenna_kind

    @antenna_kind.setter
    def antenna_kind(self, antenna_type: str) -> None:
        self.__antenna_kind = antenna_type

    @property
    def scenario_label(self) -> str:
        """Configured Quadriga scenario label."""

        return self.__scenario_label

    @scenario_label.setter
    def scenario_label(self, label: str) -> None:
        """Modify the configured Quadriga scenario label."""

        self.__scenario_label = label

    @property
    def channels(self) -> List[QuadrigaChannel]:
        """List of registered Quadriga channels."""

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
        """

        if self.channel_registered(channel):
            self.__channels.pop(self.__channels.index(channel))

    def channel_registered(self, channel: QuadrigaChannel) -> bool:
        """Is the channel currently registered at the interface?

        Args:

            channel (QuadrigaChannel):
                The channel in question.

        Returns: Boolean indicating if the channel is registered.
        """

        return channel in self.__channels

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
        channel_path = self.__cirs[channel_indices[0], channel_indices[1]]  # type: ignore
        return channel_path.path_impulse_responses, channel_path.tau

    def __launch_quadriga(self) -> None:
        """Launches quadriga channel simulator.

        Raises:

            RuntimeError: If no channels are registered
            RuntimeError: If transmitter sampling rates are not identical.
        """

        transmitters: List[SimulatedDevice] = []
        receivers: List[SimulatedDevice] = []

        self.__channel_indices = np.empty((len(self.__channels), 2), dtype=int)
        receiver_index = 0
        transmitter_index = 0

        for channel_idx, channel in enumerate(self.__channels):
            self.__channel_indices[channel_idx, :] = (receiver_index, transmitter_index)

            if channel.alpha_device not in transmitters:
                transmitters.append(channel.alpha_device)
                transmitter_index += 1

            if channel.beta_device not in receivers:
                receivers.append(channel.beta_device)
                receiver_index += 1

        carriers = np.empty(len(self.__channels), dtype=float)
        tx_positions = np.empty((len(transmitters), 3), dtype=float)
        rx_positions = np.empty((len(receivers), 3), dtype=float)
        tx_num_antennas = np.empty(len(transmitters), dtype=float)
        rx_num_antennas = np.empty(len(receivers), dtype=float)
        sampling_rates = np.empty(len(transmitters), dtype=float)

        for t, transmitter in enumerate(transmitters):
            position = transmitter.position

            if np.array_equal(position, np.array([0, 0, 0])):
                raise RuntimeError("Position of transmitter must not be [0, 0, 0]")

            sampling_rates[t] = transmitter.sampling_rate
            carriers[t] = transmitter.carrier_frequency
            tx_positions[t, :] = position
            tx_num_antennas[t] = transmitter.num_antennas

        for r, receiver in enumerate(receivers):
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
            "seed": self._rng.integers(0, 2**32 - 1),
        }

        # Run quadriga for the specific interface implementation
        cirs = self._run_quadriga(**parameters)
        self.__cirs = cirs

    def _run_quadriga(self, **parameters) -> np.ndarray:
        """Run the quadriga model.

        Must be realised by interface implementations.

        Args:
            **parameters: Quadriga channel parameters.
        """

        raise NotImplementedError(
            "Neither a Matlab or Octave interface was found during Quadriga execution"
        )
