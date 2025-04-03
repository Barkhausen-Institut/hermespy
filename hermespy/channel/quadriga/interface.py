# -*- coding: utf-8 -*-

from __future__ import annotations
from os import getenv
import os.path as path

import numpy as np

from hermespy.core import RandomNode
from ..channel import LinkState

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
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

    _instance: QuadrigaInterface | None = None
    __path_quadriga_src: str
    __antenna_kind: str  # TODO: Implement Enumeration for possible types of antennas
    __scenario_label: str

    def __init__(
        self,
        path_quadriga_src: str | None = None,
        antenna_kind: str = "omni",
        scenario_label: str = "3GPP_38.901_UMa_LOS",
        seed: int | None = None,
    ) -> None:
        """
        Args:
            path_quadriga_src:
                Path to the Quadriga Matlab source files.
                If not specified, the environment variable `HERMES_QUADRIGA` is assumed.

            antenna_kind:
                Type of antenna considered.
                Defaults to "omni".

            scenario_label:
                Scenario label.

            seed:
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

    @property
    def path_launch_script(self) -> str:
        """Generate path to the launch Matlab script.

        Returns:
            Path to the launch file.
        """

        return path.split(__file__)[0]

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

    def sample_quadriga(self, state: LinkState) -> np.ndarray:
        """Sample the Quadriga channel model.

        Args:

            state:
                Physical state of the channel at the time of sampling.
        """

        # ToDo: Check if the transmitter position is in the coordinate system origin

        parameters = {
            "sampling_rate": state.bandwidth,
            "carriers": np.array([state.carrier_frequency]),
            "tx_position": np.array([state.transmitter.position]),
            "rx_position": np.array([state.receiver.position]),
            "scenario_label": self.__scenario_label,
            "path_quadriga_src": self.__path_quadriga_src,
            "txs_number_antenna": np.array([state.transmitter.antennas.num_transmit_antennas]),
            "rxs_number_antenna": np.array([state.receiver.antennas.num_receive_antennas]),
            "tx_antenna_kind": self.__antenna_kind,
            "rx_antenna_kind": self.__antenna_kind,
            "number_tx": 1,
            "number_rx": 1,
            "tracks_speed": np.zeros(1),
            "tracks_length": np.zeros(1),
            "tracks_angle": np.zeros(1),
            "seed": self._rng.integers(0, 2**32 - 1),
            "sec_per_snap": 1e-2,  # Referred tp as update_rate in the documentation
        }

        # Run quadriga for the specific interface implementation
        return self._run_quadriga(**parameters)

    def _run_quadriga(self, **parameters) -> np.ndarray:
        """Run the quadriga model.

        Must be realised by interface implementations.

        Args:
            \**parameters: Quadriga channel parameters.
        """

        raise NotImplementedError(
            "Neither a Matlab or Octave interface was found during Quadriga execution"
        )
