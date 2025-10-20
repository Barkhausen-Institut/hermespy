# -*- coding: utf-8 -*-

from __future__ import annotations
from typing_extensions import override

import numpy as np
from scipy.linalg import pinvh

from hermespy.beamforming import TransmitBeamformer, ReceiveBeamformer
from hermespy.core import AntennaArrayState, DeserializationProcess, SerializationProcess

__author__ = "Alan Thomas"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Alan Thomas", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class NullSteeringBeamformer(TransmitBeamformer, ReceiveBeamformer):

    def __init__(self) -> None:
        TransmitBeamformer.__init__(self)
        ReceiveBeamformer.__init__(self)

    def num_transmit_input_streams(self, num_output_streams: int) -> int:
        # The null steering beamformer distirbutes a single stream
        # to an arbitrary number of antenna streams
        return 1

    def num_receive_output_streams(self, num_input_streams: int) -> int:
        # The null steering beamformer will always return a single stream,
        # combining all antenna signals into one
        return 1

    @property
    def num_transmit_focus_points(self) -> int:
        # The null steering beamformer focuses a single direction,
        # while steering the nulls in two other directions
        return 3

    @property
    def num_receive_focus_points(self) -> int:
        # The null steering beamformer focuses a single direction,
        # while steering the nulls in two other directions
        return 3

    # calculate the null steering beamformer weights
    def _weights(
        self, carrier_frequency: float, focus_angles: np.ndarray, array: AntennaArrayState
    ) -> np.ndarray:

        a0 = array.spherical_phase_response(
            carrier_frequency, focus_angles[0, 0], focus_angles[0, 1]
        )
        a1 = array.spherical_phase_response(
            carrier_frequency, focus_angles[1, 0], focus_angles[1, 1]
        )
        a2 = array.spherical_phase_response(
            carrier_frequency, focus_angles[2, 0], focus_angles[2, 1]
        )

        A = np.array([a1, a2]).T
        PA = A.conj() @ pinvh(A.T @ A.conj(), check_finite=False) @ A.T
        Identity_Matrix = np.eye(PA.shape[0])
        wns = (Identity_Matrix - PA) @ a0
        wns /= np.linalg.norm(wns)
        return wns

    def _encode(
        self,
        samples: np.ndarray,
        carrier_frequency: float,
        focus_angles: np.ndarray,
        array: AntennaArrayState,
    ) -> np.ndarray:

        # Compute nullsteering beamformer weights
        weights = self._weights(carrier_frequency, focus_angles, array)
        # Weight the streams accordingly
        samples = weights[:, np.newaxis] @ samples

        return samples

    def _decode(
        self,
        samples: np.ndarray,
        carrier_frequency: float,
        angles: np.ndarray,
        array: AntennaArrayState,
    ) -> np.ndarray:

        # Query the sensor array response vectors for the angles of interest and create a dictionary from it which contains the beamforming weights
        dictionary = np.empty((array.num_receive_antennas, angles.shape[0]), dtype=complex)
        for d, focus in enumerate(angles):
            dictionary[:, d] = self._weights(carrier_frequency, focus, array)

        beamformed_samples = dictionary.T @ samples
        return beamformed_samples[:, np.newaxis, :]

    @override
    def serialize(self, process: SerializationProcess) -> None:
        return

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> NullSteeringBeamformer:
        return cls()
