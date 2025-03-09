# -*- coding: utf-8 -*-

from __future__ import annotations
from typing_extensions import override

import numpy as np

from hermespy.core import (
    AntennaMode,
    AntennaArrayState,
    DeserializationProcess,
    SerializationProcess,
)
from .beamformer import ReceiveBeamformer

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class CaponBeamformer(ReceiveBeamformer):

    def __init__(self, loading: float = 0.0) -> None:
        """
        Args:

            loading (float, optional):
                Diagonal covariance loading coefficient :math:`\\lambda`.
                Defaults to zero.
        """

        self.loading = loading
        ReceiveBeamformer.__init__(self)

    def num_receive_output_streams(self, num_input_streams: int) -> int:
        # The capon beaformer combies all input streams into a single output stream
        return 1

    @property
    def num_receive_focus_points(self) -> int:
        return 1

    @property
    def loading(self) -> float:
        """Magnitude of the diagonal sample covariance matrix loading.

        Required for robust matrix inversion in the case of rank-deficient sample covariances.

        Returns:

            Diagonal loading coefficient :math:`\\lambda`.

        Raises:

            ValueError: For loading coefficients smaller than zero.
        """

        return self.__loading

    @loading.setter
    def loading(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("Diagonal loading coefficient must be greater or equal to zero")

        self.__loading = value

    def _decode(
        self,
        samples: np.ndarray,
        carrier_frequency: float,
        angles: np.ndarray,
        array: AntennaArrayState,
    ) -> np.ndarray:
        # Compute the inverse sample covariance matrix R
        # In order to avoid algebra exceptions on decodings without noise, we will resort to the pseudo-inverse,
        # which is able to invert rank-deficient matrices
        sample_covariance = np.linalg.inv(
            samples @ samples.T.conj() + self.loading * np.eye(samples.shape[0])
        )

        # Query the sensor array response vectors for the angles of interest and create a dictionary from it
        dictionary = np.empty((samples.shape[0], angles.shape[0]), dtype=complex)
        for d, focus in enumerate(angles):
            array_response = array.spherical_phase_response(
                carrier_frequency, focus[0, 0], focus[0, 1], AntennaMode.RX
            )
            dictionary[:, d] = (
                sample_covariance
                @ array_response
                / (array_response.T.conj() @ sample_covariance @ array_response)
            )

        beamformed_samples = dictionary.T.conj() @ samples
        return beamformed_samples[:, np.newaxis, :]

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_floating(self.loading, "loading")

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> CaponBeamformer:
        return CaponBeamformer(process.deserialize_floating("loading"))
