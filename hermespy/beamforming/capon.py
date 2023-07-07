# -*- coding: utf-8 -*-
"""
================
Capon Beamformer
================
"""

import numpy as np

from hermespy.core import Serializable
from .beamformer import ReceiveBeamformer

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class CaponBeamformer(Serializable, ReceiveBeamformer):
    """Implementation of the Capon beamformer, also referred to as Minimum Variance Distortionless Response (MVDR).

    The Capon\ :footcite:`1969:capon` beamformer estimates the power :math:`\\hat{P}` received from a direction :math:`(\\theta, \\phi)`, where :math:`\\theta` is the zenith and :math:`\\phi`  is the azimuth angle of interest in spherical coordinates, respectively.
    Let :math:`\\mathbf{X} \in \mathbb{C}^{N \\times T}` be the the matrix of :math:`T` time-discrete samples acquired by an antenna arrary featuring :math:`N` antennas and

    .. math::

       \\mathbf{R}^{-1} = \\left( \\mathbf{X}\\mathbf{X}^{\\mathsf{H}} + \\lambda \\mathbb{I} \\right)^{-1}

    be the respective inverse sample correlation matrix loaded by a factor :math:`\\lambda \\in \\mathbb{R}_{+}`.
    The antenna array's response towards a source within its far field emitting a signal of small relative bandwidth is :math:`\\mathbf{a}(\\theta, \\phi) \\in \\mathbb{C}^{N}`.
    Then, the Capon beamformer's spatial power response is defined as

    .. math::

       \\hat{P}_{\\mathrm{Capon}}(\\theta, \\phi) = \\frac{1}{\\mathbf{a}^{\\mathsf{H}}(\\theta, \\phi) \mathbf{R}^{-1} \\mathbf{a}(\\theta, \\phi)}

    with

    .. math::

       \\mathbf{w}(\\theta, \\phi) = \\frac{\\mathbf{R}^{-1} \\mathbf{a}(\\theta, \\phi)}{\\mathbf{a}^{\\mathsf{H}}(\\theta, \\phi) \mathbf{R}^{-1} \\mathbf{a}(\\theta, \\phi)} \\in \\mathbb{C}^{N}

    being the beamforming weights to steer the sensor array's receive characteristics towards direction :math:`(\\theta, \\phi)`, so that

    .. math::

       \\mathcal{B}\\lbrace \\mathbf{X} \\rbrace = \\mathbf{w}^\\mathsf{H}(\\theta, \\phi) \\mathbf{X}

    is the implemented beamforming equation.
    """

    yaml_tag = "Capon"

    def __init__(self, loading: float = 0.0, **kwargs) -> None:
        """
        Args:

            loading (float, optional):
                Diagonal covariance loading coefficient :math:`\\lambda`.
                Defaults to zero.
        """

        self.loading = loading
        ReceiveBeamformer.__init__(self, **kwargs)

    @property
    def num_receive_input_streams(self) -> int:
        return self.operator.device.antennas.num_antennas

    @property
    def num_receive_output_streams(self) -> int:
        return 1

    @property
    def num_receive_focus_angles(self) -> int:
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

    def _decode(self, samples: np.ndarray, carrier_frequency: float, angles: np.ndarray) -> np.ndarray:
        # Compute the inverse sample covariance matrix R
        # In order to avoid algebra exceptions on decodings without noise, we will resort to the pseudo-inverse,
        # which is able to invert rank-deficient matrices
        sample_covariance = np.linalg.inv(samples @ samples.T.conj() + self.loading * np.eye(samples.shape[0]))

        # Query the sensor array response vectors for the angles of interest and create a dictionary from it
        dictionary = np.empty((self.num_receive_input_streams, angles.shape[0]), dtype=complex)
        for d, focus in enumerate(angles):
            array_response = self.operator.device.antennas.spherical_phase_response(carrier_frequency, focus[0, 0], focus[0, 1])
            dictionary[:, d] = sample_covariance @ array_response / (array_response.T.conj() @ sample_covariance @ array_response)

        beamformed_samples = dictionary.T.conj() @ samples
        return beamformed_samples[:, np.newaxis, :]
