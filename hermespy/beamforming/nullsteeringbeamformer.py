# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import pinvh

from hermespy.beamforming import TransmitBeamformer, ReceiveBeamformer
from hermespy.core import AntennaArrayState, Serializable

__author__ = "Alan Thomas"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Alan Thomas", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


# Define the NullStearingBeamformer class by inheriting from the ReceiveBeamformer class
class NullSteeringBeamformer(Serializable, TransmitBeamformer, ReceiveBeamformer):
    """Implementation of the Null Steering Beamformer.
    Null-steering :footcite:`nullsteeringbeamformer:zarifi` transmit beamformers aim to maximize the received signal power in the direction of the intended receiver while substantially reducing the power impinging on the unintended receivers located in other directions.
    Let us introduce

    .. math::
        \\mathbf{a}_{\\psi_{\\bullet}} \\triangleq
        \\left[e^{j \\frac{2\pi}{\\lambda} r_1 \\cos(\\phi - \\psi_1)}, \\dots, e^{j \\frac{2\pi}{\lambda} r_K \\cos(\\phi - \\psi_K)} \\right]^T

    and

    .. math::
        \\mathbf{w} \\triangleq
        \\left[ w_1 \\dots w_K \\right]^T

    Where :math:`\mathbf{a}_{\\psi_{\\bullet}}` is the array propagation matrix and the vector W corresponds to the beamforming weights.
    Let

    .. math::
        \\mathbf{A} \\triangleq \\left[ \\mathbf{a}_{\phi_1} \\dots \\mathbf{a}_{\\phi_L} \\right]

    where A is the matrix that contains the array propagation vectors of all AoAs and :math:`P_T` denote the maximum admissible total transmission power.
    The beamformer has maximized power in the intended direction and nulls at the unintended directions when:

    .. math::
        \\max_{\mathbf{w}} & \quad |\mathbf{w}^H \mathbf{a}_0|^2 \\\\
        \\text{subject to} & \quad \mathbf{w}^H \mathbf{A} = 0 \\\\
                          & \quad \mathbf{w}^H \mathbf{w} \leq P_T.

    The optimal solution to the above expression can be derived as:

    .. math::
        \\mathbf{w}_{\\text{ns}} = \\frac{\sqrt{P_T}}{\|(\mathbf{I} - \\mathbf{P_A})\\mathbf{a}_0\|} \\cdot (\\mathbf{I} - \\mathbf{P_A})\\mathbf{a}_0

    where

    .. math::

        \\mathbf{P_A} \\triangleq \\mathbf{A}(\\mathbf{A}^H\\mathbf{A})^{-1}\\mathbf{A}^H

    is the orthogonal projection matrix onto the subspace spanned by the columns of A.
    As such, :math:`W_{\\text{ns}}` is in fact the orthogonal projection of onto the null space of :math:`a_0`.
    Thus, :math:`W_{\\text{ns}}` is the null steering beamformer.
    """

    yaml_tag = "NullSteeringBeamformer"

    def __init__(self) -> None:
        TransmitBeamformer.__init__(self)
        ReceiveBeamformer.__init__(self)

    def _num_transmit_input_streams(self, num_output_streams: int) -> int:
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
