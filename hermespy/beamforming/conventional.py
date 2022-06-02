# -*- coding: utf-8 -*-
"""
========================
Conventional Beamforming
========================

Also refererd to as Delay and Sum Beamformer.
"""

from functools import lru_cache
from typing import Optional

import numpy as np
from numba import jit

from hermespy.core import Operator, Serializable
from .beamformer import TransmitBeamformer, ReceiveBeamformer


__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.7"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class ConventionalBeamformer(Serializable, TransmitBeamformer, ReceiveBeamformer):
    """Conventional delay and sum beamforming."""
    
    yaml_tag = u'ConventionalBeamformer'
    """YAML serialization tag."""
    
    def __init__(self, operator: Optional[Operator] = None) -> None:
        
        TransmitBeamformer.__init__(self, operator=operator)
        ReceiveBeamformer.__init__(self, operator=operator)
        
    @property
    def num_receive_focus_angles(self) -> int:
        
        # The conventional beamformer focuses a single angle
        return 1
        
    @property
    def num_receive_input_streams(self) -> int:
        
        # The conventional beamformer will allways consider all antennas streams
        return self.operator.device.antennas.num_antennas
    
    @property
    def num_receive_output_streams(self) -> int:
        
        # The convetional beamformer will always return a single stream,
        # combining all antenna signals into one
        return 1
    
    @property
    def num_transmit_focus_angles(self) -> int:
        
        # The conventional beamformer focuses a single angle
        return 1
    
    @property
    def num_transmit_output_streams(self) -> int:
        
        # The conventional beamformer will allways consider all antennas streams
        return self.operator.device.antennas.num_antennas
    
    @property
    def num_transmit_input_streams(self) -> int:
        
        # The convetional beamformer will always return a single stream,
        # combining all antenna signals into one
        return 1
    
    #@lru_cache(maxsize=2)
    def _codebook(self,
                  carrier_frequency: float,
                  angles: np.ndarray) -> np.ndarray:
        """Compute the beamforming codebook for a given set of angles of interest.
        
        Args:
        
            carrier_frequency (float):
                The assumed carrier central frequency of the samples.
        
            angles: (np.ndarray):
                Spherical coordinate system angles of arrival in radians.
                A two dimensional numpy array with the first dimension representing the number of angles,
                and the second dimension of magnitude two containing the azimuth and zenith angle in radians, respectively.
        
        Returns:
        
            The codebook represented by a two-dimensional numpy array,
            with the first dimension being the number of angles and the second dimension the number of antennas.
        """
    
        book = np.empty((angles.shape[0], self.operator.device.antennas.num_antennas), dtype=complex)
        for n, (azimuth, zenith) in enumerate(angles):
            book[n, :] = self.operator.device.antennas.spherical_response(carrier_frequency, azimuth, zenith).conj()

        return book / self.operator.device.antennas.num_antennas

    def _encode(self,
                samples: np.ndarray,
                carrier_frequency: float,
                focus_angles: np.ndarray) -> np.ndarray:
        
        azimuth, zenith = focus_angles[0, :]

        # Compute conventional beamformer weights
        weights = self.operator.device.antennas.spherical_response(carrier_frequency, azimuth, zenith).conj()
        
        # Weight the streams accordingly
        samples = weights[:, np.newaxis] @ samples
        
        # That's it
        return samples
    
    @staticmethod
    @jit(nopython=True)
    def _beamform(codebook: np.ndarray, 
                  samples: np.ndarray,
                  conjugate: bool = False) -> np.ndarray: # pragma: no cover
    
        if conjugate:
            return codebook.conj() @ samples
        
        else:
            return codebook @ samples

    def _decode(self,
                samples: np.ndarray,
                carrier_frequency: float,
                angles: np.ndarray) -> np.ndarray:

        codebook = self._codebook(carrier_frequency, angles[:, 0, :])
        beamformed_samples = self._beamform(codebook, samples, True)
        
        return beamformed_samples[:, np.newaxis, :]
