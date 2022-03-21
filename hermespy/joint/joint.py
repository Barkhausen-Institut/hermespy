from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
from scipy.constants import speed_of_light
from scipy.signal import correlate

from hermespy.core.signal_model import Signal
from hermespy.modem import Modem
from hermespy.radar import Radar
from hermespy.radar.radar import RadarCube

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "Jan Adler"
__version__ = "0.2.7"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MatchedFilterJoint(Modem, Radar):
    """Joint Communication and Sensing Operator."""
    
    __transmission: Optional[Signal]
    
    def __init__(self) -> None:
        
        Modem.__init__(self)
        Radar.__init__(self)

        
    def transmit(self, duration: float = 0) -> Tuple[Signal, Symbols, np.ndarray]:
        
        # Cache the recently transmitted waveform for correlation during reception
        signal, symbols, bits =  super().transmit(duration)
        self.__transmission = signal
        
        return signal, symbols, bits

        
    def receive(self) -> Tuple[Signal, Symbols, np.ndarray, RadarCube]:
        
        # There must be a recent transmission being cached in order to correlate
        if self.__transmission is None:
            raise RuntimeError("Receiving from a matched filter joint must be preceeded by a transmission")
        
        # Receive information
        signal, symbols, bits = Modem.receive(self)
        
        correlation = correlate(signal, self.__transmission.samples, mode='full', method='fft')
        correlation /= (np.linalg.norm(self.__transmission.samples) ** 2)  # Normalize correlation
        
        angle_bins = np.array([0.])
        velocity_bins = np.array([0.])
        range_bins = np.array(signal.timestamps * speed_of_light)
        cube_data = np.array([[correlation]], dtype=float)
        cube = RadarCube(cube_data, angle_bins, velocity_bins, range_bins)

        return signal, symbols, bits, cube
        