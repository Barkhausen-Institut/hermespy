# -*- coding: utf-8 -*-
"""
===========
Beamforming
===========
"""

from abc import ABC, abstractmethod, abstractproperty
from enum import Enum
from typing import Optional

import numpy as np

from hermespy.core import Signal
from hermespy.precoding import SymbolPrecoder


__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.7"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class BeamMode(Enum):
    """Beamformer Mode."""

    TX = 0
    RX = 1
    BOTH = 2

class Beamformer(SymbolPrecoder, ABC):
    """Base class for beam steering precodings."""
    
    __mode: BeamMode
    
    def __init__(self,
                 mode: BeamMode = BeamMode.BOTH) -> None:
        """Args:
        
            mode (BeamMode):
                Mode the beamformer operates in.
                Both (TX and RX) by default.
        """
        
        self.mode = mode
        
    @property
    def mode(self) -> BeamMode:
        """The mode the beamformer operates in.
        
        Returns:
        
            The mode.
        """
        
        return self.__mode
    
    @mode.setter
    def mode(self, value: BeamMode) -> None:
        
        self.__mode = value
        
    @abstractproperty
    def focus(self) -> np.ndarray:
        """Focus of the beamformer during transmit and receive.
        
        Returns:
        
            The focus point.
        """
        ...
    
    def encode(self, symbol_stream: np.ndarray) -> np.ndarray:
        
        
    
    def transmit(self, signal: Signal, focus: Optional[np.ndarray]) -> Signal:
        
        steered_signal = signal.copy()
        
        