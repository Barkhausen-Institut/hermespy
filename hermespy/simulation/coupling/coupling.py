# -*- coding: utf-8 -*-
"""
===============
Mutual Coupling
===============

.. toctree::
   :glob:

   simulation.coupling.perfect
   simulation.coupling.impedance
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from abc import ABC, abstractmethod

from hermespy.core import Signal, FloatingError

if TYPE_CHECKING:
    from ..simulated_device import SimulatedDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Coupling(ABC):
    """Base class for mutual coupling model implementations."""

    __device: Optional[SimulatedDevice]
    
    def __init__(self, device: Optional[SimulatedDevice] = None) -> None:
        """
        Args:
        
            device (SimulatedDevice, optional): Device the model is configured to.
        """
        
        self.device = device
        
    @property
    def device(self) -> Optional[SimulatedDevice]:
        """Device the model is configured to.
        
        Returns:
            Handle to the device.
            `None`, if the model is considered floating.
        """
        
        return self.__device
    
    @device.setter
    def device(self, value: Optional[SimulatedDevice]) -> None:
        
        self.__device = value
        
    def transmit(self, signal: Signal) -> Signal:
        """Apply the mutual coupling model during signal transmission.
        
        Args:
        
            signal (Signal): The signal to be transmitted.
            
        Returns: The signal resulting from coupling modeling.
        
        Raises:
        
            FloatingError: If the device is not specified.
            ValueError: If the number of signal streams does not match the number of transmitting antennas.
        """
        
        if self.device is None:
            raise FloatingError("Error trying to simulate coupling of a floating model")
        
        if self.device.antennas.num_transmit_antennas != signal.num_streams:
            raise ValueError("Number of signal streams ({signal.num_streams}) does not match the number of transmitting antennas ({self.device.antennas.num_transmit_antennas})")
        
        return self._transmit(signal)
    
    @abstractmethod
    def _transmit(self, signal: Signal) -> Signal:
        """Apply the mutual coupling model during signal transmission.
        
        Args:
        
            signal (Signal): The signal to be transmitted.
            
        Returns: The signal resulting from coupling modeling.
        """
        ...  # pragma no cover
        
        
    def receive(self, signal: Signal) -> Signal:
        """Apply the mutual coupling model during signal reception.
        
        Args:
        
            signal (Signal): The signal to be received.
            
        Returns: The signal resulting from coupling modeling.
        
        Raises:
        
            FloatingError: If the device is not specified.
            ValueError: If the number of signal streams does not match the number of transmitting antennas.
        """
        
        if self.device is None:
            raise FloatingError("Error trying to simulate coupling of a floating model")
        
        if self.device.antennas.num_transmit_antennas != signal.num_streams:
            raise ValueError("Number of signal streams ({signal.num_streams}) does not match the number of transmitting antennas ({self.device.antennas.num_transmit_antennas})")
        
        return self._receive(signal)
    
    @abstractmethod
    def _receive(self, signal: Signal) -> Signal:
        """Apply the mutual coupling model during signal reception.
        
        Args:
        
            signal (Signal): The signal to be received.
            
        Returns: The signal resulting from coupling modeling.
        """
        ...  # pragma no cover
