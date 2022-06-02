# -*- coding: utf-8 -*-
"""
==========
Beamformer
==========

Beamforming is split into the prototype classes :class:`.TransmitBeamformer` and :class:`.ReceiveBeamformer`
for beamforming operations during signal transmission and reception, respectively.
They are both derived from the base :class:`BeamformerBase`.
This is due to the fact that some beamforming algorithms may be exclusive to transmission or reception use-cases.
Should a beamformer be applicable during both transmission and reception both prototypes can be inherited.
An example for such an implementation is the :class:`Conventional <.conventional.ConventionalBeamformer>` beamformer.
"""

from abc import ABC, abstractmethod, abstractproperty
from enum import Enum
from typing import Optional, Tuple, Union

import numpy as np

from hermespy.core import FloatingError, Operator, Signal, Receiver, Transmitter

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "3.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"
    

class FocusMode(Enum):
    """The focus mode of the beamformer."""
    
    SPHERICAL = 0
    """Focus points in spherical coordinates, i.e. azimuth and zenith angles in radians."""
    
    HORIZONTAL = 1
    """Focus points in horizontal coordinates, i.e. azimuth and elevation angles in radians."""
    
    CARTESIAN = 2
    """Focus points in Cartesian coordinates, i.e. xyz in m."""
    
    DEVICE = 3
    """Focus points considering peer devices."""
    
    
class BeamformerBase(ABC):
    """Base class for all beam steering precodings."""
    
    __operator: Optional[Operator]      # Reference to the operator the beamformer is attached to
    
    def __init__(self,
                 operator: Optional[Operator] = None) -> None:
        """Args:
        
            operator (Operator, optional):
                The operator this beamformer is attached to.
                By default, the beamformer is considered floating.
        """
        
        self.operator = operator
        
    @property
    def operator(self) -> Optional[Operator]:
        """The operator this beamformer is assigned to.
        
        Returns:
        
            Handle to the operator.
            `None` if the beamformer is considered floating.
        """
        
        return self.__operator
    
    @operator.setter
    def operator(self, value: Optional[Operator]) -> None:
        
        self.__operator = value
    
    
class TransmitBeamformer(BeamformerBase, ABC):
    """Base class for beam steering precodings during signal transmissions."""
    
    __focus_points: np.ndarray
    __focus_mode: FocusMode
    
    def __init__(self,
                 operator: Optional[Transmitter] = None) -> None:
        """Args:
        
            operator (Transmitter, optional):
                The operator this beamformer is attached to.
                By default, the beamformer is considered floating.
        """
        
        self.__focus_points = np.array([[0., 0.]], dtype=float)
        self.__focus_mode = FocusMode.SPHERICAL
        
        BeamformerBase.__init__(self, operator=operator)
        
    @abstractproperty
    def num_transmit_input_streams(self) -> int:
        """Number of input streams required by this beamformer.
        
        Returns:
        
            Number of input streams.
        """
        ...  # pragma no cover
        
    @abstractproperty
    def num_transmit_output_streams(self) -> int:
        """Number of output streams generated by this beamformer.
        
        Returns:
        
            Number of output streams.
        """
        ...  # pragma no cover
        
    @abstractproperty
    def num_transmit_focus_angles(self) -> int:
        """Number of required transmit focus angles.
        
        Returns:
        
            Number of focus angles.
        """
        ...  # pragma no cover

    @abstractmethod
    def _encode(self,
                samples: np.ndarray,
                carrier_frequency: float,
                focus_angles: np.ndarray) -> np.ndarray:
        """Encode signal streams for transmit beamforming.
        
        Args:
        
            samples (np.ndarray):
                Signal samples, first dimension being the number of transmit antennas, second the number of samples.
                
            carrier_frequency (float):
                The assumed carrier central frequency of the samples.
                
            focus_angles (np.ndarray):
                Focused angles of departure in radians.
                Two-dimensional numpy array with the first dimension representing the number of focus points
                and the second dimension of magnitude two being the azimuth and elevation angles, respectively.
        
            azimuth (float):
                Azimuth angle of departure in Radians.
                
            zenith (float):
                Zenith angle of departure in Radians.
        """
        ...  # pragma no cover
        
    @property
    def transmit_focus(self) -> Tuple[np.ndarray, FocusMode]:
        """Focus points of the beamformer during transmission.
        
        Returns:
    
            - Numpy array of focus points elevation and azimuth angles
            - Focus mode
        """
        
        return self.__focus_points, self.__focus_mode
    
    @transmit_focus.setter
    def transmit_focus(self, value: Union[np.ndarray, Tuple[np.ndarray, FocusMode]]) -> None:
        
        if not isinstance(value, tuple):
            value = (value, self.__focus_mode)
            
        if value[0].ndim != 2:
            raise ValueError("Focus must be a two-dimensional array")
        
        if value[0].shape[0] != self.num_transmit_focus_angles:
            raise ValueError(f"Focus requires {self.num_transmit_focus_angles} points, but {value[0].shape[0]} were provided")
            
        self.__focus_points = value[0]
        self.__focus_mode = value[1]
        
    def transmit(self, signal: Signal, focus: Optional[np.ndarray] = None) -> Signal:
        """Focus a signal model towards a certain target.
        
        Args:
        
            signal (Signal):
                The signal to be steered.
                
            focus (np.ndarray, optional):
                Focus point of the steered signal power.
        
        Returns:
        
            Samples of the focused signal.
        """
        
        if self.operator is None:
            raise FloatingError("Unable to steer a signal over a floating beamformer")
        
        if self.operator.device is None:
            raise FloatingError("Unable to steer a signal over a floating operator")
        
        if signal.num_streams != self.num_transmit_input_streams:
            raise RuntimeError(f"The provided signal contains {signal.num_streams}, but the beamformer requires {self.num_transmit_input_streams} streams")

        carrier_frequency = signal.carrier_frequency
        samples = signal.samples.copy()
        focus = self.transmit_focus[0] if focus is None else focus
    
        steered_samples = self._encode(samples, carrier_frequency, focus)
        return Signal(steered_samples, sampling_rate=signal.sampling_rate, carrier_frequency=signal.carrier_frequency)
        
    
class ReceiveBeamformer(ABC):
    """Base class for beam steering precodings during signal receptions."""
        
    __focus_points: np.ndarray
    __focus_mode: FocusMode
    
    def __init__(self,
                 operator: Optional[Receiver] = None) -> None:
        """Args:
        
            operator (Receiver, optional):
                The operator this beamformer is attached to.
                By default, the beamformer is considered floating.
        """      
          
        self.__focus_points = np.array([[0., 0.]], dtype=float)
        self.__focus_mode = FocusMode.SPHERICAL
        self.probe_focus_points = np.zeros((1, self.num_receive_focus_angles, 2), dtype=float)
        
        BeamformerBase.__init__(self, operator=operator)
        
    @abstractproperty
    def num_receive_input_streams(self) -> int:
        """Number of input streams required by this beamformer.
        
        Returns:
        
            Number of input streams.
        """
        ...  # pragma no cover
        
    @abstractproperty
    def num_receive_output_streams(self) -> int:
        """Number of output streams generated by this beamformer.
        
        Returns:
        
            Number of output streams.
        """
        ...  # pragma no cover
        
        
    @abstractproperty
    def num_receive_focus_angles(self) -> int:
        """Number of required transmit focus angles.
        
        Returns:
        
            Number of focus angles.
        """
        ...  # pragma no cover

    @abstractmethod
    def _decode(self,
                samples: np.ndarray,
                carrier_frequency: float,
                angles: np.ndarray) -> np.ndarray:
        """Decode signal streams for receive beamforming.
        
        Args:
        
            samples (np.ndarray):
                Signal samples, first dimension being the number of transmit antennas, second the number of samples.
                
            carrier_frequency (float):
                The assumed carrier central frequency of the samples.
        
            angles: (np.ndarray):
                Spherical coordinate system angles of arrival in radians.
                A three-dimensional numpy array with the first dimension representing the number of angles,
                and the third dimension containing the azimuth and zenith angle in radians, respectively.
        
        Returns:
        
            Stream samples of the focused signal towards all focus points.
            A three-dimensional numpy array with the first dimension representing the number of focus points,
            the second dimension the number of returned streams and the third dimension the amount of samples.
        """
        ...  # pragma no cover
        
    @property
    def receive_focus(self) -> Tuple[np.ndarray, FocusMode]:
        """Focus points of the beamformer during reception.
        
        Returns:
    
            - Numpy array of focus points elevation and azimuth angles
            - Focus mode
        """
        
        return self.__focus_points, self.__focus_mode
    
    @receive_focus.setter
    def receive_focus(self, value: Union[np.ndarray, Tuple[np.ndarray, FocusMode]]) -> None:
        
        if not isinstance(value, tuple):
            value = (value, self.__focus_mode)
            
        if value[0].ndim != 2:
            raise ValueError("Focus must be a two-dimensional array")
        
        if value[0].shape[0] != self.num_receive_focus_angles:
            raise ValueError(f"Focus requires {self.num_receive_focus_angles} points, but {value[0].shape[0]} were provided")
            
        self.__focus_points = value[0]
        self.__focus_mode = value[1]
    
    def receive(self,
                signal: Signal,
                focus_points: Optional[np.ndarray] = None,
                focus_mode: FocusMode = FocusMode.SPHERICAL) -> Signal:
        """Focus a signal model towards a certain target.
        
        Args:
        
            signal (Signal):
                The signal to be steered.
                
            focus_points (np.ndarray, optional):
                Focus point of the steered signal power.
                Two-dimensional numpy array with the first dimension representing the number of points
                and the second dimension representing the point values.
        
            focus_mode (FocusMode, optional):
                Type of focus points.
                By default, spherical coordinates are expected.
        
        Returns:
        
            Signal focused towards the requested focus points.
        """
        
        if self.operator is None:
            raise FloatingError("Unable to steer a signal over a floating beamformer")
        
        if self.operator.device is None:
            raise FloatingError("Unable to steer a signal over a floating operator")

        if signal.num_streams != self.num_receive_input_streams:
            raise RuntimeError(f"The provided signal contains {signal.num_streams}, but the beamformer requires {self.num_receive_input_streams} streams")

        carrier_frequency = signal.carrier_frequency
        samples = signal.samples.copy()
        focus_angles = self.receive_focus[0][np.newaxis, ::] if focus_points is None else focus_points[np.newaxis, ::]
    
        beamformed_samples = self._decode(samples, carrier_frequency, focus_angles)
        return Signal(beamformed_samples[0, ::], signal.sampling_rate)
    
    @property
    def probe_focus_points(self) -> np.ndarray:
        
        return self.__probe_focus_points
    
    @probe_focus_points.setter
    def probe_focus_points(self, value: np.ndarray) -> None:
        
        # Expand points by new dimension if only a single focus tuple was requested
        if value.ndim == 2:
            value = value[np.newaxis, ::]
        
        if value.ndim != 3:
            raise RuntimeError("Probing focus points must be a three-dimensional array")
        
        if value.shape[1] != self.num_receive_focus_angles:
            raise ValueError(f"Focus requires {self.num_receive_focus_angles} points, but {value.shape[1]} were provided")

        self.__probe_focus_points = value

    def probe(self,
               signal: Signal,
               focus_points: Optional[np.ndarray] = None) -> np.ndarray:
        """Focus a signal model towards a certain directions of interest.
        
        Args:
        
            signal (Signal):
                The signal to be steered.
                
            focus_points (np.ndarray, optional):
                Focus point of the steered signal power.
                Two-dimensional numpy array with the first dimension representing the number of points
                and the second dimension representing the point values.
        
        Returns:
        
            Stream samples of the focused signal towards all focus points.
            A three-dimensional numpy array with the first dimension representing the number of focus points,
            the second dimension the number of returned streams and the third dimension the amount of samples.
        """
        
        focus_points = self.probe_focus_points if focus_points is None else focus_points

        if self.operator is None:
            raise FloatingError("Unable to steer a signal over a floating beamformer")
        
        if self.operator.device is None:
            raise FloatingError("Unable to steer a signal over a floating operator")

        if signal.num_streams != self.num_receive_input_streams:
            raise RuntimeError(f"The provided signal contains {signal.num_streams}, but the beamformer requires {self.num_receive_input_streams} streams")

        carrier_frequency = signal.carrier_frequency
        samples = signal.samples.copy()
    
        return self._decode(samples, carrier_frequency, focus_points)
