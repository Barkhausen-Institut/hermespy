# -*- coding: utf-8 -*-
"""
===================
Matched Filter JCAS
===================
"""

from __future__ import annotations
from typing import Optional, Type, Tuple

import numpy as np
from ruamel.yaml import SafeConstructor, MappingNode
from scipy.constants import speed_of_light
from scipy.signal import correlate, correlation_lags

from hermespy.core import Signal
from hermespy.modem import Modem, Symbols, WaveformGenerator
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


class MatchedFilterJcas(Modem, Radar):
    """Joint Communication and Sensing Operator.
    
    A combination of communication and sensing operations.
    Senses the enviroment via a correlatiom-based time of flight estimation of transmitted waveforms.
    """

    yaml_tag = u'MatchedFilterJcas'
    """YAML serialization tag."""
    
    __transmission: Optional[Signal]        # Most recent transmission
    __sampling_rate: Optional[float]        # The specific required sampling rate
    __max_range: float                      # Maximally detectable range
    
    def __init__(self, 
                 max_range: float) -> None:
        """
        Args:
        
            max_range (float):
                Maximally detectable range in m.
        """
        
        self.__sampling_rate = None
        self.max_range = max_range
        
        Modem.__init__(self)
        Radar.__init__(self)

    def transmit(self, duration: float = 0) -> Tuple[Signal, Symbols, np.ndarray]:
        
        # Cache the recently transmitted waveform for correlation during reception
        signal, symbols, bits = super().transmit(duration)
        
        # Resample the signal for an improved range resolution visualization
        signal = signal.resample(self.sampling_rate)
        
        # Cache the resampled transmission
        self.__transmission = signal
        if self._transmitter.attached:
            self._transmitter.slot.add_transmission(self._transmitter, signal)

        return signal, symbols, bits
        
    def receive(self) -> Tuple[Signal, Symbols, np.ndarray, RadarCube]:
        
        # There must be a recent transmission being cached in order to correlate
        if self.__transmission is None:
            raise RuntimeError("Receiving from a matched filter joint must be preceeded by a transmission")
        
        # Receive information
        modem_signal, symbols, bits = Modem.receive(self)
        
        # Re-sample communication waveform
        signal = self._receiver.signal.resample(self.sampling_rate)
        
        resolution = self.range_resolution
        num_propagated_samples = int(2 * self.max_range / resolution)
        
        # Append additional samples if the signal is too short
        required_num_received_samples = self.__transmission.num_samples + num_propagated_samples
        if signal.num_samples < required_num_received_samples:
            signal.append_samples(Signal(np.zeros((1, required_num_received_samples - signal.num_samples), dtype=complex), self.sampling_rate, signal.carrier_frequency))

        # Remove possible overhead samples if signal is too long
        # resampled_signal.samples = resampled_signal.samples[:, :num_samples]
        
        correlation = abs(correlate(signal.samples, self.__transmission.samples, mode='valid', method='fft').flatten()) / self.__transmission.num_samples
        lags = correlation_lags(signal.num_samples, self.__transmission.num_samples, mode='valid')

        # Append zeros for correct depth estimation
        #num_appended_zeros = max(0, num_samples - resampled_signal.num_samples)
        #correlation = np.append(correlation, np.zeros(num_appended_zeros))

        angle_bins = np.array([0.])
        velocity_bins = np.array([0.])
        range_bins = .5 * lags * resolution
        cube_data = np.array([[correlation]], dtype=float)
        cube = RadarCube(cube_data, angle_bins, velocity_bins, range_bins)

        return signal, symbols, bits, cube
        
    @property
    def sampling_rate(self) -> float:
        
        modem_sampling_rate = self.waveform_generator.sampling_rate
        
        if self.__sampling_rate is None:
            return modem_sampling_rate
        
        return max(modem_sampling_rate, self.__sampling_rate)
        
    @sampling_rate.setter
    def sampling_rate(self, value: Optional[float]) -> None:
        
        if value is None:
            
            self.__sampling_rate = None
            return
        
        if value <= 0.:
            raise ValueError("Sampling rate must be greater than zero")
        
        self.__sampling_rate = value
        
    @property
    def range_resolution(self) -> float:
        """Resolution of the Range Estimation.
        
        Returns:
            float:
                Resolution in m.
                
        Raises:
        
            ValueError:
                If the range resolution is smaller or equal to zero.
        """
        
        return speed_of_light / self.sampling_rate
    
    @range_resolution.setter
    def range_resolution(self, value: float) -> None:
        
        if value <= 0.:
            raise ValueError("Range resolution must be greater than zero")
        
        self.sampling_rate = speed_of_light / value
    
    @property
    def max_range(self) -> float:
        """Maximally Estimated Range.
        
        Returns:
            The maximum range in m.
            
        Raises:
        
            ValueError:
                If `max_range` is smaller or equal to zero.
        """
        
        return self.__max_range
        
    @max_range.setter
    def max_range(self, value) -> None:
        
        if value <= 0.:
            raise ValueError("Maximum range must be greater than zero")
        
        self.__max_range = value

    @classmethod
    def from_yaml(cls: Type[MatchedFilterJcas], constructor: SafeConstructor, node: MappingNode) -> MatchedFilterJcas:
        """Recall a new `MatchedFilterJcas` class instance from YAML.

        Args:

            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (MappingNode):
                YAML node representing the `MatchedFilterJcas` serialization.

        Returns:

            MatchedFilterJcas:
                Newly created serializable instance.
        """

        state = constructor.construct_mapping(node)
        return cls.InitializationWrapper(state)
