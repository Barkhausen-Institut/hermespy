from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from parameters_parser.parameters_waveform_generator import ParametersWaveformGenerator
from channel.channel import Channel


class WaveformGenerator(ABC):
    """Implements an abstract waveform generator.

    Implementations for specific technologies should inherit from this class.
    """

    def __init__(self, param: ParametersWaveformGenerator) -> None:
        self.sampling_rate = param.sampling_rate
        self._channel: Channel = None
        self._samples_in_frame: int = None
        # overhead to account for possible filtering effects (may overlap with following frames)
        self._samples_overhead_in_frame = 0

    @property
    def samples_in_frame(self) -> int:
        """int: samples contained in current frame."""
        return self._samples_in_frame

    @property
    def max_frame_length(self) -> float:
        """float: Maximum length of a data frame (in seconds)"""
        return (self.samples_in_frame + self._samples_overhead_in_frame) / self.sampling_rate

    @abstractmethod
    def get_bit_energy(self) -> float:
        """Returns the theoretical average (discrete-time) bit energy of the modulated signal.

        Energy of signal x[k] is defined as \\sum{|x[k]}^2
        Only data bits are considered, i.e., reference, guard intervals are ignored.
        """
        pass

    @abstractmethod
    def get_symbol_energy(self) -> float:
        """Returns the theoretical average symbol (discrete-time) energy of the modulated signal.

        Energy of signal x[k] is defined as \\sum{|x[k]}^2
        Only data bits are considered, i.e., reference, guard intervals are ignored.
        """
        pass

    @abstractmethod
    def get_power(self) -> float:
        """Returns the theoretical average symbol (unitless) power,

        Power of signal x[k] is defined as \\sum_{k=1}^N{|x[k]}^2 / N
        Power is the average power of the data part of the transmitted frame, i.e., bit energy x raw bit rate
        """
        pass

    @abstractmethod
    def create_frame(self, old_timestamp: int,
                     data_bits: np.array) -> Tuple[np.ndarray, int, int]:
        """Creates a new transmission frame.

        Args:
            old_timestamp (int): Initial timestamp (in sample number) of new frame.
            data_bits (np.array):
                Flattened blocks, whose bits are supposed to fit into this frame.

        Returns:
            (np.ndarray, int, int):
                `np.ndarray`: Baseband complex samples of transmission frame.
                `int`: timestamp(in sample number) of frame end.
                `int`: Sample number of first sample in frame (to account for possible filtering).
        """
        pass

    @abstractmethod
    def receive_frame(self,
                      rx_signal: np.ndarray,
                      timestamp_in_samples: int,
                      noise_var: float) -> Tuple[List[np.array], np.ndarray]:
        """Receives and detects the bits from a new received frame.


        The method receives the whole received signal 'rx_signal' from the drop,
        extracts the signal corresponding to the current frame, and detects the
        transmitted bits, which are returned in 'bits'.

        Args:
            rx_signal (np.ndarray):
                Received signal in drop. Rows denoting antennas and columns
                being the samples.
            timestamp_in_samples(int):
                First timestamp in samples of this frame-
            noise_var(float):
                ES/NO required for equalization.

        Returns:
            (List[np.narray], np.ndarray):
                `List[np.array]`: Detected bits as a list of data blocks.
                `np.ndarray`: remeaining received signal corresponding to the
                following frames.
        """
        pass

    def set_channel(self, channel: Channel) -> None:
        """Associates a given propagation channel to this modem.

        It can be used to obtain perfect channel state information for
        detection/precoding algorithms.
        """
        self._channel = channel
