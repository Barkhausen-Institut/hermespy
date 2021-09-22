from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, TYPE_CHECKING, Optional, Type
from ruamel.yaml import SafeConstructor, SafeRepresenter, Node
import numpy as np

from parameters_parser.parameters_waveform_generator import ParametersWaveformGenerator
from channel.channel import Channel

if TYPE_CHECKING:
    from modem import Modem


class WaveformGenerator(ABC):
    """Implements an abstract waveform generator.

    Implementations for specific technologies should inherit from this class.
    """

    __modem: Optional[Modem]
    __sampling_rate: float

    def __init__(self,
                 modem: Modem = None,
                 sampling_rate: float = None) -> None:
        """Object initialization.

        Args:
            modem (Modem):
                A modem this generator is attached to.
                By default, the generator is considered to be floating.

            sampling_rate (float):
                Rate at which the generated signals are sampled.
        """

        self.__modem = None
        self.__sampling_rate = 1e3

        if modem is not None:
            self.modem = modem

        if sampling_rate is not None:
            self.sampling_rate = sampling_rate

        #self.sampling_rate = param.sampling_rate
        #self._samples_in_frame: int = None
        # overhead to account for possible filtering effects (may overlap with following frames)
        #self._samples_overhead_in_frame = 0

    @classmethod
    def to_yaml(cls: Type[WaveformGenerator], representer: SafeRepresenter, node: WaveformGenerator) -> Node:
        """Serialize an `WaveformGenerator` object to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (WaveformGenerator):
                The `WaveformGenerator` instance to be serialized.

        Returns:
            Node:
                The serialized YAML node
        """

        state = {
            "sampling_rate": node.__sampling_rate
        }

        return representer.represent_mapping(cls.yaml_tag, state)

    @classmethod
    def from_yaml(cls: Type[WaveformGenerator], constructor: SafeConstructor, node: Node) -> WaveformGenerator:
        """Recall a new `WaveformGenerator` instance from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `WaveformGenerator` serialization.

        Returns:
            WaveformGenerator:
                Newly created `WaveformGenerator` instance.
        """

        state = constructor.construct_mapping(node)
        return cls(**state)

    @property
    @abstractmethod
    def samples_in_frame(self) -> int:
        """The number of discrete samples per generated frame.

        Returns:
            int:
                The number of samples.
        """
        pass

    @property
    def max_frame_length(self) -> float:
        """float: Maximum length of a data frame (in seconds)"""
        return (self.samples_in_frame + self._samples_overhead_in_frame) / self.sampling_rate

    @property
    def modem(self) -> Modem:
        """Access the `Modem` this waveform generator is attached to.

        Returns:
            Modem:
                Handle to the `Modem`.
                None if the generator is floating.
        """

        return self.__modem

    @abstractmethod
    def get_bit_energy(self) -> float:
        """Returns the theoretical average (discrete-time) bit energy of the modulated signal.

        Energy of signal x[k] is defined as \\sum{|x[k]}^2
        Only data bits are considered, i.e., reference, guard intervals are ignored.
        """
        pass

    @property
    @abstractmethod
    def symbol_energy(self) -> float:
        """The theoretical average symbol (discrete-time) energy of the modulated signal.

        Energy of signal x[k] is defined as \\sum{|x[k]}^2
        Only data bits are considered, i.e., reference, guard intervals are ignored.

        Returns:
            The average symbol energy in UNIT.
        """
        pass

    @abstractmethod
    def get_power(self) -> float:
        """Returns the theoretical average symbol (unitless) power,

        Power of signal x[k] is defined as \\sum_{k=1}^N{|x[k]}^2 / N
        Power is the average power of the data part of the transmitted frame, i.e., bit energy x raw bit rate
        """

        return 0.0

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

    @property
    def sampling_rate(self) -> float:
        """Access the configured sampling rate.

        Returns:
            float:
                The current sampling rate.
        """

        return self.__sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, sampling_rate: float) -> None:
        """Modify the sampling rate configuration.

        Args:
            sampling_rate (float):
                The new sampling rate.

        Raises:
            ValueError:
                If the sampling rate is smaller or equal to zero.
        """

        if sampling_rate <= 0.0:
            raise ValueError("Sampling rate must be greater than zero")

        self.__sampling_rate = sampling_rate

    @property
    def modem(self) -> Modem:
        """Access the modem this generator is attached to.

        Returns:
            Modem:
                A handle to the modem.

        Raises:
            RuntimeError:
                If this waveform generator is not attached to a modem.
        """

        if self.__modem is None:
            raise RuntimeError("This waveform generator is not attached to any modem")

        return self.__modem

    @modem.setter
    def modem(self, modem: Modem) -> None:
        """Modify the modem this generator is attached to.

        Args:
            modem (Modem):
                Handle to a modem.

        Raises:
            RuntimeError:
                If the `modem` does not reference this generator.
        """

        if modem.waveform_generator is not self:
            raise RuntimeError("Invalid modem attachment routine")

        self.__modem = modem
