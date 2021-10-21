from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, TYPE_CHECKING, Optional, Type
from ruamel.yaml import SafeConstructor, SafeRepresenter, Node
import numpy as np

if TYPE_CHECKING:
    from modem import Modem


class WaveformGenerator(ABC):
    """Implements an abstract waveform generator.

    Implementations for specific technologies should inherit from this class.
    """

    yaml_tag: str = "Waveform"
    __modem: Optional[Modem]
    __sampling_rate: Optional[float]
    __oversampling_factor: int
    __modulation_order: int

    def __init__(self,
                 modem: Modem = None,
                 sampling_rate: float = None,
                 oversampling_factor: int = None,
                 modulation_order: int = None) -> None:
        """Object initialization.

        Args:
            modem (Modem, optional):
                A modem this generator is attached to.
                By default, the generator is considered to be floating.

            sampling_rate (float, optional):
                Rate at which the generated signals are sampled.

            oversampling_factor (int, optional):
                The factor at which the simulated signal is oversampled.

            modulation_order (int, optional):
                Order of modulation.
                Must be a non-negative power of two.
        """

        # Default parameters
        self.__modem = None
        self.__sampling_rate = None
        self.__oversampling_factor = 4
        self.__modulation_order = 256

        if modem is not None:
            self.modem = modem

        if sampling_rate is not None:
            self.sampling_rate = sampling_rate

        if oversampling_factor is not None:
            self.oversampling_factor = oversampling_factor

        if modulation_order is not None:
            self.modulation_order = modulation_order

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
            "sampling_rate": node.__sampling_rate,
            "oversampling_factor": node.__oversampling_factor,
            "modulation_order": node.__modulation_order,
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
    def oversampling_factor(self) -> int:
        """Access the oversampling factor.

        Returns:
            int:
                The oversampling factor.
        """

        return self.__oversampling_factor

    @oversampling_factor.setter
    def oversampling_factor(self, factor: int) -> None:
        """Modify the oversampling factor.

        Args:
            factor (int):
                The new oversampling factor.

        Raises:
            ValueError:
                If the oversampling `factor` is less than one.
        """

        if factor < 1:
            raise ValueError("The oversampling factor must be greater or equal to one")

        self.__oversampling_factor = factor

    @property
    def modulation_order(self) -> int:
        """Access the modulation order.

        Returns:
            int:
                The modulation order.
        """

        return self.__modulation_order

    @modulation_order.setter
    def modulation_order(self, order: int) -> None:
        """Modify the modulation order.

        Must be a positive power of two.

        Args:
            order (int):
                The new modulation order.

        Raises:
            ValueError:
                If `order` is not a positive power of two.
        """

        if order <= 0 or (order & (order - 1)) != 0:
            raise ValueError("Modulation order must be a positive power of two")

        self.__modulation_order = order

    @property
    @abstractmethod
    def bits_per_frame(self) -> int:
        """Number of bits required to generate a single data frame.

        Returns:
            int: Number of bits
        """
        ...

    @property
    def frame_duration(self) -> float:
        """Length of one data frame in seconds.

        Returns:
            float: Frame length in seconds.
        """

        return self.samples_in_frame / self.sampling_rate

    @property
    def max_frame_duration(self) -> float:
        """float: Maximum length of a data frame (in seconds)"""

        # TODO: return (self.samples_in_frame + self._samples_overhead_in_frame) / self.sampling_rate
        return self.samples_in_frame / self.sampling_rate

    @property
    @abstractmethod
    def bit_energy(self) -> float:
        """Returns the theoretical average (discrete-time) bit energy of the modulated signal.

        Energy of signal x[k] is defined as \\sum{|x[k]}^2
        Only data bits are considered, i.e., reference, guard intervals are ignored.
        """
        ...

    @property
    @abstractmethod
    def symbol_energy(self) -> float:
        """The theoretical average symbol (discrete-time) energy of the modulated signal.

        Energy of signal x[k] is defined as \\sum{|x[k]}^2
        Only data bits are considered, i.e., reference, guard intervals are ignored.

        Returns:
            The average symbol energy in UNIT.
        """
        ...

    @property
    @abstractmethod
    def power(self) -> float:
        """Returns the theoretical average symbol (unitless) power,

        Power of signal x[k] is defined as \\sum_{k=1}^N{|x[k]|}^2 / N
        Power is the average power of the data part of the transmitted frame, i.e., bit energy x raw bit rate
        """
        ...

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
        ...

    @abstractmethod
    def receive_frame(self,
                      rx_signal: np.ndarray,
                      timestamp_in_samples: int,
                      noise_var: float) -> Tuple[np.array, np.ndarray]:
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
            (np.ndarray, np.ndarray):
                `np.array`: Detected bits.
                `np.ndarray`: remeaining received signal corresponding to the
                following frames.
        """
        ...

    @property
    def sampling_rate(self) -> float:
        """Access the configured sampling rate.

        Returns:
            float:
                The configured sampling rate, alternatively the attached modem's sampling rate in Hz.

        Raises:
            RuntimeError:
                If the sampling rate is not configured and the generator is floating.
        """

        if self.__sampling_rate is not None:
            return self.__sampling_rate

        if self.__modem is not None:
            return self.__modem.sampling_rate

        raise RuntimeError("Tried to access the unknown sampling rate of a floating generator")

    @sampling_rate.setter
    def sampling_rate(self, sampling_rate: Optional[float]) -> None:
        """Modify the sampling rate configuration.

        Args:
            sampling_rate (Optional[float]):
                The new sampling rate.
                None, if the modem's sampling rate is identical.

        Raises:
            ValueError:
                If the sampling rate is smaller or equal to zero.
        """

        if sampling_rate is None:
            self.__sampling_rate = sampling_rate

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
    def modem(self, handle: Modem) -> None:
        """Modify the modem this generator is attached to.

        Args:
            handle (Modem):
                Handle to a modem.

        Raises:
            RuntimeError:
                If the `modem` does not reference this generator.
        """

        if handle.waveform_generator is not self:
            raise RuntimeError("Invalid modem attachment routine")

        self.__modem = handle
