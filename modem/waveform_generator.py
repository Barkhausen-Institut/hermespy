# -*- coding: utf-8 -*-
"""Prototype for Waveform Generation Modeling."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, TYPE_CHECKING, Optional, Type
from ruamel.yaml import SafeConstructor, SafeRepresenter, Node
from math import floor
import numpy as np

if TYPE_CHECKING:
    from modem import Modem

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Tobias Kronauer"
__email__ = "tobias.kronauer@barkhauseninstitut.org"
__status__ = "Prototype"


class WaveformGenerator(ABC):
    """Implements an abstract waveform generator.

    Implementations for specific technologies should inherit from this class.
    """

    yaml_tag: str = "Waveform"
    symbol_type: np.dtype = complex
    __modem: Optional[Modem]
    __oversampling_factor: int
    __modulation_order: int

    def __init__(self,
                 modem: Modem = None,
                 oversampling_factor: int = None,
                 modulation_order: int = None) -> None:
        """Object initialization.

        Args:
            modem (Modem, optional):
                A modem this generator is attached to.
                By default, the generator is considered to be floating.

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
    def bits_per_symbol(self) -> int:
        """Number of bits transmitted per modulated symbol.

        Returns:
            int: Number of bits per symbol
        """

        return 2 ** self.__modulation_order

    @property
    @abstractmethod
    def bits_per_frame(self) -> int:
        """Number of bits required to generate a single data frame.

        Returns:
            int: Number of bits
        """
        ...

    @property
    @abstractmethod
    def symbols_per_frame(self) -> int:
        """Number of dat symbols per transmitted frame.

        Returns:
            int: Number of data symbols
        """
        ...

    @property
    def frame_duration(self) -> float:
        """Length of one data frame in seconds.

        Returns:
            float: Frame length in seconds.
        """

        return self.samples_in_frame / self.modem.scenario.sampling_rate

    @property
    def max_frame_duration(self) -> float:
        """float: Maximum length of a data frame (in seconds)"""

        # TODO: return (self.samples_in_frame + self._samples_overhead_in_frame) / self.sampling_rate
        return self.samples_in_frame / self.modem.scenario.sampling_rate

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
    def map(self, data_bits: np.ndarray) -> np.ndarray:
        """Map a stream of bits to data symbols.

        Args:
            data_bits (np.ndarray):
                Vector containing a sequence of L hard data bits to be mapped onto data symbols.

        Returns:
            np.ndarray:
                Vector containing the resulting sequence of K data symbols.
                In general, K is less or equal to L.
        """
        ...

    @abstractmethod
    def unmap(self, data_symbols: np.ndarray) -> np.ndarray:
        """Map a stream of data symbols to data bits.

        Args:
            data_symbols (np.ndarray):
                Vector containing a sequence of K data symbols to be mapped onto bit sequences.

        Returns:
            np.ndarray:
                Vector containing the resulting sequence of L data bits
                In general, L is greater or equal to K.
        """
        ...

    @abstractmethod
    def modulate(self, data_symbols: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """Modulate a stream of data symbols to a base-band signal.

        Args:

            data_symbols (np.ndarray):
                Vector of data symbols to be modulated.

            timestamps (np.ndarray):
                Vector if sample times in seconds, at which the resulting base-band signal should be sampled.

        Returns:
            np.ndarray:
                Complex-valued vector containing samples of the modulated base-band signals.
        """
        ...

    # Hint: Channel propagation occurs here

    def synchronize(self, signal: np.ndarray, stream_response: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulates time-synchronization at the receiver-side.

        Sorts signal-sections into frames in time-domain.

        Args:

            signal (np.ndarray):
                Vector of complex base-band signal samples of a single input stream with `num_samples` entries.

            channel_response (np.ndarray):
                Vector of channel impulse responses. ToDo: Add domains.

        Returns:
            Tuple[np.ndarray, np.ndarry]: Tuple of signal samples and channel responses sorted into frames.

        Raises:
            ValueError: If the length of `signal` and the first dimension of `channel_response` is not identical.
        """

        if len(signal) != (stream_response.shape[0] + stream_response.shape[1] - 1):
            raise ValueError("Signal length and the first response dimension must be matching")

        samples_per_frame = self.samples_in_frame
        num_frames = int(floor(len(signal) / samples_per_frame))

        frames = np.empty((num_frames, samples_per_frame), dtype=complex)
        frame_responses = np.empty((num_frames, samples_per_frame), dtype=complex)

        # By default, there is no synchronization, i.e. we assume the first signal is also the first sample of
        # the first frame. ToDo: Check with André how to implement general equalization
        for f in range(num_frames):

            # ToDo: This currently does not account for delay overhead....
            frames[f, :] = signal[f*samples_per_frame:(f+1)*samples_per_frame]
            frame_responses[f, :] = np.sum(stream_response[f*samples_per_frame:(f+1)*samples_per_frame, :], axis=1)

        return frames, frame_responses

    @abstractmethod
    def demodulate(self,
                   signal: np.ndarray,
                   impulse_response: np.ndarray,
                   noise_variance: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Demodulate a base-band signal stream to data symbols.

        Args:

            signal:
                Vector of complex-valued base-band samples of a modulated signal.

            impulse_response:
                Vector of complex valued base-band channel impulse responses of a modulated signal.

            noise_variance:
                Variance of the thermal noise introduced during reception.

        Returns:
            (np.ndarray, np.ndarray, np.ndarray):
                Tuple of 3 vectors of equal length:
                The demodulated data symbols, their channel estimates and their noise variance.
        """
        ...

    @property
    @abstractmethod
    def bandwidth(self) -> float:
        """Bandwidth of the frame generated by this generator.

        Used to estimate the minimal sampling frequency required to adequately simulate the scenario.

        Returns:
            float: Bandwidth in Hz.
        """
        ...

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
