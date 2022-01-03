# -*- coding: utf-8 -*-
"""Prototype for Waveform Generation Modeling."""

from __future__ import annotations
from abc import ABC, abstractmethod
from math import floor, sqrt
from typing import Tuple, TYPE_CHECKING, Optional, Type, List

import numpy as np
from ruamel.yaml import SafeConstructor, SafeRepresenter, Node

from hermespy.channel import ChannelStateInformation
from hermespy.signal import Signal

if TYPE_CHECKING:
    from hermespy.modem import Modem

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.5"
__maintainer__ = "Tobias Kronauer"
__email__ = "tobias.kronauer@barkhauseninstitut.org"
__status__ = "Prototype"


class Synchronization(ABC):
    """Abstract base class for synchronization routines of waveform generators."""

    __waveform_generator: Optional[WaveformGenerator]       # Waveform generator this routine is attached to

    def __init__(self,
                 waveform_generator: Optional[WaveformGenerator] = None) -> None:
        """
        Args:
            waveform_generator (WaveformGenerator, optional):
                The waveform generator this synchronization routine is attached to.
        """

        self.__waveform_generator = waveform_generator

    @property
    def waveform_generator(self) -> Optional[WaveformGenerator]:
        """Waveform generator this synchronization routine is attached to.

        Returns:
            Optional[WaveformGenerator]:
                Handle to the waveform generator. None if the synchronization routine is floating.
        """

        return self.__waveform_generator

    @waveform_generator.setter
    def waveform_generator(self, value: Optional[WaveformGenerator]) -> None:
        """Set waveform generator this synchronization routine is attached to."""

        if self.__waveform_generator is not None:
            raise RuntimeError("Error trying to re-attach already attached synchronization routine.")

        self.__waveform_generator = value

    def synchronize(self,
                    signal: np.ndarray,
                    channel_state: ChannelStateInformation) -> List[Tuple[np.ndarray, ChannelStateInformation]]:
        """Simulates time-synchronization at the receiver-side.

        Sorts base-band signal-sections into frames in time-domain.

        Args:

            signal (np.ndarray):
                Vector of complex base-band samples of a single input stream with `num_samples` entries.

            channel_state (ChannelStateInformation):
                State of the wireless transmission channel over which `signal` has been propagated.

        Returns:
            List[Tuple[np.ndarray, ChannelStateInformation]]:
                Tuple of `signal` samples and channel transformations sorted into frames

        Raises:

            ValueError:
                If the number of received streams in `channel_state` does not equal one.
                If the length of `signal` and the number of samples in `channel_state` are not identical.

            RuntimeError:
                If the synchronization routine is floating
        """

        if self.__waveform_generator is None:
            raise RuntimeError("Trying to synchronize with a floating synchronization routine")

        if len(signal) != channel_state.num_samples + channel_state.num_delay_taps - 1:
            raise ValueError("Base-band signal and channel state contain a different amount of samples")

        if channel_state.num_receive_streams != 1:
            raise ValueError("Channel state during synchronization may only contain a single receive stream")

        samples_per_frame = self.__waveform_generator.samples_in_frame
        num_frames = int(floor(len(signal) / samples_per_frame))

        # Slice signals and channel state information into frame-sized portions
        # Default synchronization does NOT account for possible delays,
        # i.e. assume the the first base-band signal's sample to also be the first frame's initial sample
        synchronized_frames: List[Tuple[np.ndarray, ChannelStateInformation]] = []
        for frame_idx in range(num_frames):

            frame_samples = signal[frame_idx*samples_per_frame:(1+frame_idx)*samples_per_frame]
            frame_channel_state = channel_state[:, :,  frame_idx*samples_per_frame:(1+frame_idx)*samples_per_frame, :]
            synchronized_frames.append((frame_samples, frame_channel_state))

        return synchronized_frames


class WaveformGenerator(ABC):
    """Implements an abstract waveform generator.

    Implementations for specific technologies should inherit from this class.
    """

    yaml_tag: str = "Waveform"
    """YAML serialization tag."""

    synchronization: Synchronization
    """Synchronization routine."""

    symbol_type: np.dtype = complex
    """Symbol type."""

    __modem: Optional[Modem]            # Modem this waveform generator is attached to
    __oversampling_factor: int          # Oversampling factor
    __modulation_order: int             # Cardinality of the set of communication symbols

    def __init__(self,
                 modem: Optional[Modem] = None,
                 oversampling_factor: int = 1,
                 modulation_order: int = 16) -> None:
        """Waveform Generator initialization.

        Args:
            modem (Modem, optional):
                A modem this generator is attached to.
                By default, the generator is considered to be floating.

            oversampling_factor (int, optional):
                The factor at which the simulated baseband_signal is oversampled.

            modulation_order (int, optional):
                Order of modulation.
                Must be a non-negative power of two.
        """

        # Default parameters
        self.__modem = None
        self.oversampling_factor = oversampling_factor
        self.modulation_order = modulation_order
        self.synchronization = Synchronization(self)

        if modem is not None:
            self.modem = modem

        if oversampling_factor is not None:
            self.oversampling_factor = oversampling_factor

        if modulation_order is not None:
            self.modulation_order = modulation_order

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

        return int(sqrt(self.__modulation_order))

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

        return self.samples_in_frame / self.sampling_rate

    @property
    @abstractmethod
    def bit_energy(self) -> float:
        """Returns the theoretical average (discrete-time) bit energy of the modulated baseband_signal.

        Energy of baseband_signal x[k] is defined as \\sum{|x[k]}^2
        Only data bits are considered, i.e., reference, guard intervals are ignored.
        """
        ...

    @property
    @abstractmethod
    def symbol_energy(self) -> float:
        """The theoretical average symbol (discrete-time) energy of the modulated baseband_signal.

        Energy of baseband_signal x[k] is defined as \\sum{|x[k]}^2
        Only data bits are considered, i.e., reference, guard intervals are ignored.

        Returns:
            The average symbol energy in UNIT.
        """
        ...

    @property
    @abstractmethod
    def power(self) -> float:
        """Returns the theoretical average symbol (unitless) power,

        Power of baseband_signal x[k] is defined as \\sum_{k=1}^N{|x[k]|}^2 / N
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
    def modulate(self, data_symbols: np.ndarray) -> Signal:
        """Modulate a stream of data symbols to a base-band signal containing a single data frame.

        Args:

            data_symbols (np.ndarray):
                Vector of data symbols to be modulated.

        Returns:
            Signal: Signal model of a single modulate data frame.
        """
        ...

    # Hint: Channel propagation occurs here

    @abstractmethod
    def demodulate(self,
                   signal: np.ndarray,
                   channel_state: ChannelStateInformation,
                   noise_variance: float) -> Tuple[np.ndarray, ChannelStateInformation, np.ndarray]:
        """Demodulate a base-band signal stream to data symbols.

        Args:

            signal (np.ndarray):
                Vector of complex-valued base-band samples of a single communication frame.

            channel_state (ChannelStateInformation):
                Channel state information of a single communication frame.

            noise_variance (float):
                Variance of the thermal noise introduced during reception.

        Returns:
            (np.ndarray, ChannelStateInformation, np.ndarray):
                Tuple of 3 vectors of equal-length first dimension `num_symbols`.
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
            handle.waveform_generator = self

        self.__modem = handle

    @property
    @abstractmethod
    def sampling_rate(self) -> float:
        """Rate at which the waveform generator signal is internally sampled.

        Returns:
            float: Sampling rate in Hz.
        """
        ...

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
