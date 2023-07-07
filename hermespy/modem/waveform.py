# -*- coding: utf-8 -*-
"""
===========================
Communication Waveform Base
===========================
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from math import ceil
from typing import Generic, TYPE_CHECKING, Optional, Tuple, TypeVar, List

import numpy as np

from hermespy.core import ChannelStateInformation, Serializable, Signal, ChannelStateFormat
from hermespy.modem.tools.psk_qam_mapping import PskQamMapping
from .symbols import StatedSymbols, Symbols

if TYPE_CHECKING:
    from hermespy.modem.modem import BaseModem  # pragma: no cover

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


WaveformType = TypeVar("WaveformType", bound="WaveformGenerator")
"""Type hint for waveform generator classes."""


class Synchronization(Generic[WaveformType], ABC, Serializable):
    """Abstract base class for synchronization routines of waveform generators.

    Refer to :footcite:t:`2016:nasir` for an overview of the current state of the art.
    """

    yaml_tag = "Synchronization"
    property_blacklist = {"waveform_generator"}

    # Waveform generator this routine is attached to
    __waveform_generator: Optional[WaveformType]

    def __init__(self, waveform_generator: Optional[WaveformType] = None) -> None:
        """
        Args:
            waveform_generator (WaveformGenerator, optional):
                The waveform generator this synchronization routine is attached to.
        """

        self.__waveform_generator = waveform_generator

    @property
    def waveform_generator(self) -> Optional[WaveformType]:
        """Waveform generator this synchronization routine is attached to.

        Returns:
            Optional[WaveformType]:
                Handle to tghe waveform generator. None if the synchronization routine is floating.
        """

        return self.__waveform_generator

    @waveform_generator.setter
    def waveform_generator(self, value: Optional[WaveformType]) -> None:
        """Set waveform generator this synchronization routine is attached to."""

        # Un-register this synchronization routine from its previously assigned waveform
        if self.__waveform_generator is not None and self.__waveform_generator.synchronization is self:
            self.__waveform_generator.synchronization = Synchronization()

        self.__waveform_generator = value

    def synchronize(self, signal: np.ndarray) -> List[int]:
        """Simulates time-synchronization at the receiver-side.

        Sorts base-band signal-sections into frames in time-domain.

        Args:

            signal (np.ndarray):
                Vector of complex base-band samples of with `num_streams`x`num_samples` entries.

        Returns:

            List of time indices indicating the first samples of frames detected in `signal`.

        Raises:

            RuntimeError: If the synchronization routine is floating
        """

        if self.__waveform_generator is None:
            raise RuntimeError("Trying to synchronize with a floating synchronization routine")

        # samples_per_frame = self.__waveform_generator.samples_in_frame
        # num_frames = int(signal.shape[1] / samples_per_frame)

        return [0]


class ChannelEstimation(Generic[WaveformType], Serializable):
    """Base class for channel estimation routines of waveform generators."""

    yaml_tag = "NoChannelEstimation"
    property_blacklist = {"waveform_generator"}

    def __init__(self, waveform_generator: Optional[WaveformType] = None) -> None:
        """
        Args:
            waveform_generator (WaveformGenerator, optional):
                The waveform generator this estimation routine is attached to.
        """

        self.__waveform_generator = waveform_generator

    @property
    def waveform_generator(self) -> Optional[WaveformType]:
        """Waveform generator this synchronization routine is attached to.

        Returns:
            Optional[WaveformType]:
                Handle to the waveform generator. None if the synchronization routine is floating.
        """

        return self.__waveform_generator

    @waveform_generator.setter
    def waveform_generator(self, value: Optional[WaveformType]) -> None:
        """Set waveform generator this synchronization routine is attached to."""

        if self.__waveform_generator is not None:
            raise RuntimeError("Error trying to re-attach already attached synchronization routine.")

        self.__waveform_generator = value

    def estimate_channel(self, symbols: Symbols) -> Tuple[StatedSymbols, ChannelStateInformation]:
        """Estimate the wireless channel of a received communication frame.

        Args:

            symbols (Symbols):
                Demodulated communication symbols.

        Returns: The symbols and their respective channel states.
        """

        state = np.ones((symbols.num_streams, 1, symbols.num_blocks, symbols.num_symbols), dtype=complex)
        return StatedSymbols(symbols.raw, state), ChannelStateInformation(ChannelStateFormat.IMPULSE_RESPONSE, state)


class IdealChannelEstimation(Generic[WaveformType], ChannelEstimation[WaveformType]):
    """Channel estimation accessing the ideal channel state informaion.

    This type of channel estimation is only available during simulation runtime.
    """

    yaml_tag = "IdealChannelEstimation"

    def _csi(self) -> ChannelStateInformation:
        """Query the ideal channel state information.

        Returns: Ideal channel state information of the most recent reception.

        Raises:

            RuntimeError: If the estimation routine is not attached.
            RuntimeError: If no channel state is available.
        """

        if self.waveform_generator is None:
            raise RuntimeError("Ideal channel state estimation routine floating")

        if self.waveform_generator.modem is None or self.waveform_generator.modem.receiving_device is None:
            raise RuntimeError("Operating modem floating")

        cached_csi = self.waveform_generator.modem.csi
        if cached_csi is None:
            raise RuntimeError("No ideal channel state information available")

        return cached_csi


class ChannelEqualization(Generic[WaveformType], ABC, Serializable):
    """Abstract base class for channel equalization routines of waveform generators."""

    yaml_tag = "NoEqualization"
    property_blacklist = {"waveform_generator"}

    def __init__(self, waveform_generator: Optional[WaveformType] = None) -> None:
        """
        Args:
            waveform_generator (WaveformGenerator, optional):
                The waveform generator this equalization routine is attached to.
        """

        self.__waveform_generator = waveform_generator

    @property
    def waveform_generator(self) -> Optional[WaveformType]:
        """Waveform generator this equalization routine is attached to.

        Returns:
            Optional[WaveformType]:
                Handle to the waveform generator. None if the equalization routine is floating.
        """

        return self.__waveform_generator

    @waveform_generator.setter
    def waveform_generator(self, value: Optional[WaveformType]) -> None:
        """Set waveform generator this equalization routine is attached to."""

        if self.__waveform_generator is not None:
            raise RuntimeError("Error trying to re-attach already attached equalization routine.")

        self.__waveform_generator = value

    def equalize_channel(self, stated_symbols: StatedSymbols) -> Symbols:
        """Equalize the wireless channel of a received communication frame.

        Args:

            frame (Symbols): Symbols and channel state of the received communication frame.

        Returns: The equalize symbols.
        """

        # The default routine performs no equalization
        return stated_symbols


class ZeroForcingChannelEqualization(Generic[WaveformType], ChannelEqualization[WaveformType], Serializable):
    """Zero-Forcing channel equalization for arbitrary waveforms."""

    yaml_tag = "ZeroForcing"
    """YAML serialization tag"""

    def equalize_channel(self, symbols: StatedSymbols) -> Symbols:
        if symbols.num_streams < 2:
            summed_tx_states = np.sum(symbols.states, axis=1, keepdims=False)
            equalized_symbols = symbols.raw / summed_tx_states

        else:
            equalized_symbols = np.empty((symbols.num_transmit_streams, symbols.num_blocks, symbols.num_symbols), dtype=np.complex_)
            for b, s in np.ndindex(symbols.num_blocks, symbols.num_symbols):
                equalization = np.linalg.pinv(symbols.states[:, :, b, s])
                equalized_symbols[:, b, s] = np.dot(equalization, symbols.raw[:, b, s])

        return Symbols(equalized_symbols)


class WaveformGenerator(ABC, Serializable):
    """Implements an abstract waveform generator.

    Implementations for specific technologies should inherit from this class.
    """

    property_blacklist = {"modem"}

    symbol_type: type = np.complex_
    """Symbol type."""

    # Modem this waveform generator is attached to
    __modem: Optional[BaseModem]
    __synchronization: Synchronization  # Synchronization routine
    __channel_estimation: ChannelEstimation  # Channel estimation routine
    __channel_equalization: ChannelEqualization  # Channel equalization routine
    __oversampling_factor: int  # Oversampling factor
    # Cardinality of the set of communication symbols
    __modulation_order: int

    def __init__(self, modem: Optional[BaseModem] = None, oversampling_factor: int = 1, modulation_order: int = 16, channel_estimation: ChannelEstimation | None = None, channel_equalization: ChannelEqualization | None = None) -> None:
        """Waveform Generator initialization.

        Args:
            modem (BaseModem, optional):
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
        self.channel_estimation = ChannelEstimation(self) if channel_estimation is None else channel_estimation
        self.channel_equalization = ChannelEqualization(self) if channel_equalization is None else channel_equalization

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
        ...  # pragma: no cover

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

        return int(np.log2(self.modulation_order))

    @property
    @abstractmethod
    def bits_per_frame(self) -> int:
        """Number of bits required to generate a single data frame.

        Returns:
            int: Number of bits
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def symbols_per_frame(self) -> int:
        """Number of dat symbols per transmitted frame.

        Returns:
            int: Number of data symbols
        """
        ...  # pragma: no cover

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

        Energy of baseband_signal :math:`x[k]` is defined as :math:`\\sum{|x[k]}^2`
        Only data bits are considered, i.e., reference, guard intervals are ignored.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def symbol_energy(self) -> float:
        """The theoretical average symbol (discrete-time) energy of the modulated baseband_signal.

        Energy of baseband_signal :math:`x[k]` is defined as :math:`\\sum{|x[k]}^2`
        Only data bits are considered, i.e., reference, guard intervals are ignored.

        Returns:
            The average symbol energy in UNIT.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def power(self) -> float:
        """Returns the theoretical average symbol (unitless) power,

        Power of baseband_signal :math:`x[k]` is defined as :math:`\\sum_{k=1}^N{|x[k]|}^2 / N`
        Power is the average power of the data part of the transmitted frame, i.e., bit energy x raw bit rate
        """
        ...  # pragma: no cover

    @abstractmethod
    def map(self, data_bits: np.ndarray) -> Symbols:
        """Map a stream of bits to data symbols.

        Args:
            data_bits (np.ndarray):
                Vector containing a sequence of L hard data bits to be mapped onto data symbols.

        Returns:
            Symbols: Mapped data symbols.
        """
        ...  # pragma: no cover

    @abstractmethod
    def unmap(self, symbols: Symbols) -> np.ndarray:
        """Map a stream of data symbols to data bits.

        Args:
            symbols (Symbols):
                Sequence of K data symbols to be mapped onto bit sequences.

        Returns:
            np.ndarray:
                Vector containing the resulting sequence of L data bits
                In general, L is greater or equal to K.
        """
        ...  # pragma: no cover

    @abstractmethod
    def modulate(self, data_symbols: Symbols) -> Signal:
        """Modulate a stream of data symbols to a base-band signal containing a single data frame.

        Args:

            data_symbols (Symbols):
                Singular stream of data symbols to be modulated by this waveform.

        Returns:
            Signal: Signal model of a single modulate data frame.
        """
        ...  # pragma: no cover

    # Hint: Channel propagation occurs here

    @abstractmethod
    def demodulate(self, signal: np.ndarray) -> Symbols:
        """Demodulate a base-band signal stream to data symbols.

        Args:

            signal (np.ndarray):
                Vector of complex-valued base-band samples of a single communication frame.

        Returns:

            The demodulated communication symbols
        """
        ...  # pragma: no cover

    def estimate_channel(self, frame: Symbols) -> Tuple[StatedSymbols, ChannelStateInformation]:
        return self.channel_estimation.estimate_channel(frame)

    def equalize_symbols(self, symbols: StatedSymbols) -> Symbols:
        return self.channel_equalization.equalize_channel(symbols)

    @property
    @abstractmethod
    def bandwidth(self) -> float:
        """Bandwidth of the frame generated by this generator.

        Used to estimate the minimal sampling frequency required to adequately simulate the scenario.

        Returns:
            float: Bandwidth in Hz.
        """
        ...  # pragma: no cover

    @property
    def data_rate(self) -> float:
        """Data rate theoretically achieved by this waveform configuration.

        Returns:

            Bits per second.
        """

        time = self.frame_duration  # ToDo: Consider guard interval
        bits = self.bits_per_frame

        return bits / time

    @property
    def modem(self) -> Optional[BaseModem]:
        """Access the modem this generator is attached to.

        Returns: A handle to the modem.
        """

        return self.__modem

    @modem.setter
    def modem(self, handle: BaseModem | None) -> None:
        """Modify the modem this generator is attached to.

        Args:
            handle (Modem):
                Handle to a modem.

        Raises:
            RuntimeError:
                If the `modem` does not reference this generator.
        """

        if handle is None and self.__modem is None:
            return

        if handle.waveform_generator is not self:
            handle.waveform_generator = self

        self.__modem = handle
        self.random_mother = handle

    @property
    def synchronization(self) -> Synchronization:
        """Synchronization routine.

        Returns:
            Synchronization: Handle to the synchronization routine.
        """

        return self.__synchronization

    @synchronization.setter
    def synchronization(self, value: Synchronization) -> None:
        self.__synchronization = value

        if value.waveform_generator is not self:
            value.waveform_generator = self

    @property
    def channel_estimation(self) -> ChannelEstimation:
        """Channel estimation routine.

        Returns:
            ChannelEstimation: Handle to the estimation routine.
        """

        return self.__channel_estimation

    @channel_estimation.setter
    def channel_estimation(self, value: ChannelEstimation) -> None:
        self.__channel_estimation = value

        if value.waveform_generator is not self:
            value.waveform_generator = self

    @property
    def channel_equalization(self) -> ChannelEqualization:
        """Channel estimation routine.

        Returns:
            ChannelEqualization: Handle to the equalization routine.
        """

        return self.__channel_equalization

    @channel_equalization.setter
    def channel_equalization(self, value: ChannelEqualization) -> None:
        self.__channel_equalization = value

        if value.waveform_generator is not self:
            value.waveform_generator = self

    @property
    @abstractmethod
    def sampling_rate(self) -> float:
        """Rate at which the waveform generator signal is internally sampled.

        Returns:
            float: Sampling rate in Hz.
        """
        ...  # pragma: no cover

    @property
    def symbol_precoding_support(self) -> bool:
        """Flag indicating if this waveforms supports symbol precodings.

        Returns: Boolean support flag.
        """

        return True


class PilotWaveformGenerator(WaveformGenerator, ABC):
    """Abstract base class of communication waveform generators generating a pilot signal."""

    @property
    @abstractmethod
    def pilot_signal(self) -> Signal:
        """Model of the pilot sequence within this communication waveform.

        Returns:
            Signal: The pilot sequence.
        """
        ...  # pragma: no cover


class PilotSymbolSequence(ABC):
    """Abstract base class for pilot sequences."""

    @property
    @abstractmethod
    def sequence(self) -> np.ndarray:
        """The scalar sequence of pilot symbols.

        For a configurable pilot section, this symbol sequence will be repeated accordingly.

        Returns:
            The symbol sequence.
        """
        ...  # pragma: no cover


class UniformPilotSymbolSequence(PilotSymbolSequence):
    """A pilot symbol sequence containing identical symbols.

    Only viable for testing purposes, since it makes the pilot sections easy to spot within the frame.
    Not advisable to be used in production scenarios.
    """

    __pilot_symbol: complex  # The configured pilot symbol

    def __init__(self, pilot_symbol: complex = 1.0 + 0.0j) -> None:
        """
        Args:

            pilot_symbol (complex):
                The configured single pilot symbol.
                `1.` by default.
        """

        self.__pilot_symbol = pilot_symbol

    @property
    def sequence(self) -> np.ndarray:
        return np.array([self.__pilot_symbol], dtype=complex)


class CustomPilotSymbolSequence(PilotSymbolSequence):
    """A customized pilot symbol sequence.

    The user may freely chose the pilot symbols from samples within the complex plane.
    """

    __pilot_symbols: np.ndarray  # The configured pilot symbols

    def __init__(self, pilot_symbols: np.ndarray) -> None:
        """
        Args:

            pilot_symbols (np.ndarray):
                The configured pilot symbols
        """

        self.__pilot_symbols = pilot_symbols

    @property
    def sequence(self) -> np.ndarray:
        return self.__pilot_symbols


class MappedPilotSymbolSequence(CustomPilotSymbolSequence):
    """Pilot symbol sequence derived from a mapping."""

    def __init__(self, mapping: PskQamMapping) -> None:
        """

        Args:
            mapping (PskQamMapping): Mapping from which the symbols pilot symbols should be inferred
        """

        CustomPilotSymbolSequence.__init__(self, mapping.get_mapping())


class ConfigurablePilotWaveform(PilotWaveformGenerator, ABC):
    pilot_symbol_sequence: PilotSymbolSequence
    """The configured pilot symbol sequence."""

    repeat_pilot_symbol_sequence: bool
    """Allow the repetition of pilot symbol sequences."""

    def __init__(self, symbol_sequence: Optional[PilotSymbolSequence] = None, repeat_symbol_sequence: bool = True, **kwargs) -> None:
        """
        Args:

           symbol_sequence (Optional[PilotSymbolSequence], optional):
               The configured pilot symbol sequence.
               Uniform by default.

           repeat_symbol_sequence (bool, optional):
               Allow the repetition of pilot symbol sequences.
               Enabled by default.

           **kwargs:
               Additional :class:`WaveformGenerator` initialization parameters.
        """

        self.pilot_symbol_sequence = UniformPilotSymbolSequence() if symbol_sequence is None else symbol_sequence
        self.repeat_pilot_symbol_sequence = repeat_symbol_sequence

        # Initialize base class
        PilotWaveformGenerator.__init__(self, **kwargs)

    def pilot_symbols(self, num_symbols: int) -> np.ndarray:
        """Sample a pilot symbol sequence.

        Args:
            num_symbols (int):
                The expected number of symbols within the sequence.

        Returns:
            A pilot symbol sequence of length `num_symbols`.

        Raises:

            RuntimeError:
                If a repetition of the symbol sequence is required but not allowed.
        """

        symbol_sequence = self.pilot_symbol_sequence.sequence
        num_repetitions = int(ceil(num_symbols / len(symbol_sequence)))

        if num_repetitions > 1:
            if not self.repeat_pilot_symbol_sequence:
                raise RuntimeError("Pilot symbol repetition required for sequence generation but not allowed")

            symbol_sequence = np.tile(symbol_sequence, num_repetitions)

        return symbol_sequence[:num_symbols]
