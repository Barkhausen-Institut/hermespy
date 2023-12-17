# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod
from math import ceil
from typing import Generic, TYPE_CHECKING, Optional, TypeVar, List

import numpy as np
from sparse import GCXS  # type: ignore

from hermespy.core import Serializable, Signal
from hermespy.modem.tools.psk_qam_mapping import PskQamMapping
from .symbols import StatedSymbols, Symbols

if TYPE_CHECKING:
    from hermespy.modem.modem import BaseModem  # pragma: no cover

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


WaveformType = TypeVar("WaveformType", bound="CommunicationWaveform")
"""Type hint for waveform generator classes."""


class Synchronization(Generic[WaveformType], ABC, Serializable):
    """Abstract base class for synchronization routines of waveform generators.

    Refer to :footcite:t:`2016:nasir` for an overview of the current state of the art.
    """

    yaml_tag = "Synchronization"
    property_blacklist = {"waveform"}

    # Waveform generator this routine is attached to
    __waveform: Optional[WaveformType]

    def __init__(self, waveform: Optional[WaveformType] = None) -> None:
        """
        Args:
            waveform (CommunicationWaveform, optional):
                The waveform generator this synchronization routine is attached to.
        """

        self.__waveform = waveform

    @property
    def waveform(self) -> Optional[WaveformType]:
        """Waveform generator this synchronization routine is attached to.

        Returns:
            Optional[WaveformType]:
                Handle to tghe waveform generator. None if the synchronization routine is floating.
        """

        return self.__waveform

    @waveform.setter
    def waveform(self, value: Optional[WaveformType]) -> None:
        """Set waveform generator this synchronization routine is attached to."""

        # Un-register this synchronization routine from its previously assigned waveform
        if (
            self.__waveform is not None
            and self.__waveform.synchronization is self
        ):
            self.__waveform.synchronization = Synchronization()

        self.__waveform = value

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

        if self.__waveform is None:
            raise RuntimeError("Trying to synchronize with a floating synchronization routine")

        samples_per_frame = self.waveform.samples_per_frame
        num_frames = signal.shape[1] // samples_per_frame

        return [] if num_frames < 1 else [0]


class ChannelEstimation(Generic[WaveformType], Serializable):
    """Base class for channel estimation routines of waveform generators."""

    yaml_tag = "NoChannelEstimation"
    property_blacklist = {"waveform"}

    def __init__(self, waveform: Optional[WaveformType] = None) -> None:
        """
        Args:
            waveform (CommunicationWaveform, optional):
                The waveform generator this estimation routine is attached to.
        """

        self.__waveform = waveform

    @property
    def waveform(self) -> Optional[WaveformType]:
        """Waveform generator this synchronization routine is attached to.

        Returns:
            Optional[WaveformType]:
                Handle to the waveform generator. None if the synchronization routine is floating.
        """

        return self.__waveform

    @waveform.setter
    def waveform(self, value: Optional[WaveformType]) -> None:
        """Set waveform generator this synchronization routine is attached to."""

        if value is None:
            waveform = self.__waveform
            self.__waveform = None

            if waveform is not None:
                waveform.channel_estimation = ChannelEstimation()

        else:
            if self.__waveform is not value and self.__waveform is not None:
                self.__waveform.channel_estimation = ChannelEstimation()

            self.__waveform = value
            value.channel_estimation = self

    def estimate_channel(self, symbols: Symbols, delay: float = 0.0) -> StatedSymbols:
        """Estimate the wireless channel of a received communication frame.

        Args:

            symbols (Symbols):
                Demodulated communication symbols.

            delay (float, optional):
                The considered frame's delay offset to the drop start in seconds.

        Returns: The symbols and their respective channel states.
        """

        state = GCXS.from_numpy(
            np.ones(
                (symbols.num_streams, 1, symbols.num_blocks, symbols.num_symbols), dtype=complex
            )
        )
        return StatedSymbols(symbols.raw, state)


class ChannelEqualization(Generic[WaveformType], ABC, Serializable):
    """Abstract base class for channel equalization routines of waveform generators."""

    yaml_tag = "NoEqualization"
    property_blacklist = {"waveform"}

    def __init__(self, waveform: Optional[WaveformType] = None) -> None:
        """
        Args:
            waveform (CommunicationWaveform, optional):
                The waveform generator this equalization routine is attached to.
        """

        self.__waveform = waveform

    @property
    def waveform(self) -> Optional[WaveformType]:
        """Waveform generator this equalization routine is attached to.

        Returns:
            Optional[WaveformType]:
                Handle to the waveform generator. None if the equalization routine is floating.
        """

        return self.__waveform

    @waveform.setter
    def waveform(self, value: Optional[WaveformType]) -> None:
        """Set waveform generator this equalization routine is attached to."""

        if self.__waveform is not None:
            raise RuntimeError("Error trying to re-attach already attached equalization routine.")

        self.__waveform = value

    def equalize_channel(self, stated_symbols: StatedSymbols) -> Symbols:
        """Equalize the wireless channel of a received communication frame.

        Args:

            frame (Symbols): Symbols and channel state of the received communication frame.

        Returns: The equalize symbols.
        """

        # The default routine performs no equalization
        return stated_symbols


class ZeroForcingChannelEqualization(
    Generic[WaveformType], ChannelEqualization[WaveformType], Serializable
):
    """Zero-Forcing channel equalization for arbitrary waveforms."""

    yaml_tag = "ZeroForcing"
    """YAML serialization tag"""

    def equalize_channel(self, symbols: StatedSymbols) -> Symbols:
        equalized_symbols: np.ndarray

        if symbols.num_streams < 2:
            summed_tx_states = np.sum(symbols.states, axis=1, keepdims=False)
            equalized_symbols = symbols.raw / summed_tx_states

        else:
            equalization = np.linalg.pinv(symbols.dense_states().transpose((2, 3, 0, 1)))
            equalized_symbols = np.einsum("ijkl,lij->kij", equalization, symbols.raw)

        return Symbols(equalized_symbols)


class CommunicationWaveform(ABC, Serializable):
    """Abstract base class for all communication waveform descriptions."""

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

    def __init__(
        self,
        modem: Optional[BaseModem] = None,
        oversampling_factor: int = 1,
        modulation_order: int = 16,
        channel_estimation: ChannelEstimation | None = None,
        channel_equalization: ChannelEqualization | None = None,
    ) -> None:
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
        self.channel_estimation = (
            ChannelEstimation(self) if channel_estimation is None else channel_estimation
        )
        self.channel_equalization = (
            ChannelEqualization(self) if channel_equalization is None else channel_equalization
        )

        if modem is not None:
            self.modem = modem

        if oversampling_factor is not None:
            self.oversampling_factor = oversampling_factor

        if modulation_order is not None:
            self.modulation_order = modulation_order

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

    def bits_per_frame(self, num_data_symbols: int | None = None) -> int:
        """Number of bits required to generate a single data frame.

        Args:

            num_data_symbols (int):
                Number of unique data symbols contained within the frame.
                If not specified, the waveform's default number of data symbols will be assumed.

        Returns: Number of bits.
        """

        _num_data_symbols = self.num_data_symbols if num_data_symbols is None else num_data_symbols
        return _num_data_symbols * self.bits_per_symbol

    @property
    @abstractmethod
    def num_data_symbols(self) -> int:
        """Number of bit-mapped symbols contained within each communication frame."""
        ...  # pragma: no cover

    @property
    @abstractmethod
    def samples_per_frame(self) -> int:
        """Number of time-domain samples per processed communication frame."""
        ...  # pragma: no cover

    @property
    def frame_duration(self) -> float:
        """Duration a single communication frame in seconds."""

        return self.samples_per_frame / self.sampling_rate

    @property
    @abstractmethod
    def symbol_duration(self) -> float:
        """Duration of a single symbol block."""
        ...  # pragma: no cover

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
    def place(self, symbols: Symbols) -> Symbols:
        """Place the mapped symbols within the communicaton frame.

        Additionally interleaves pilot symbols.

        Args:

            symbols (Symbols):
                The mapped symbols.

        Returns: The symbols with the mapped symbols placed within the frame.
        """
        ...  # pragma: no cover

    @abstractmethod
    def pick(self, placed_symbols: StatedSymbols) -> StatedSymbols:
        """Pick the mapped symbols from the communicaton frame.

        Additionally removes interleaved pilot symbols.

        Args:

            placed_symbols (StatedSymbols):
                The placed symbols.

        Returns: The symbols with the mapped symbols picked from the frame.
        """
        ...  # pragma: no cover

    @abstractmethod
    def modulate(self, data_symbols: Symbols) -> np.ndarray:
        """Modulate a stream of data symbols to a base-band signal containing a single data frame.

        Args:

            data_symbols (Symbols):
                Singular stream of data symbols to be modulated by this waveform.

        Returns: Samples of the modulated base-band signal.
        """
        ...  # pragma: no cover

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

    def estimate_channel(self, frame: Symbols, frame_delay: float = 0.0) -> StatedSymbols:
        return self.channel_estimation.estimate_channel(frame, frame_delay)

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

    def data_rate(self, num_data_symbols: int) -> float:
        """Data rate theoretically achieved by this waveform configuration.

        Args:

            num_data_symbols (int):
                Number of data symbols contained within the frame.

        Returns: Bits per second.
        """

        time = self.frame_duration  # ToDo: Consider guard interval
        bits = self.bits_per_frame(num_data_symbols)

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

        if handle is None:
            modem = self.__modem
            self.__modem = None

            if modem is not None:
                modem.waveform = None

        else:
            if handle.waveform is not self:
                handle.waveform = self

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

        if value.waveform is not self:
            value.waveform = self

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

        if value.waveform is not self:
            value.waveform = self

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

        if value.waveform is not self:
            value.waveform = self

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


class PilotCommunicationWaveform(CommunicationWaveform):
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


class ConfigurablePilotWaveform(PilotCommunicationWaveform):
    pilot_symbol_sequence: PilotSymbolSequence
    """The configured pilot symbol sequence."""

    repeat_pilot_symbol_sequence: bool
    """Allow the repetition of pilot symbol sequences."""

    def __init__(
        self,
        symbol_sequence: PilotSymbolSequence | None = None,
        repeat_symbol_sequence: bool = True,
        **kwargs,
    ) -> None:
        """
        Args:

           symbol_sequence (PilotSymbolSequence, optional):
               The configured pilot symbol sequence.
               Uniform by default.

           repeat_symbol_sequence (bool, optional):
               Allow the repetition of pilot symbol sequences.
               Enabled by default.

           **kwargs:
               Additional :class:`CommunicationWaveform` initialization parameters.
        """

        self.pilot_symbol_sequence = (
            UniformPilotSymbolSequence() if symbol_sequence is None else symbol_sequence
        )
        self.repeat_pilot_symbol_sequence = repeat_symbol_sequence

        # Initialize base class
        PilotCommunicationWaveform.__init__(self, **kwargs)

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
                raise RuntimeError(
                    "Pilot symbol repetition required for sequence generation but not allowed"
                )

            symbol_sequence = np.tile(symbol_sequence, num_repetitions)

        return symbol_sequence[:num_symbols]
