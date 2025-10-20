# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod
from math import ceil
from typing import Generic, TypeVar
from typing_extensions import override

import numpy as np
from sparse import GCXS  # type: ignore

from hermespy.core import Serializable, Signal, SerializationProcess, DeserializationProcess
from hermespy.modem.tools.psk_qam_mapping import PskQamMapping
from .symbols import StatedSymbols, Symbols

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


CWT = TypeVar("CWT", bound="CommunicationWaveform")
"""Communication waveform type."""


class Synchronization(Generic[CWT], Serializable):
    """Abstract base class for synchronization routines of waveform generators.

    Refer to :footcite:t:`2016:nasir` for an overview of the current state of the art.
    """

    # Waveform generator this routine is attached to
    __waveform: CWT | None

    def __init__(self, waveform: CWT | None = None) -> None:
        """
        Args:
            waveform:
                The waveform generator this synchronization routine is attached to.
        """

        self.__waveform = waveform

    @property
    def waveform(self) -> CWT | None:
        """Waveform generator this synchronization routine is attached to.

        Returns: Handle to tghe waveform generator. None if the synchronization routine is floating.
        """

        return self.__waveform

    @waveform.setter
    def waveform(self, value: CWT | None) -> None:
        """Set waveform generator this synchronization routine is attached to."""

        # Un-register this synchronization routine from its previously assigned waveform
        if self.__waveform is not None and self.__waveform.synchronization is self:
            self.__waveform.synchronization = Synchronization()

        self.__waveform = value

    def synchronize(
        self, signal: np.ndarray, bandwidth: float, oversampling_factor: int
    ) -> list[int]:
        """Simulates time-synchronization at the receiver-side.

        Sorts base-band signal-sections into frames in time-domain.
        Note that the overall sampling rate is defined as `bandwidth * oversampling_factor`.

        Args:
            signal:
                Vector of complex base-band samples of with `num_streams`x`num_samples` entries.
            bandwidth:
                Bandwidth of the communication waveform in Hz.
            oversampling_factor:
                Oversampling factor of the communication waveform.

        Returns:
            List of time indices indicating the first samples of frames detected in `signal`.

        Raises:
            RuntimeError: If the synchronization routine is floating
        """

        if self.waveform is None:
            raise RuntimeError("Trying to synchronize with a floating synchronization routine")

        samples_per_frame = self.waveform.samples_per_frame(bandwidth, oversampling_factor)
        num_frames = signal.shape[1] // samples_per_frame

        if num_frames < 1:
            return []

        return list(range(0, num_frames * samples_per_frame, samples_per_frame))

    @override
    def serialize(self, process: SerializationProcess) -> None:
        pass

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> Synchronization:
        return cls()


class ChannelEstimation(Generic[CWT], Serializable):
    """Base class for channel estimation routines of waveform generators."""

    def __init__(self, waveform: CWT | None = None) -> None:
        """
        Args:
            waveform:
                The waveform generator this estimation routine is attached to.
        """

        self.__waveform = waveform

    @property
    def waveform(self) -> CWT | None:
        """Waveform generator this synchronization routine is attached to.

        Returns: Handle to the waveform generator. None if the synchronization routine is floating.
        """

        return self.__waveform

    @waveform.setter
    def waveform(self, value: CWT | None) -> None:
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

    def estimate_channel(
        self, symbols: Symbols, bandwidth: float, oversampling_factor: int, delay: float = 0.0
    ) -> StatedSymbols:
        """Estimate the wireless channel of a received communication frame.

        Args:
            symbols: Demodulated communication symbols.
            bandwidth: Bandwidth of the communication waveform in Hz.
            oversampling_factor: Oversampling factor of the communication waveform.
            delay: The considered frame's delay offset to the drop start in seconds.

        Returns:
            The symbols and their respective channel states.
        """

        state = GCXS.from_numpy(
            np.ones(
                (symbols.num_streams, 1, symbols.num_blocks, symbols.num_symbols), dtype=complex
            )
        )
        return StatedSymbols(symbols.raw, state)

    @override
    def serialize(self, process: SerializationProcess) -> None:
        pass

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> ChannelEstimation:
        return cls()


class ChannelEqualization(Generic[CWT], ABC, Serializable):
    """Abstract base class for channel equalization routines of waveform generators."""

    def __init__(self, waveform: CWT | None = None) -> None:
        """
        Args:
            waveform:
                The waveform generator this equalization routine is attached to.
        """

        self.__waveform = waveform

    @property
    def waveform(self) -> CWT | None:
        """Waveform generator this equalization routine is attached to.

        Returns: Handle to the waveform generator. None if the equalization routine is floating.
        """

        return self.__waveform

    @waveform.setter
    def waveform(self, value: CWT | None) -> None:
        """Set waveform generator this equalization routine is attached to."""

        if self.__waveform is not None:
            raise RuntimeError("Error trying to re-attach already attached equalization routine.")

        self.__waveform = value

    def equalize_channel(self, stated_symbols: StatedSymbols) -> Symbols:
        """Equalize the wireless channel of a received communication frame.

        Args:

            frame: Symbols and channel state of the received communication frame.

        Returns: The equalize symbols.
        """

        # The default routine performs no equalization
        return stated_symbols

    @override
    def serialize(self, process: SerializationProcess) -> None:
        pass

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> ChannelEqualization:
        return cls()


class ZeroForcingChannelEqualization(Generic[CWT], ChannelEqualization[CWT], Serializable):
    """Zero-Forcing channel equalization for arbitrary waveforms."""

    @override
    def equalize_channel(self, stated_symbols: StatedSymbols) -> Symbols:
        equalized_symbols: np.ndarray

        if stated_symbols.num_streams < 2:
            summed_tx_states = np.sum(stated_symbols.states, axis=1, keepdims=False)
            equalized_symbols = stated_symbols.raw / summed_tx_states

        else:
            equalization = np.linalg.pinv(stated_symbols.dense_states().transpose((2, 3, 0, 1)))
            equalized_symbols = np.einsum("ijkl,lij->kij", equalization, stated_symbols.raw)

        return Symbols(equalized_symbols)


class CommunicationWaveform(ABC, Serializable):
    """Abstract base class for all communication waveform descriptions."""

    __DEFAULT_MODULATION_ORDER: int = 16  # Default modulation order

    __synchronization: Synchronization  # Synchronization routine
    __channel_estimation: ChannelEstimation  # Channel estimation routine
    __channel_equalization: ChannelEqualization  # Channel equalization routine
    __modulation_order: int

    def __init__(
        self,
        modulation_order: int = __DEFAULT_MODULATION_ORDER,
        channel_estimation: ChannelEstimation | None = None,
        channel_equalization: ChannelEqualization | None = None,
        synchronization: Synchronization | None = None,
    ) -> None:
        """
        Args:

            modulation_order:
                Order of modulation.
                Must be a non-negative power of two.
                By default, a modulation order of :math:`16` is assumed.

            channel_estimation:
                Channel estimation algorithm.
                If not specified, no channel estimation is performed, which implicitly assumes an ideal channel.

            channel_equalization:
                Channel equalization algorithm. If not specified, no symbol equalization is performed.

            synchronization:
                Time-domain synchronization routine.
                If not specified, no synchronization is performed, which implicitly assumes a channel without delay.
        """

        # Default parameters
        self.modulation_order = modulation_order
        self.synchronization = Synchronization(self) if synchronization is None else synchronization
        self.channel_estimation = (
            ChannelEstimation(self) if channel_estimation is None else channel_estimation
        )
        self.channel_equalization = (
            ChannelEqualization(self) if channel_equalization is None else channel_equalization
        )

        if modulation_order is not None:
            self.modulation_order = modulation_order

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
            order:
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

            num_data_symbols:
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

    @abstractmethod
    def samples_per_frame(self, bandwidth: float, oversampling_factor: int) -> int:
        """Number of time-domain samples per processed communication frame.

        Args:

            bandwidth: Bandwidth of the communication waveform in Hz.
            oversampling_factor: Oversampling factor of the communication waveform.

        Note that the overall sampling rate is defined as `bandwidth * oversampling_factor`.

        Returns:
            int: Number of samples per frame.
        """
        ...  # pragma: no cover

    @abstractmethod
    def frame_duration(self, bandwidth: float) -> float:
        """Duration of a single communication frame in seconds.

        Args:
            bandwidth: Target bandwidth of the communication waveform in Hz.

        Returns:
            float: Duration of a single communication frame in seconds.
        """
        ...  # pragma: no cover

    def bit_energy(self, bandwidth: float, oversampling_factor: int) -> float:
        """Expected energy of a single data bit within the modulated baseband-signal.

        Typically denoted by :math:`E_b`.
        Can be derived from the :attr:`~symbol_energy` via the modulation order:

        .. math::
           E_b = \\frac{E_s}{\\log_2(M)}

        where :math:`M` is the modulation order, i.e. the number of unique symbols.

        Returns:
            float: The expected bit energy.
        """

        return self.symbol_energy(bandwidth, oversampling_factor) / self.bits_per_symbol

    @abstractmethod
    def symbol_energy(self, bandwidth: float, oversampling_factor: int) -> float:
        """Expected energy of a single communication symbol within the modulated baseband-signal.

        Args:
            bandwidth: Bandwidth of the communication waveform in Hz.
            oversampling_factor: Oversampling factor of the communication waveform.

        Typically denoted by :math:`E_s`.
        HermesPy defines the symbol energy as the expected sum of squared magnitudes

        .. math::
           E_s = \\sum_{n=0}^{N-1} |x[n]|^2

        where :math:`x[n]` are the complex base-band samples of a single communication symbol.

        Returns:
            float: The expected symbol energy.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def power(self) -> float:
        """Expected in-band power of the generated baseband-signal for within given target bandwidth.

        Typically denoted by :math:`P`.

        Returns:
            The expected power of the modulated signal.
        """
        ...  # pragma: no cover

    @abstractmethod
    def map(self, data_bits: np.ndarray[tuple[int], np.dtype[np.uint8]]) -> Symbols:
        """Map a stream of bits to data symbols.

        Args:
            data_bits:
                Vector containing a sequence of L hard data bits to be mapped onto data symbols.

        Returns: Mapped data symbols.
        """
        ...  # pragma: no cover

    @abstractmethod
    def unmap(self, symbols: Symbols) -> np.ndarray[tuple[int], np.dtype[np.uint8]]:
        """Map a stream of data symbols to data bits.

        Args:
            symbols:
                Sequence of K data symbols to be mapped onto bit sequences.

        Returns:
            Vector containing the resulting sequence of L data bits
            In general, L is greater or equal to K.
        """
        ...  # pragma: no cover

    @abstractmethod
    def place(self, symbols: Symbols) -> Symbols:
        """Place the mapped symbols within the communicaton frame.

        Additionally interleaves pilot symbols.

        Args:
            symbols: The mapped symbols.

        Returns: The symbols with the mapped symbols placed within the frame.
        """
        ...  # pragma: no cover

    @abstractmethod
    def pick(self, placed_symbols: StatedSymbols) -> StatedSymbols:
        """Pick the mapped symbols from the communicaton frame.

        Additionally removes interleaved pilot symbols.

        Args:
            placed_symbols: The placed symbols.

        Returns: The symbols with the mapped symbols picked from the frame.
        """
        ...  # pragma: no cover

    @abstractmethod
    def modulate(
        self, data_symbols: Symbols, bandwidth: float, oversampling_factor: int
    ) -> np.ndarray:
        """Modulate a stream of data symbols to a base-band signal containing a single data frame.

        Args:
            data_symbols: Singular stream of data symbols to be modulated by this waveform.
            bandwidth: Bandwidth of the communication waveform in Hz.
            oversampling_factor: Oversampling factor of the communication waveform.

        Returns:
            Samples of the modulated base-band signal.
        """
        ...  # pragma: no cover

    @abstractmethod
    def demodulate(self, signal: np.ndarray, bandwidth: float, oversampling_factor: int) -> Symbols:
        """Demodulate a base-band signal stream to data symbols.

        Args:
            signal:  Vector of complex-valued base-band samples of a single communication frame.
            bandwidth: Bandwidth of the communication waveform in Hz.
            oversampling_factor: Oversampling factor of the communication waveform.

        Returns:
            The demodulated communication symbols
        """
        ...  # pragma: no cover

    def estimate_channel(
        self, symbols: Symbols, bandwidth: float, oversampling_factor: int, delay: float = 0.0
    ) -> StatedSymbols:
        """Estimate the wireless channel of a received communication frame.

        Wrapper around the channel estimation rountine assigned to this waveform via
        the :attr:`~channel_estimation` property.

        Args:
            symbols: Demodulated communication symbols.
            bandwidth: Bandwidth of the communication waveform in Hz.
            oversampling_factor: Oversampling factor of the communication waveform.
            delay: The considered frame's delay offset to the drop start in seconds.

        Returns:
            The symbols and their respective channel states.
        """
        return self.channel_estimation.estimate_channel(
            symbols, bandwidth, oversampling_factor, delay
        )

    def equalize_symbols(self, symbols: StatedSymbols) -> Symbols:
        return self.channel_equalization.equalize_channel(symbols)

    def data_rate(self, num_data_symbols: int, bandwidth: float) -> float:
        """Data rate theoretically achieved by this waveform configuration.

        Args:
            num_data_symbols: Number of data symbols contained within the frame.
            bandwidth: Target bandwidth of the communication waveform in Hz.

        Returns: Bits per second.
        """

        time = self.frame_duration(bandwidth)  # ToDo: Consider guard interval
        bits = self.bits_per_frame(num_data_symbols)

        return bits / time

    @property
    def synchronization(self) -> Synchronization:
        """Synchronization routine."""

        return self.__synchronization

    @synchronization.setter
    def synchronization(self, value: Synchronization) -> None:
        self.__synchronization = value

        if value.waveform is not self:
            value.waveform = self

    @property
    def channel_estimation(self) -> ChannelEstimation:
        """Channel estimation routine."""

        return self.__channel_estimation

    @channel_estimation.setter
    def channel_estimation(self, value: ChannelEstimation) -> None:
        self.__channel_estimation = value

        if value.waveform is not self:
            value.waveform = self

    @property
    def channel_equalization(self) -> ChannelEqualization:
        """Channel estimation routine."""

        return self.__channel_equalization

    @channel_equalization.setter
    def channel_equalization(self, value: ChannelEqualization) -> None:
        self.__channel_equalization = value

        if value.waveform is not self:
            value.waveform = self

    @property
    def symbol_precoding_support(self) -> bool:
        """Flag indicating if this waveforms supports symbol precodings.

        Returns: Boolean support flag.
        """

        return True

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_integer(self.modulation_order, "modulation_order")
        process.serialize_object(self.synchronization, "synchronization")
        process.serialize_object(self.channel_estimation, "channel_estimation")
        process.serialize_object(self.channel_equalization, "channel_equalization")

    @classmethod
    def _DeserializeParameters(cls, process: DeserializationProcess) -> dict[str, object]:
        """Deserialize the initialization paramters of the communication waveform base class.

        Args:
            process: The deserialization process.

        Returns: The deserialized parameters.
        """

        return {
            "modulation_order": process.deserialize_integer(
                "modulation_order", cls.__DEFAULT_MODULATION_ORDER
            ),
            "synchronization": process.deserialize_object("synchronization", Synchronization),
            "channel_estimation": process.deserialize_object(
                "channel_estimation", ChannelEstimation
            ),
            "channel_equalization": process.deserialize_object(
                "channel_equalization", ChannelEqualization
            ),
        }


class PilotCommunicationWaveform(CommunicationWaveform):
    """Abstract base class of communication waveform generators generating a pilot signal."""

    @abstractmethod
    def pilot_signal(self, bandwidth: float, oversampling_factor: int) -> Signal:
        """Model of the pilot sequence within this communication waveform.

        Args:
            bandwidth: Bandwidth of the communication waveform in Hz.
            oversampling_factor: Oversampling factor of the communication waveform.

        Returns: The pilot sequence.
        """
        ...  # pragma: no cover


class PilotSymbolSequence(Serializable):
    """Abstract base class for pilot sequences."""

    @property
    @abstractmethod
    def sequence(self) -> np.ndarray:
        """The scalar sequence of pilot symbols.

        For a configurable pilot section, this symbol sequence will be repeated accordingly.

        Returns: The symbol sequence.
        """
        ...  # pragma: no cover


class UniformPilotSymbolSequence(PilotSymbolSequence):
    """A pilot symbol sequence containing identical symbols.

    Only viable for testing purposes, since it makes the pilot sections easy to spot within the frame.
    Not advisable to be used in production scenarios.
    """

    __DEFAULT_PILOT_SYMBOL: complex = 1.0 + 0.0j  # Default pilot symbol

    __pilot_symbol: complex  # The configured pilot symbol

    def __init__(self, pilot_symbol: complex = __DEFAULT_PILOT_SYMBOL) -> None:
        """
        Args:
            pilot_symbol:
                The configured single pilot symbol.
                `1.` by default.
        """

        self.__pilot_symbol = pilot_symbol

    @property
    @override
    def sequence(self) -> np.ndarray:
        return np.array([self.__pilot_symbol], dtype=complex)

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_array(
            np.array([self.__pilot_symbol], dtype=np.complex128), "pilot_symbol"
        )

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> UniformPilotSymbolSequence:
        pilot_symbol = process.deserialize_array(
            "pilot_symbol", np.complex128, np.array([cls.__DEFAULT_PILOT_SYMBOL])
        )[0]
        return UniformPilotSymbolSequence(pilot_symbol)


class CustomPilotSymbolSequence(PilotSymbolSequence):
    """A customized pilot symbol sequence.

    The user may freely chose the pilot symbols from samples within the complex plane.
    """

    __pilot_symbols: np.ndarray  # The configured pilot symbols

    def __init__(self, pilot_symbols: np.ndarray) -> None:
        """
        Args:

            pilot_symbols:
                The configured pilot symbols
        """

        self.__pilot_symbols = pilot_symbols

    @property
    def sequence(self) -> np.ndarray:
        return self.__pilot_symbols

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_array(self.__pilot_symbols, "pilot_symbols")

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> CustomPilotSymbolSequence:
        pilot_symbols = process.deserialize_array("pilot_symbols", np.complex128)
        return CustomPilotSymbolSequence(pilot_symbols)


class MappedPilotSymbolSequence(CustomPilotSymbolSequence):
    """Pilot symbol sequence derived from a mapping."""

    def __init__(self, mapping: PskQamMapping) -> None:
        """

        Args:
            mapping: Mapping from which the symbols pilot symbols should be inferred
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

           symbol_sequence:
               The configured pilot symbol sequence.
               Uniform by default.

           repeat_symbol_sequence:
               Allow the repetition of pilot symbol sequences.
               Enabled by default.

           kwargs:
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
            num_symbols: The expected number of symbols within the sequence.

        Returns: A pilot symbol sequence of length `num_symbols`.

        Raises:
            RuntimeError: If a repetition of the symbol sequence is required but not allowed.
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
