# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from typing_extensions import override

import matplotlib.pyplot as plt
import numpy as np

from hermespy.core import (
    Executable,
    FloatingError,
    Serializable,
    Signal,
    SerializationProcess,
    DeserializationProcess,
)
from .waveform import (
    ConfigurablePilotWaveform,
    MappedPilotSymbolSequence,
    CommunicationWaveform,
    ChannelEstimation,
    ChannelEqualization,
    PilotSymbolSequence,
    ZeroForcingChannelEqualization,
)
from hermespy.modem.tools.psk_qam_mapping import PskQamMapping
from .symbols import StatedSymbols, Symbols
from .waveform_correlation_synchronization import CorrelationSynchronization

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class FilteredSingleCarrierWaveform(ConfigurablePilotWaveform):
    """This method provides a class for a generic PSK/QAM modem.

    The modem has the following characteristics:
    - root-raised cosine filter with arbitrary roll-off factor
    - arbitrary constellation, as defined in modem.tools.psk_qam_mapping:PskQamMapping

    This implementation has currently the following limitations:
    - hard output only (no LLR)
    - no reference signal
    - ideal channel estimation
    - equalization of ISI with FMCW in AWGN channel only
    - no equalization (only amplitude and phase of first propagation path is compensated)
    """

    __DEFAULT_SYMBOL_RATE: float = 1e6
    __DEFAULT_NUM_PREAMBLE_SYMBOLS: int = 16
    __DEFAULT_NUM_DATA_SYMBOLS: int = 256
    __DEFAULT_NUM_POSTAMBLE_SYMBOLS: int = 0
    __DEFAULT_PILOT_RATE: int = 0
    __DEFAULT_GUARD_INTERVAL: float = 0.0
    __DEFAULT_OVERSAMPLING_FACTOR: int = 4

    __symbol_rate: float
    __num_preamble_symbols: int
    __num_data_symbols: int
    __num_postamble_symbols: int
    __guard_interval: float
    __mapping: PskQamMapping
    __pilot_rate: int
    _data_symbol_idx: np.ndarray | None
    _pulse_correlation_matrix: np.ndarray | None

    def __init__(
        self,
        symbol_rate: float = __DEFAULT_SYMBOL_RATE,
        num_preamble_symbols: int = __DEFAULT_NUM_PREAMBLE_SYMBOLS,
        num_data_symbols: int = __DEFAULT_NUM_DATA_SYMBOLS,
        num_postamble_symbols: int = __DEFAULT_NUM_POSTAMBLE_SYMBOLS,
        pilot_rate: int = __DEFAULT_PILOT_RATE,
        guard_interval: float = __DEFAULT_GUARD_INTERVAL,
        oversampling_factor: int = __DEFAULT_OVERSAMPLING_FACTOR,
        pilot_symbol_sequence: PilotSymbolSequence | None = None,
        repeat_pilot_symbol_sequence: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Args:

            symbol_rate:
                Rate at which symbols are being generated in Hz.

            num_preamble_symbols:
                Number of preamble symbols within a single communication frame.

            num_data_symbols:
                Number of data symbols within a single communication frame.

            num_postamble_symbols:
                Number of postamble symbols within a single communication frame.

            guard_interval:
                Guard interval between communication frames in seconds.
                Zero by default.

            oversampling_factor:
                The oversampling factor of the waform.

            pilot_rate:
                Pilot symbol rate.
                Zero by default, i.e. no pilot symbols.

            pilot_symbol_sequence:
                The configured pilot symbol sequence.
                Uniform by default.

            repeat_pilot_symbol_sequence:
                Allow the repetition of pilot symbol sequences.
                Enabled by default.

            \*\*kwargs:
                Waveform generator base class initialization parameters.
        """

        # Init base class
        ConfigurablePilotWaveform.__init__(
            self,
            repeat_symbol_sequence=repeat_pilot_symbol_sequence,
            oversampling_factor=oversampling_factor,
            **kwargs,
        )

        self.symbol_rate = symbol_rate
        self.num_preamble_symbols = num_preamble_symbols
        self.num_data_symbols = num_data_symbols
        self.num_postamble_symbols = num_postamble_symbols
        self.pilot_rate = pilot_rate
        self.guard_interval = guard_interval
        self.pilot_symbol_sequence = (
            MappedPilotSymbolSequence(self.__mapping)
            if pilot_symbol_sequence is None
            else pilot_symbol_sequence
        )

    @abstractmethod
    def _transmit_filter(self) -> np.ndarray:
        """Pulse shaping filter applied to data symbols during transmission.

        Returns: The shaping filter impulse response.
        """
        ...  # pragma: no cover

    @abstractmethod
    def _receive_filter(self) -> np.ndarray:
        """Pulse shaping filter applied to signal streams during reception.


        Returns: The shaping filter impulse response.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def _filter_delay(self) -> int:
        """Cumulative delay introduced during transmit and receive filtering.

        Returns: Delay in samples.
        """
        ...  # pragma: no cover

    @property
    def symbol_rate(self) -> float:
        """Repetition rate of symbols.

        Returns: Symbol rate in Hz.

        Raises:
            ValueError: For rates smaller or equal to zero.
        """

        return self.__symbol_rate

    @symbol_rate.setter
    def symbol_rate(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("Symbol rate must be greater than zero")

        self.__symbol_rate = value

    @property
    def num_preamble_symbols(self) -> int:
        """Number of preamble symbols.

        Transmitted at the beginning of communication frames.

        Raises:
            ValueError: If the number of symbols is smaller than zero.
        """

        return self.__num_preamble_symbols

    @num_preamble_symbols.setter
    def num_preamble_symbols(self, value: int) -> None:
        if value < 0:
            raise ValueError("Nummber of preamble symbols must be greater or equal to zero")

        self.__num_preamble_symbols = value

    @property
    def num_postamble_symbols(self) -> int:
        """Number of postamble symbols.

        Transmitted at the end of communication frames.

        Raises:
            ValueError: If the number of symbols is smaller than zero.
        """

        return self.__num_postamble_symbols

    @num_postamble_symbols.setter
    def num_postamble_symbols(self, value: int) -> None:
        if value < 0:
            raise ValueError("Nummber of postamble symbols must be greater or equal to zero")

        self.__num_postamble_symbols = value

    @CommunicationWaveform.modulation_order.setter  # type: ignore
    def modulation_order(self, value: int) -> None:
        self.__mapping = PskQamMapping(value, soft_output=False)
        self.pilot_symbol_sequence = MappedPilotSymbolSequence(
            self.__mapping
        )  # ToDo: Find a better way to update the pilot symbol sequence
        CommunicationWaveform.modulation_order.fset(self, value)  # type: ignore

    @property
    def pilot_signal(self) -> Signal:
        if self.num_preamble_symbols < 1:
            return Signal.Empty(self.sampling_rate)

        pilot_symbols = np.zeros(
            1 + (self.num_preamble_symbols - 1) * self.oversampling_factor, dtype=complex
        )
        pilot_symbols[:: self.oversampling_factor] = self.pilot_symbols(self.num_preamble_symbols)

        return Signal.Create(
            np.convolve(pilot_symbols, self._transmit_filter()), sampling_rate=self.sampling_rate
        )

    @override
    def map(self, bits: np.ndarray) -> Symbols:
        return Symbols(self.__mapping.get_symbols(bits))

    @override
    def unmap(self, symbols: Symbols) -> np.ndarray:
        return self.__mapping.detect_bits(symbols.raw.flatten())

    @override
    def place(self, data_symbols: Symbols) -> Symbols:
        # Generate pilot symbol sequences
        pilot_symbols = self.pilot_symbols(
            self.num_preamble_symbols + self._num_pilot_symbols + self.num_postamble_symbols
        )
        placed_symbols = np.empty(self._num_frame_symbols, dtype=np.complex128)

        # Assign preamble symbols within the frame
        placed_symbols[: self.num_preamble_symbols] = pilot_symbols[: self.num_preamble_symbols]

        # Assign postamble symbols within the frame
        placed_symbols[self.num_preamble_symbols + self._num_payload_symbols :] = pilot_symbols[
            self.num_preamble_symbols + self._num_pilot_symbols :
        ]

        # Assign payload symbols within the frame
        # The payload consists of data symbols interleaved with pilots according to the pilot rate
        placed_symbols[self.num_preamble_symbols + self._pilot_symbol_indices] = pilot_symbols[
            self.num_preamble_symbols : self.num_preamble_symbols + self._num_pilot_symbols
        ]
        placed_symbols[self.num_preamble_symbols + self._data_symbol_indices] = (
            data_symbols.raw.flatten()
        )

        return Symbols(placed_symbols[np.newaxis, :, np.newaxis])

    @override
    def pick(self, symbols: StatedSymbols) -> StatedSymbols:
        data_block_indices = self.num_preamble_symbols + self._data_symbol_indices
        picked_symbol_blocks = symbols.raw[:, data_block_indices, :]
        picked_state_blocks = symbols.states[:, :, data_block_indices, :]

        return StatedSymbols(picked_symbol_blocks, picked_state_blocks)

    @override
    def modulate(self, symbols: Symbols) -> np.ndarray:
        frame = np.zeros(
            1 + (self._num_frame_symbols - 1) * self.oversampling_factor, dtype=complex
        )
        frame[:: self.oversampling_factor] = symbols.raw.flatten()

        # Generate waveforms by treating the frame as a comb and convolving with the impulse response
        output_signal = np.convolve(frame, self._transmit_filter())
        return output_signal

    @override
    def demodulate(self, signal: np.ndarray) -> Symbols:
        # Query filters
        filter_delay = self._filter_delay

        # Filter the signal and csi
        filtered_signal = np.convolve(signal, self._receive_filter())
        symbols = filtered_signal[
            filter_delay : filter_delay
            + self._num_frame_symbols * self.oversampling_factor : self.oversampling_factor
        ]

        return Symbols(symbols[np.newaxis, :, np.newaxis])

    @property
    def guard_interval(self) -> float:
        """Frame guard interval.

        Raises:
            ValueError: If `interval` is smaller than zero.
        """

        return self.__guard_interval

    @guard_interval.setter
    def guard_interval(self, interval: float) -> None:
        if interval < 0.0:
            raise ValueError("Guard interval must be greater or equal to zero")

        self.__guard_interval = interval

    @property
    def num_guard_samples(self) -> int:
        """Number of samples within the guarding section of a frame."""

        return int(round(self.guard_interval * self.sampling_rate))

    @property
    def pilot_rate(self) -> int:
        """Repetition rate of pilot symbols within the frame.

        A pilot rate of zero indicates no pilot symbols within the data frame.

        Raises:
            ValueError: If the pilot rate is smaller than zero.
        """

        return self.__pilot_rate

    @pilot_rate.setter
    def pilot_rate(self, value: int) -> None:
        if value < 0:
            raise ValueError("Pilot symbol rate must be greater or equal to zero")

        self.__pilot_rate = int(value)

    @property
    def _num_pilot_symbols(self) -> int:
        if self.pilot_rate <= 0:
            return 0

        return max(0, int(self.num_data_symbols / self.pilot_rate) - 1)

    @property
    def _num_payload_symbols(self) -> int:
        num_symbols = self.num_data_symbols + self._num_pilot_symbols
        return num_symbols

    @property
    def _num_frame_symbols(self) -> int:
        return self.num_preamble_symbols + self._num_payload_symbols + self.num_postamble_symbols

    @property
    def _pilot_symbol_indices(self) -> np.ndarray:
        """Indices of pilot symbols within the ful communication frame."""

        if self.pilot_rate <= 0:
            return np.empty(0, dtype=int)

        pilot_indices = np.arange(1, 1 + self._num_pilot_symbols) * (1 + self.pilot_rate) - 1
        return pilot_indices

    @property
    def _data_symbol_indices(self) -> np.ndarray:
        """Indices of data symbols within the full communication frame."""

        data_indices = np.arange(self._num_payload_symbols)

        payload_indices = self._pilot_symbol_indices
        if len(payload_indices) > 0:
            data_indices = np.delete(data_indices, self._pilot_symbol_indices)

        return data_indices

    @property
    def num_data_symbols(self) -> int:
        """Number of data symbols per frame.

        Raises:
            ValueError: If `num` is smaller than zero.
        """

        return self.__num_data_symbols

    @num_data_symbols.setter
    def num_data_symbols(self, num: int) -> None:
        if num < 0:
            raise ValueError("Number of data symbols must be greater or equal to zero")

        self.__num_data_symbols = num

    @property
    def samples_per_frame(self) -> int:
        return (
            self._num_frame_symbols - 1
        ) * self.oversampling_factor + self._transmit_filter().shape[0]

    @property
    @override
    def symbol_duration(self) -> float:
        return 1 / self.symbol_rate

    @property
    @override
    def bit_energy(self) -> float:
        return 1 / self.bits_per_symbol

    @property
    @override
    def symbol_energy(self) -> float:
        return 1.0

    @property
    @override
    def power(self) -> float:
        return 1 / self.oversampling_factor

    @property
    @override
    def sampling_rate(self) -> float:
        return self.symbol_rate * self.oversampling_factor

    def plot_filter_correlation(self) -> plt.Figure:
        """Plot the convolution between transmit and receive filter shapes.

        Returns: Handle to the generated matplotlib figure.
        """

        with Executable.style_context():
            tx_filter = self._transmit_filter()
            rx_filter = self._receive_filter()

            autocorrelation = np.convolve(tx_filter, rx_filter)

            fig, axes = plt.subplots()
            fig.suptitle("Pulse Autocorrelation")

            axes.plot(np.abs(autocorrelation))

        return fig

    def plot_filter(self) -> plt.Figure:
        """Plot the transmit filter shape.

        Returns: Handle to the generated matplotlib figure.
        """

        with Executable.style_context():
            tx_filter = self._transmit_filter()

            fig, axes = plt.subplots()
            fig.suptitle("Pulse Shape")

            axes.plot(tx_filter.real)
            axes.plot(tx_filter.imag)

        return fig

    @override
    def serialize(self, process: SerializationProcess) -> None:
        ConfigurablePilotWaveform.serialize(self, process)
        process.serialize_floating(self.symbol_rate, "symbol_rate")
        process.serialize_integer(self.num_preamble_symbols, "num_preamble_symbols")
        process.serialize_integer(self.num_data_symbols, "num_data_symbols")
        process.serialize_integer(self.num_postamble_symbols, "num_postamble_symbols")
        process.serialize_floating(self.guard_interval, "guard_interval")
        process.serialize_integer(self.pilot_rate, "pilot_rate")

    @override
    @classmethod
    def _DeserializeParameters(cls, process: DeserializationProcess) -> dict[str, Any]:
        parameters = ConfigurablePilotWaveform._DeserializeParameters(process)
        parameters["symbol_rate"] = process.deserialize_floating(
            "symbol_rate", cls.__DEFAULT_SYMBOL_RATE
        )
        parameters["num_preamble_symbols"] = process.deserialize_integer(
            "num_preamble_symbols", cls.__DEFAULT_NUM_PREAMBLE_SYMBOLS
        )
        parameters["num_data_symbols"] = process.deserialize_integer(
            "num_data_symbols", cls.__DEFAULT_NUM_DATA_SYMBOLS
        )
        parameters["num_postamble_symbols"] = process.deserialize_integer(
            "num_postamble_symbols", cls.__DEFAULT_NUM_POSTAMBLE_SYMBOLS
        )
        parameters["guard_interval"] = process.deserialize_floating(
            "guard_interval", cls.__DEFAULT_GUARD_INTERVAL
        )
        parameters["pilot_rate"] = process.deserialize_integer(
            "pilot_rate", cls.__DEFAULT_PILOT_RATE
        )
        return parameters

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> FilteredSingleCarrierWaveform:
        return cls(**cls._DeserializeParameters(process))  # type: ignore[arg-type]


class SingleCarrierCorrelationSynchronization(
    CorrelationSynchronization[FilteredSingleCarrierWaveform]
):
    """Correlation-based clock-synchronization for PSK-QAM waveforms."""


class SingleCarrierChannelEstimation(ChannelEstimation[FilteredSingleCarrierWaveform], ABC):
    """Channel estimation for Psk Qam waveforms."""

    def __init__(self, waveform: FilteredSingleCarrierWaveform | None = None) -> None:
        """
        Args:

            waveform: The waveform generator this synchronization routine is attached to.
        """

        ChannelEstimation.__init__(self, waveform)


class SingleCarrierLeastSquaresChannelEstimation(SingleCarrierChannelEstimation):
    """Least-Squares channel estimation for Psk Qam waveforms."""

    def __init__(self, waveform: FilteredSingleCarrierWaveform | None = None) -> None:
        """
        Args:

            waveform: The waveform generator this channel estimation routine is attached to.
        """

        SingleCarrierChannelEstimation.__init__(self, waveform)

    def estimate_channel(self, symbols: Symbols, delay: float = 0.0) -> StatedSymbols:
        if self.waveform is None:
            raise FloatingError(
                "Error trying to fetch the pilot section of a floating channel estimator"
            )

        # Query required waveform information
        num_preamble_symbols = self.waveform.num_preamble_symbols
        num_postamble_symbols = self.waveform.num_postamble_symbols
        num_payload_symbols = self.waveform._num_payload_symbols
        num_pilot_symbols = self.waveform._num_pilot_symbols
        pilot_symbol_indices = self.waveform._pilot_symbol_indices
        transmitted_reference_symbols = self.waveform.pilot_symbols(
            num_preamble_symbols + num_pilot_symbols + num_postamble_symbols
        )

        # Extract reference symbols
        preamble_symbols = symbols.raw[:, :num_preamble_symbols, 0]
        pilot_symbols = symbols.raw[
            :, num_preamble_symbols : num_preamble_symbols + num_payload_symbols, 0
        ][:, pilot_symbol_indices]
        postamble_symbols = symbols.raw[:, num_preamble_symbols + num_payload_symbols :, 0]
        received_reference_symbols = np.concatenate(
            (preamble_symbols, pilot_symbols, postamble_symbols), axis=1
        )

        # Estimate the channel over all reference symbols
        channel_estimation_stems = received_reference_symbols / transmitted_reference_symbols
        channel_estimation_stem_indices = np.concatenate(
            (
                np.arange(num_preamble_symbols),
                pilot_symbol_indices + num_preamble_symbols,
                np.arange(num_postamble_symbols) + num_preamble_symbols + num_payload_symbols,
            )
        )

        # Interpolate to the whole channel estimation
        channel_estimation_indices = np.arange(symbols.num_blocks)
        channel_estimation = np.empty((symbols.num_streams, symbols.num_blocks), dtype=complex)
        for s, stems in enumerate(channel_estimation_stems):
            channel_estimation[s, :] = np.interp(
                channel_estimation_indices, channel_estimation_stem_indices, stems
            )

        return StatedSymbols(symbols.raw, channel_estimation[:, np.newaxis, :, np.newaxis])


class SingleCarrierChannelEqualization(ChannelEqualization[FilteredSingleCarrierWaveform], ABC):
    """Channel estimation for Psk Qam waveforms."""

    def __init__(self, waveform: FilteredSingleCarrierWaveform | None = None) -> None:
        """
        Args:

            waveform:
                The waveform generator this equalization routine is attached to.
        """

        ChannelEqualization.__init__(self, waveform)


class SingleCarrierZeroForcingChannelEqualization(
    ZeroForcingChannelEqualization[FilteredSingleCarrierWaveform]
):
    """Zero-Forcing Channel estimation for Psk Qam waveforms."""


class SingleCarrierMinimumMeanSquareChannelEqualization(SingleCarrierChannelEqualization, ABC):
    """Minimum-Mean-Square Channel estimation for Psk Qam waveforms."""

    def __init__(self, waveform: FilteredSingleCarrierWaveform | None = None) -> None:
        """
        Args:

            waveform:
                The waveform generator this equalization routine is attached to.
        """

        SingleCarrierChannelEqualization.__init__(self, waveform)

    def equalize_channel(self, symbols: StatedSymbols) -> Symbols:
        # Query SNR and cached CSI from the device
        snr = float("inf")  # self.waveform.modem.receiving_device.snr

        # If no information about transmitted streams is available, assume orthogonal channels
        if symbols.num_transmit_streams < 2 and symbols.num_streams < 2:
            return Symbols(symbols.raw / (symbols.states[:, 0, :, :] + 1 / snr))

        if symbols.num_transmit_streams > symbols.num_streams:
            raise RuntimeError(
                "MMSE equalization is not supported for more transmit streams than receive streams"
            )

        # Default behaviour for mimo systems is to use the pseudo-inverse for equalization
        raw_equalized_symbols = np.empty(
            (symbols.num_transmit_streams, symbols.num_blocks, symbols.num_symbols), dtype=complex
        )
        for b, s in np.ndindex(symbols.num_blocks, symbols.num_symbols):
            symbol_slice = symbols.raw[:, b, s]
            mimo_state = symbols.states[:, :, b, s]

            # ToDo: Introduce noise term here
            equalization = np.linalg.pinv(mimo_state)
            raw_equalized_symbols[:, b, s] = equalization @ symbol_slice

        return Symbols(raw_equalized_symbols)


class RolledOffSingleCarrierWaveform(FilteredSingleCarrierWaveform):
    """Base class for single carrier waveforms applying linear filters longer than a single symbol duration."""

    __DEFAULT_RELATIVE_BANDWIDTH: float = 1.0  # Default pulse bandwidth relative to the symbol rate
    __DEFAULT_ROLL_OFF: float = 0.0  # Default filter pulse roll off factor
    __DEFAULT_FILTER_LENGTH: int = 16  # Default filter length in modulation symbols

    # Pulse bandwidth relative to the configured symbol rate
    __relative_bandwidth: float
    __roll_off: float  # Filter pulse roll off factor
    __filter_length: int  # Filter length in modulation symbols

    def __init__(
        self,
        relative_bandwidth: float = __DEFAULT_RELATIVE_BANDWIDTH,
        roll_off: float = __DEFAULT_ROLL_OFF,
        filter_length: int = __DEFAULT_FILTER_LENGTH,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:

            relative_bandwidth:
                Bandwidth relative to the configured symbol rate.
                One by default, meaning the pulse bandwidth is equal to the symbol rate in Hz,
                assuming zero `roll_off`.

            roll_off:
                Filter pulse shape roll off factor between zero and one.
                Zero by default, meaning no inter-symbol interference at the sampling instances.

            filter_length:
                Filter length in modulation symbols.
                16 by default.
        """

        self.relative_bandwidth = relative_bandwidth
        self.roll_off = roll_off
        self.filter_length = filter_length

        FilteredSingleCarrierWaveform.__init__(self, *args, **kwargs)

    @property
    def filter_length(self) -> int:
        """Filter length in modulation symbols.

        Configures how far the shaping filter stretches in terms of the number of
        modulation symbols it overlaps with.

        Raises:
            ValueError: For filter lengths smaller than one.
        """

        return self.__filter_length

    @filter_length.setter
    def filter_length(self, value: int) -> None:
        if value < 1:
            raise ValueError("Filter length must be greater than zero")

        self.__filter_length = value

    @property
    def relative_bandwidth(self) -> float:
        """Bandwidth relative to the configured symbol rate.

        Raises:
            ValueError: On values smaller or equal to zero.
        """

        return self.__relative_bandwidth

    @relative_bandwidth.setter
    def relative_bandwidth(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("Relative pulse bandwidth must be greater than zero")

        self.__relative_bandwidth = value

    @property
    def roll_off(self) -> float:
        """Filter pulse shape roll off factor.

        Raises:
            ValueError: On values smaller than zero or larger than one.
        """

        return self.__roll_off

    @roll_off.setter
    def roll_off(self, value: float) -> None:
        if value < 0.0 or value > 1.0:
            raise ValueError(
                "Filter pulse shape roll off factor value must be between zero and one"
            )

        self.__roll_off = value

    @property
    def bandwidth(self) -> float:
        return self.symbol_rate * self.relative_bandwidth * (1 + self.roll_off)

    @abstractmethod
    def _base_filter(self) -> np.ndarray:
        """Generate the base filter impulse response.

        Returns:
            The base filter impulse response as a numpy array.
        """
        ...  # pragma: no cover

    def _transmit_filter(self) -> np.ndarray:
        return self._base_filter()

    def _receive_filter(self) -> np.ndarray:
        return self._base_filter()

    @property
    def _filter_delay(self) -> int:
        return 2 * int(0.5 * self.filter_length) * self.oversampling_factor

    @override
    def serialize(self, process: SerializationProcess) -> None:
        FilteredSingleCarrierWaveform.serialize(self, process)
        process.serialize_floating(self.relative_bandwidth, "relative_bandwidth")
        process.serialize_floating(self.roll_off, "roll_off")
        process.serialize_integer(self.filter_length, "filter_length")

    @override
    @classmethod
    def _DeserializeParameters(cls, process: DeserializationProcess) -> dict[str, Any]:
        parameters = FilteredSingleCarrierWaveform._DeserializeParameters(process)
        parameters["relative_bandwidth"] = process.deserialize_floating(
            "relative_bandwidth", cls.__DEFAULT_RELATIVE_BANDWIDTH
        )
        parameters["roll_off"] = process.deserialize_floating("roll_off", cls.__DEFAULT_ROLL_OFF)
        parameters["filter_length"] = process.deserialize_integer(
            "filter_length", cls.__DEFAULT_FILTER_LENGTH
        )
        return parameters


class RootRaisedCosineWaveform(RolledOffSingleCarrierWaveform, Serializable):
    """Root-Raised-Cosine filtered single carrier modulation."""

    def __init__(self, *args, **kwargs) -> None:
        RolledOffSingleCarrierWaveform.__init__(self, *args, **kwargs)

    def _base_filter(self) -> np.ndarray:
        impulse_response = np.zeros(self.oversampling_factor * self.filter_length)

        # Generate timestamps
        time = (
            np.linspace(
                -int(0.5 * self.filter_length),
                int(0.5 * self.filter_length),
                self.filter_length * self.oversampling_factor,
                endpoint=(self.filter_length % 2 == 1),
            )
            * self.relative_bandwidth
        )

        # Build filter response
        idx_0_by_0 = time == 0  # indices with division of zero by zero

        if self.roll_off != 0:
            # indices with division by zero
            idx_x_by_0 = abs(time) == 1 / (4 * self.roll_off)
        else:
            idx_x_by_0 = np.zeros_like(time, dtype=bool)
        idx = (~idx_0_by_0) & (~idx_x_by_0)

        impulse_response[idx] = (
            np.sin(np.pi * time[idx] * (1 - self.roll_off))
            + 4 * self.roll_off * time[idx] * np.cos(np.pi * time[idx] * (1 + self.roll_off))
        ) / (np.pi * time[idx] * (1 - (4 * self.roll_off * time[idx]) ** 2))
        if np.any(idx_x_by_0):
            impulse_response[idx_x_by_0] = (
                self.roll_off
                / np.sqrt(2)
                * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * self.roll_off))
                    + (1 - 2 / np.pi) * np.cos(np.pi / (4 * self.roll_off))
                )
            )
        impulse_response[idx_0_by_0] = 1 + self.roll_off * (4 / np.pi - 1)
        return impulse_response / np.linalg.norm(impulse_response)


class RaisedCosineWaveform(RolledOffSingleCarrierWaveform, Serializable):
    """Root-Raised-Cosine filtered single carrier modulation."""

    def __init__(self, *args, **kwargs) -> None:
        RolledOffSingleCarrierWaveform.__init__(self, *args, **kwargs)

    def _base_filter(self) -> np.ndarray:
        impulse_response = np.zeros(self.oversampling_factor * self.filter_length)

        # Generate timestamps
        time = (
            np.linspace(
                -int(0.5 * self.filter_length),
                int(0.5 * self.filter_length),
                self.filter_length * self.oversampling_factor,
                endpoint=(self.filter_length % 2 == 1),
            )
            * self.relative_bandwidth
        )

        # Build filter response
        if self.roll_off != 0:
            # indices with division of zero by zero
            idx_0_by_0 = abs(time) == 1 / (2 * self.roll_off)
        else:
            idx_0_by_0 = np.zeros_like(time, dtype=bool)
        idx = ~idx_0_by_0
        impulse_response[idx] = (
            np.sinc(time[idx])
            * np.cos(np.pi * self.roll_off * time[idx])
            / (1 - (2 * self.roll_off * time[idx]) ** 2)
        )
        if np.any(idx_0_by_0):
            impulse_response[idx_0_by_0] = np.pi / 4 * np.sinc(1 / (2 * self.roll_off))

        return impulse_response / np.linalg.norm(impulse_response)


class RectangularWaveform(FilteredSingleCarrierWaveform, Serializable):
    """Rectangular filtered single carrier modulation."""

    __DEFAULT_RELATIVE_BANDWIDTH: float = 1.0  # Default pulse bandwidth relative to the symbol rate

    __relative_bandwidth: float

    def __init__(
        self, relative_bandwidth: float = __DEFAULT_RELATIVE_BANDWIDTH, *args, **kwargs
    ) -> None:

        # Init base class
        FilteredSingleCarrierWaveform.__init__(self, *args, **kwargs)

        # Init attributes
        self.relative_bandwidth = relative_bandwidth

    @property
    def relative_bandwidth(self) -> float:
        """Bandwidth relative to the configured symbol rate.

        Raises:
            ValueError: On values smaller or equal to zero.
        """

        return self.__relative_bandwidth

    @relative_bandwidth.setter
    def relative_bandwidth(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("Relative pulse bandwidth must be greater than zero")

        self.__relative_bandwidth = value

    @property
    def bandwidth(self) -> float:
        return self.symbol_rate * self.relative_bandwidth

    def _transmit_filter(self) -> np.ndarray:
        pulse_width = int(self.oversampling_factor / self.relative_bandwidth)
        return np.ones(pulse_width, dtype=complex) / np.sqrt(pulse_width)

    def _receive_filter(self) -> np.ndarray:
        return self._transmit_filter()

    @property
    def _filter_delay(self) -> int:
        return int(self.oversampling_factor / self.relative_bandwidth) - 1

    @override
    def serialize(self, process: SerializationProcess) -> None:
        FilteredSingleCarrierWaveform.serialize(self, process)
        process.serialize_floating(self.relative_bandwidth, "relative_bandwidth")

    @override
    @classmethod
    def _DeserializeParameters(cls, process: DeserializationProcess) -> dict[str, Any]:
        params = FilteredSingleCarrierWaveform._DeserializeParameters(process)
        params["relative_bandwidth"] = process.deserialize_floating(
            "relative_bandwidth", cls.__DEFAULT_RELATIVE_BANDWIDTH
        )
        return params


class FMCWWaveform(FilteredSingleCarrierWaveform, Serializable):
    """Frequency Modulated Continuous Waveform Filter Modulation Scheme."""

    __bandwidth: float  # Chirp bandwidth in Hz
    __chirp_duration: float  # Chirp duration in seconds

    def __init__(self, bandwidth: float, chirp_duration: float = 0.0, *args, **kwargs) -> None:
        """
        Args:

            bandwidth:
                The chirp bandwidth in Hz.

            chirp_duration:
                Duration of each FMCW chirp in seconds.
                By default, the inverse symbol rate is assumed.
        """

        self.bandwidth = bandwidth
        self.chirp_duration = chirp_duration

        FilteredSingleCarrierWaveform.__init__(self, *args, **kwargs)

    @property
    def chirp_duration(self) -> float:
        """FMCW Chirp duration.

        A duration of zero will result in the inverse symbol rate as chirp duration.

        Raises:
            ValueError: If the duration is smaller than zero.
        """

        return self.__chirp_duration

    @chirp_duration.setter
    def chirp_duration(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("Chirp duration must be greater or equal to zero")

        self.__chirp_duration = value

    @property
    def __true_chirp_duration(self) -> float:
        """Chirp duration for internal calculations."""

        if self.chirp_duration <= 0.0:
            return 1 / self.symbol_rate

        return self.chirp_duration

    @property
    def bandwidth(self) -> float:
        return self.__bandwidth

    @bandwidth.setter
    def bandwidth(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("Chirp bandwidth must be greater than zero")

        self.__bandwidth = value

    @property
    def chirp_slope(self) -> float:
        """Chirp slope.

        The slope is equal to the chirp bandwidth divided by its duration."""

        return self.bandwidth / self.__true_chirp_duration

    def _transmit_filter(self) -> np.ndarray:
        time = np.linspace(0, 1 / self.symbol_rate, self.oversampling_factor)
        impulse_response = np.exp(1j * np.pi * (self.bandwidth * time + self.chirp_slope * time**2))
        # Cut off the chirp appropriately
        impulse_response[time > self.__true_chirp_duration] = 0.0

        return impulse_response / np.linalg.norm(impulse_response)

    def _receive_filter(self) -> np.ndarray:
        return np.flip(self._transmit_filter().conj())

    @property
    def _filter_delay(self) -> int:
        return self.oversampling_factor - 1

    @override
    def serialize(self, process: SerializationProcess) -> None:
        FilteredSingleCarrierWaveform.serialize(self, process)
        process.serialize_floating(self.bandwidth, "bandwidth")
        process.serialize_floating(self.chirp_duration, "chirp_duration")

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> FMCWWaveform:
        return cls(
            process.deserialize_floating("bandwidth"),
            process.deserialize_floating("chirp_duration"),
            **cls._DeserializeParameters(process),
        )
