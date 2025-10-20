# -*- coding: utf-8 -*-

from __future__ import annotations
from importlib.metadata import version
from typing_extensions import override
from math import ceil
from functools import cache

import numpy as np
from scipy import integrate

from hermespy.core import Serializable, SerializationProcess, DeserializationProcess
from hermespy.modem.waveform import PilotCommunicationWaveform, StatedSymbols, Synchronization
from hermespy.core.signal_model import Signal
from .symbols import Symbols
from .waveform_correlation_synchronization import CorrelationSynchronization

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


scipy_minor_version = int(version("scipy").split(".")[1])


class ChirpFSKWaveform(PilotCommunicationWaveform):
    """Chirp Frequency Shift Keying communication waveform  description."""

    DERIVE_FREQ_DIFFERENCE: float = 0.0
    """Magic number to derive the frequency difference from the modulation order."""

    __DEFAULT_CHIRP_DURATION: float = 10e-9
    __DEFAULT_NUM_PILOT_CHIRPS: int = 14
    __DEFAULT_NUM_DATA_CHIRPS: int = 50
    __DEFAULT_GUARD_INTERVAL: float = 0.0

    # Modulation parameters
    symbol_type = np.int_
    synchronization: ChirpFSKSynchronization
    __chirp_duration: float  # Duraiton of a single chirp in seconds
    __freq_difference: float  # Frequency offset between two adjecent chirp symbols

    # Frame parameters
    __num_pilot_chirps: int
    __num_data_chirps: int
    __guard_interval: float

    def __init__(
        self,
        chirp_duration: float = __DEFAULT_CHIRP_DURATION,
        freq_difference: float = DERIVE_FREQ_DIFFERENCE,
        num_pilot_chirps: int = __DEFAULT_NUM_PILOT_CHIRPS,
        num_data_chirps: int = __DEFAULT_NUM_DATA_CHIRPS,
        guard_interval: float = __DEFAULT_GUARD_INTERVAL,
        **kwargs,
    ) -> None:
        """
        Args:

            chirp_duration:
                Duration of a single chirp in seconds.

            freq_difference:
                Frequency difference of two adjacent chirp symbols.

            num_pilot_chirps:
                Number of pilot symbols within a single frame.

            num_data_chirps:
                Number of data symbols within a single frame.

            guard_interval:
                Frame guard interval in seconds.

            kwargs:
                Base waveform generator initialization arguments.
        """

        # Init base class
        PilotCommunicationWaveform.__init__(self, **kwargs)

        # Configure waveform paramters
        self.synchronization = ChirpFSKSynchronization()
        self.chirp_duration = chirp_duration
        self.freq_difference = freq_difference
        self.num_pilot_chirps = num_pilot_chirps
        self.num_data_chirps = num_data_chirps
        self.guard_interval = guard_interval

    @override
    def frame_duration(self, bandwidth: float) -> float:
        return (
            self.chirp_duration * (self.num_data_chirps + self.num_pilot_chirps)
            + self.guard_interval
        )

    @property
    def chirp_duration(self) -> float:
        """Duration of a single chirp within a frame.

        Returns: Chirp duration in seconds.

        Raises:
            ValueError: If the duration is less or equal to zero.
        """

        return self.__chirp_duration

    @chirp_duration.setter
    def chirp_duration(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("Chirp duration must be greater than zero")

        self.__chirp_duration = value

    @property
    def freq_difference(self) -> float:
        """The frequency offset between neighbouring chirp symbols in Hz.

        If set to zero, the frequency offset will be calculated automatically based on the modulation order.

        Raises:
            ValueError: If `freq_difference` is smaller than zero.
        """

        return self.__freq_difference

    @freq_difference.setter
    def freq_difference(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("Frequency difference must be or equal to zero")

        self.__freq_difference = value

    @property
    def num_pilot_chirps(self) -> int:
        """Access the number of pilot chirps.

        Returns: The number of pilot chirps.
        """

        return self.__num_pilot_chirps

    @num_pilot_chirps.setter
    def num_pilot_chirps(self, num: int) -> None:
        """Modify the number of pilot chirps

        Args:
            num:
                The new number of pilot chirps.

        Raises:
            ValueError: If the `num`ber of pilot chirps is less than zero.
        """

        if num < 0:
            raise ValueError("The number of pilot chirps must be greater or equal to zero.")

        self.__num_pilot_chirps = num

    @property
    def num_data_chirps(self) -> int:
        """Access the number of data chirps.

        Returns: The number of data chirps.
        """

        return self.__num_data_chirps

    @num_data_chirps.setter
    def num_data_chirps(self, num: int) -> None:
        """Modify the number of pilot chirps

        Args:
            num: The new number of data chirps.

        Raises:
            ValueError: If the `num`ber of data chirps is less than zero.
        """

        if num < 0:
            raise ValueError("The number of data chirps must be greater or equal to zero")

        self.__num_data_chirps = num

    @property
    def guard_interval(self) -> float:
        """Access the guard interval.

        Returns: The guard interval in seconds.
        """

        return self.__guard_interval

    @guard_interval.setter
    def guard_interval(self, interval: float) -> None:
        """Modify the guard interval.

        Args:
            interval: The new guard `interval` in seconds.

        Raises:
            ValueError: If the frequency guard `interval` is less than zero.
        """

        if interval < 0.0:
            raise ValueError("The guard interval must be greater or equal to zero.")

        self.__guard_interval = interval

    @property
    def bits_per_symbol(self) -> int:
        """The number of bits per generated symbol."""

        return int(np.log2(self.modulation_order))

    @property
    @override
    def num_data_symbols(self) -> int:
        return self.num_data_chirps

    def samples_in_chirp(self, bandwidth: float, oversampling_factor: int) -> int:
        """The number of discrete samples per generated chirp.

        Args:
            bandwidth: The bandwidth of the chirp in Hz.
            oversampling_factor: The oversampling factor of the waveform.

        Returns:
            The number of samples per chirp.
        """

        return int(ceil(self.chirp_duration * bandwidth * oversampling_factor))

    @property
    def chirps_in_frame(self) -> int:
        """The number of chirps per generated frame."""

        return self.num_pilot_chirps + self.num_data_chirps

    @override
    def samples_per_frame(self, bandwidth: float, oversampling_factor: int) -> int:
        return self.samples_in_chirp(bandwidth, oversampling_factor) * self.chirps_in_frame + int(
            (np.around(self.__guard_interval * bandwidth * oversampling_factor))
        )

    @override
    def symbol_energy(self, bandwidth: float, oversampling_factor: int) -> float:
        # Since all chirps have a constant magnitude of 1,
        # the symbol energy is equal to the number of samples per chirp
        return self.samples_in_chirp(bandwidth, oversampling_factor)

    @override
    def map(self, data_bits: np.ndarray) -> Symbols:
        offset = self._calculate_frequency_offsets(data_bits)
        return Symbols(offset[np.newaxis, np.newaxis, :])

    @override
    def unmap(self, symbols: Symbols) -> np.ndarray:
        bits_per_symbol = self.bits_per_symbol
        bits = np.empty(symbols.num_symbols * self.bits_per_symbol)

        for s, symbol in enumerate(symbols.raw[0, ::].flat):
            symbol_bits = [
                int(x) for x in list(np.binary_repr(int(symbol.real), width=bits_per_symbol))
            ]
            bits[s * bits_per_symbol : (s + 1) * bits_per_symbol] = symbol_bits

        return bits

    @override
    def place(self, symbols: Symbols) -> Symbols:
        return symbols

    @override
    def pick(self, placed_symbols: StatedSymbols) -> StatedSymbols:
        return placed_symbols

    @override
    def modulate(
        self, data_symbols: Symbols, bandwidth: float, oversampling_factor: int
    ) -> np.ndarray:
        samples_in_chirp = self.samples_in_chirp(bandwidth, oversampling_factor)
        prototypes, _ = ChirpFSKWaveform._prototypes(
            bandwidth,
            oversampling_factor,
            self.chirp_duration,
            self.bits_per_symbol,
            self.modulation_order,
            self.freq_difference,
            samples_in_chirp,
        )
        frame_samples = np.empty(
            self.samples_per_frame(bandwidth, oversampling_factor), dtype=complex
        )

        sample_idx = 0

        # Add pilot samples
        pilot_samples = self.pilot_signal(bandwidth, oversampling_factor).view(np.ndarray).flatten()
        num_pilot_samples = len(pilot_samples)
        frame_samples[:num_pilot_samples] = pilot_samples
        sample_idx += num_pilot_samples

        # Modulate data symbols
        for symbol in data_symbols.raw[0, ::].flat:
            frame_samples[sample_idx : sample_idx + samples_in_chirp] = prototypes[
                int(symbol.real), :
            ]
            sample_idx += samples_in_chirp

        return frame_samples

    @override
    def demodulate(self, signal: np.ndarray, bandwidth: float, oversampling_factor: int) -> Symbols:
        # Assess number of frames contained within this signal
        samples_in_chirp = self.samples_in_chirp(bandwidth, oversampling_factor)
        samples_in_pilot_section = samples_in_chirp * self.num_pilot_chirps
        prototypes, _ = ChirpFSKWaveform._prototypes(
            bandwidth,
            oversampling_factor,
            self.chirp_duration,
            self.bits_per_symbol,
            self.modulation_order,
            self.freq_difference,
            samples_in_chirp,
        )

        data_frame = signal[
            samples_in_pilot_section : samples_in_pilot_section
            + samples_in_chirp * self.num_data_chirps
        ]

        symbol_signals = data_frame.reshape(-1, samples_in_chirp)
        symbol_metrics = abs(symbol_signals @ prototypes.T.conj())

        # ToDo: Unfortunately the demodulation-scheme is non-linear. Is there a better way?
        symbols = np.argmax(symbol_metrics, axis=1)
        return Symbols(symbols[np.newaxis, np.newaxis, :])

    def _calculate_frequency_offsets(self, data_bits: np.ndarray) -> np.ndarray:
        """Calculates the frequency offsets on frame creation.

        Args:
            data_bits: Data bits to calculate the offsets for.

        Returns: Array of length `number_data_chirps`.
        """
        # convert bits to integer frequency offsets
        # e.g. [8, 4, 2, 1]
        power_of_2 = 2 ** np.arange(self.bits_per_symbol - 1, -1, -1)
        bits = np.reshape(data_bits, (self.bits_per_symbol, -1), order="F")

        # generate offset according to bits
        offset = np.matmul(power_of_2, bits)
        return offset

    @property
    @override
    def power(self) -> float:
        # Chirp of constant magnitude 1
        return 1.0

    @property
    def symbol_precoding_support(self) -> bool:
        return False

    @cache
    @staticmethod
    def _prototypes(
        bandwidth: float,
        oversampling_factor: int,
        chirp_duration: float,
        bits_per_symbol: int,
        modulation_order: int,
        freq_difference: float,
        samples_in_chirp: int,
    ) -> tuple[np.ndarray, float]:
        """Generate chirp prototypes.

        This method generates the prototype chirps for all possible modulation symbols, that will be correlated with the
        received signal for detection.

        Since the computation is quite costly, the most recent output will be cached.

        Returns: Tuple of waveform prototypes and symbol energy.
        """

        # Chirp parameter inference
        sampling_rate = oversampling_factor * bandwidth
        freq_difference = (
            bandwidth / modulation_order
            if freq_difference == ChirpFSKWaveform.DERIVE_FREQ_DIFFERENCE
            else freq_difference
        )
        slope = bandwidth / chirp_duration
        chirp_time = np.arange(samples_in_chirp) / sampling_rate
        f0 = -0.5 * bandwidth
        f1 = -f0

        # non-coherent detection
        prototypes = np.zeros((2**bits_per_symbol, samples_in_chirp), dtype=complex)

        for idx in range(modulation_order):
            initial_frequency = f0 + idx * freq_difference
            frequency = chirp_time * slope + initial_frequency
            frequency[frequency > f1] -= bandwidth

            # integrate.cumtrapz was changed to cumulative_trapezoid in scipy 1.14.0
            phase: np.ndarray
            if scipy_minor_version < 14:
                phase = integrate.cumtrapz(frequency, dx=1 / sampling_rate, initial=0)  # type: ignore
            else:
                phase = integrate.cumulative_trapezoid(frequency, dx=1 / sampling_rate, initial=0)
            phase *= 2 * np.pi
            prototypes[idx, :] = np.exp(1j * phase)

        symbol_energy = sum(abs(prototypes[0, :]) ** 2)
        return prototypes, symbol_energy

    @override
    def pilot_signal(self, bandwidth: float, oversampling_factor: int) -> Signal:
        # Generate single pilot chirp prototype
        samples_in_chirp = self.samples_in_chirp(bandwidth, oversampling_factor)
        prototypes, _ = ChirpFSKWaveform._prototypes(
            bandwidth,
            oversampling_factor,
            self.chirp_duration,
            self.bits_per_symbol,
            self.modulation_order,
            self.freq_difference,
            samples_in_chirp,
        )

        pilot_samples = np.empty(samples_in_chirp * self.num_pilot_chirps, dtype=np.complex128)
        for pilot_idx in range(self.num_pilot_chirps):
            pilot_samples[pilot_idx * samples_in_chirp : (1 + pilot_idx) * samples_in_chirp] = (
                prototypes[pilot_idx % len(prototypes)]
            )

        return Signal.Create(pilot_samples.reshape((1, -1)), bandwidth * oversampling_factor)

    @override
    def serialize(self, process: SerializationProcess) -> None:
        PilotCommunicationWaveform.serialize(self, process)
        process.serialize_floating(self.chirp_duration, "chirp_duration")
        process.serialize_floating(self.freq_difference, "freq_difference")
        process.serialize_integer(self.num_pilot_chirps, "num_pilot_chirps")
        process.serialize_integer(self.num_data_chirps, "num_data_chirps")
        process.serialize_floating(self.guard_interval, "guard_interval")

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> ChirpFSKWaveform:
        return ChirpFSKWaveform(
            chirp_duration=process.deserialize_floating(
                "chirp_duration", ChirpFSKWaveform.__DEFAULT_CHIRP_DURATION
            ),
            freq_difference=process.deserialize_floating(
                "freq_difference", ChirpFSKWaveform.DERIVE_FREQ_DIFFERENCE
            ),
            num_pilot_chirps=process.deserialize_integer(
                "num_pilot_chirps", ChirpFSKWaveform.__DEFAULT_NUM_PILOT_CHIRPS
            ),
            num_data_chirps=process.deserialize_integer(
                "num_data_chirps", ChirpFSKWaveform.__DEFAULT_NUM_DATA_CHIRPS
            ),
            guard_interval=process.deserialize_floating(
                "guard_interval", ChirpFSKWaveform.__DEFAULT_GUARD_INTERVAL
            ),
            **cls._DeserializeParameters(process),
        )


class ChirpFSKSynchronization(Synchronization[ChirpFSKWaveform], Serializable):
    """Synchronization for chirp-based frequency shift keying communication waveforms."""

    def __init__(self, waveform: ChirpFSKWaveform | None = None) -> None:
        """
        Args:

            waveform:
                The waveform generator this synchronization routine is attached to.
        """

        Synchronization.__init__(self, waveform)


class ChirpFSKCorrelationSynchronization(CorrelationSynchronization[ChirpFSKWaveform]):
    """Correlation-based clock-synchronization for Chirp-FSK waveforms."""
