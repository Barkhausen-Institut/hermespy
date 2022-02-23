# -*- coding: utf-8 -*-
"""
============================
Chirp Frequency Shift Keying
============================
"""

from __future__ import annotations
from typing import Optional, Tuple, Type
from math import ceil
from functools import lru_cache

import numpy as np
from ruamel.yaml import SafeConstructor, SafeRepresenter, Node
from scipy import integrate

from hermespy.core.factory import Serializable
from hermespy.channel import ChannelStateInformation
from hermespy.modem.waveform_generator import PilotWaveformGenerator, WaveformGenerator, Synchronization
from hermespy.core.signal_model import Signal
from .symbols import Symbols
from .waveform_correlation_synchronization import CorrelationSynchronization

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.7"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class WaveformGeneratorChirpFsk(PilotWaveformGenerator, Serializable):
    """ Implements a chirp FSK waveform generator."""

    # YAML tag
    yaml_tag = u'ChirpFsk'

    # Modulation parameters
    symbol_type: np.dtype = int
    synchronization: ChirpFskSynchronization
    __chirp_duration: float
    __chirp_bandwidth: float
    __freq_difference: float

    # Frame parameters
    __num_pilot_chirps: int
    __num_data_chirps: int
    __guard_interval: float

    def __init__(self,
                 chirp_duration: float = None,
                 chirp_bandwidth: float = None,
                 freq_difference: float = None,
                 num_pilot_chirps: int = None,
                 num_data_chirps: int = None,
                 guard_interval: float = None,
                 **kwargs) -> None:
        """Frequency Shift Keying Waveform Generator object initialization.

        Args:

            chirp_duration (float, optional):
                Duration of a single chirp in seconds.

            chirp_bandwidth (float, optional):
                Bandwidth of a single chirp in Hz.

            kwargs:
                Base waveform generator initialization arguments.
        """

        # Init base class
        WaveformGenerator.__init__(self, **kwargs)

        # Default parameters
        self.synchronization = ChirpFskSynchronization()
        self.__chirp_duration = 1e-6
        self.__chirp_bandwidth = 10e6
        self.__freq_difference = self.__chirp_bandwidth / self.modulation_order
        self.__num_pilot_chirps = 0
        self.__num_data_chirps = 20
        self.__guard_interval = 0.0

        if chirp_duration is not None:
            self.chirp_duration = chirp_duration

        if chirp_bandwidth is not None:
            self.chirp_bandwidth = chirp_bandwidth

        if freq_difference is not None:
            self.freq_difference = freq_difference

        if num_pilot_chirps is not None:
            self.num_pilot_chirps = num_pilot_chirps

        if num_data_chirps is not None:
            self.num_data_chirps = num_data_chirps

        if guard_interval is not None:
            self.guard_interval = guard_interval

    @classmethod
    def to_yaml(cls: Type[WaveformGeneratorChirpFsk],
                representer: SafeRepresenter,
                node: WaveformGeneratorChirpFsk) -> Node:
        """Serialize an `WaveformGenerator` object to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (WaveformGenerator):
                The `WaveformGeneratorChirpFsk` instance to be serialized.

        Returns:
            Node:
                The serialized YAML node
        """

        state = {
            "chirp_duration": node.chirp_duration,
            "chirp_bandwidth": node.chirp_bandwidth,
            "freq_difference": node.freq_difference,
            "num_pilot_chirps": node.num_pilot_chirps,
            "num_data_chirps": node.num_data_chirps,
            "guard_interval": node.guard_interval,
        }

        mapping = representer.represent_mapping(cls.yaml_tag, state)
        mapping.value.extend(WaveformGenerator.to_yaml(representer, node).value)

        return mapping

    @classmethod
    def from_yaml(cls: Type[WaveformGeneratorChirpFsk], constructor: SafeConstructor, node: Node)\
            -> WaveformGeneratorChirpFsk:
        """Recall a new `WaveformGeneratorChirpFsk` instance from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `WaveformGeneratorChirpFsk` serialization.

        Returns:
            WaveformGenerator:
                Newly created `WaveformGeneratorChirpFsk` instance.
        """

        state = constructor.construct_mapping(node)
        return cls(**state)

    @property
    def frame_duration(self) -> float:
        """Length of one data frame in seconds.

        Returns:
            float: Frame length in seconds.
        """

        return self.chirp_duration * (self.num_data_chirps + self.num_pilot_chirps) + self.guard_interval

    @property
    def chirp_duration(self) -> float:
        """Access the chirp duration.

        Returns:
            float:
                Chirp duration in seconds.
        """

        return self.__chirp_duration

    @chirp_duration.setter
    def chirp_duration(self, duration: float) -> None:
        """Modify the chirp duration.

        Args:
            duration (float):
                The new duration in seconds.

        Raises:
            ValueError:
                If the duration is less or equal to zero.
        """

        if duration < 0.0:
            raise ValueError("Chirp duration must be greater than zero")

        self.__chirp_duration = duration
        self._clear_cache()

    @property
    def chirp_bandwidth(self) -> float:
        """Access the chirp bandwidth.

        Returns:
            float:
                The chirp bandwidth in Hz.
        """

        return self.__chirp_bandwidth

    @chirp_bandwidth.setter
    def chirp_bandwidth(self, bandwidth: float) -> None:
        """Modify the chirp bandwidth.

        Args:
            bandwidth (float):
                The new bandwidth in Hz.

        Raises:
            ValueError:
                If the bandwidth is les sor equal to zero.
        """

        if bandwidth <= 0.0:
            raise ValueError("Chirp bandwidth must be greater than zero")

        self.__chirp_bandwidth = bandwidth
        self._clear_cache()

    @property
    def freq_difference(self) -> float:
        """Access the frequency difference.

        Returns:
            float:
                The frequency difference in Hz.
        """

        return self.__freq_difference

    @freq_difference.setter
    def freq_difference(self, difference: float) -> None:
        """Modify the frequency difference.

        Args:
            difference (float):
                The new frequency difference in Hz.

        Raises:
            ValueError:
                If the frequency `difference` is less or equal to zero.
                If the frequency `difference` is larger or equal to the configured `chirp bandwidth`.
        """

        if difference <= 0.0:
            raise ValueError("The frequency difference must be greater than zero")

        if difference >= self.chirp_bandwidth:
            raise ValueError("The frequency difference must be smaller than the configured chirp bandwidth")

        self.__freq_difference = difference
        self._clear_cache()

    @property
    def num_pilot_chirps(self) -> int:
        """Access the number of pilot chirps.

        Returns:
            int:
                The number of pilot chirps.
        """

        return self.__num_pilot_chirps

    @num_pilot_chirps.setter
    def num_pilot_chirps(self, num: int) -> None:
        """Modify the number of pilot chirps

        Args:
            num (int):
                The new number of pilot chirps.

        Raises:
            ValueError:
                If the `num`ber of pilot chirps is less than zero.
        """

        if num < 0:
            raise ValueError("The number of pilot chirps must be greater or equal to zero.")

        self.__num_pilot_chirps = num
        self._clear_cache()

    @property
    def num_data_chirps(self) -> int:
        """Access the number of data chirps.

        Returns:
            int:
                The number of data chirps.
        """

        return self.__num_data_chirps

    @num_data_chirps.setter
    def num_data_chirps(self, num: int) -> None:
        """Modify the number of pilot chirps

        Args:
            num (int):
                The new number of data chirps.

        Raises:
            ValueError:
                If the `num`ber of data chirps is less than zero.
        """

        if num < 0:
            raise ValueError("The number of data chirps must be greater or equal to zero")

        self.__num_data_chirps = num

    @property
    def guard_interval(self) -> float:
        """Access the guard interval.

        Returns:
            float:
                The guard interval in seconds.
        """

        return self.__guard_interval

    @guard_interval.setter
    def guard_interval(self, interval: float) -> None:
        """Modify the guard interval.

        Args:
            interval (float):
                The new guard `interval` in seconds.

        Raises:
            ValueError:
                If the frequency guard `interval` is less than zero.
        """

        if interval < 0.0:
            raise ValueError("The guard interval must be greater or equal to zero.")

        self.__guard_interval = interval

    @property
    def bits_per_symbol(self) -> int:
        """The number of bits per generated symbol.

        Returns:
            int:
                The number of bits.
        """

        return int(np.log2(self.modulation_order))

    @property
    def bits_per_frame(self) -> int:

        return self.num_data_chirps * self.bits_per_symbol

    @property
    def symbols_per_frame(self) -> int:

        return self.num_data_chirps

    @property
    def samples_in_chirp(self) -> int:
        """The number of discrete samples per generated chirp.

        Returns:
            int:
                The number of samples.
        """

        return int(ceil(self.chirp_duration * self.sampling_rate))

    @property
    def chirps_in_frame(self) -> int:
        """The number of chirps per generated frame.

        Returns:
            int:
                The number of chirps.
        """

        return self.num_pilot_chirps + self.num_data_chirps

    @property
    def chirp_time(self) -> np.ndarray:
        """Chirp timestamps.

        Returns:
            array:
                Chirp timestamps.
        """

        return np.arange(self.samples_in_chirp) / self.sampling_rate

    @property
    def samples_in_frame(self) -> int:
        """The number of discrete samples per generated frame.

        Returns:
            int:
                The number of samples.
        """

        return (self.samples_in_chirp * self.chirps_in_frame +
                int((np.around(self.__guard_interval * self.sampling_rate))))

    @property
    def symbol_energy(self) -> float:
        """The theoretical average symbol (discrete-time) energy of the modulated signal.

        Energy of signal x[k] is defined as \\sum{|x[k]}^2
        Only data bits are considered, i.e., reference, guard intervals are ignored.

        Returns:
            The average symbol energy in UNIT.
        """

        _, energy = self._prototypes()
        return energy

    @property
    def bit_energy(self) -> float:
        """Theoretical average bit energy of the modulated signal.

        Returns:
            The average bit energy in UNIT.
        """

        _, symbol_energy = self._prototypes()
        bit_energy = symbol_energy / self.bits_per_symbol
        return bit_energy

    def map(self, data_bits: np.ndarray) -> Symbols:

        offset = self._calculate_frequency_offsets(data_bits)
        return Symbols(offset)

    def modulate(self, data_symbols: Symbols) -> Signal:

        prototypes, _ = self._prototypes()
        samples = np.empty(self.samples_in_frame, dtype=complex)

        sample_idx = 0
        samples_in_chirp = self.samples_in_chirp

        # Add pilot samples
        pilot_samples = self.pilot.samples.flatten()
        num_pilot_samples = len(pilot_samples)
        samples[:num_pilot_samples] = pilot_samples
        sample_idx += num_pilot_samples

        # Modulate data symbols
        for symbol in data_symbols.raw[0, :]:

            samples[sample_idx:sample_idx+samples_in_chirp] = prototypes[symbol, :]
            sample_idx += samples_in_chirp

        return Signal(samples, self.sampling_rate)

    def demodulate(self,
                   baseband_signal: np.ndarray,
                   channel_state: ChannelStateInformation,
                   noise_variance: float) -> Tuple[Symbols, ChannelStateInformation, np.ndarray]:

        # Assess number of frames contained within this signal
        samples_in_chirp = self.samples_in_chirp
        samples_in_pilot_section = samples_in_chirp * self.num_pilot_chirps
        prototypes, _ = self._prototypes()

        data_frame = baseband_signal[samples_in_pilot_section:]

        symbol_signals = data_frame.reshape(-1, self.samples_in_chirp)
        symbol_metrics = abs(symbol_signals @ prototypes.T.conj())

        # ToDo: Unfortunately the demodulation-scheme is non-linear. Is there a better way?
        symbols = np.argmax(symbol_metrics, axis=1)
        channel_state = ChannelStateInformation.Ideal(len(symbols),
                                                      channel_state.num_transmit_streams,
                                                      channel_state.num_receive_streams)
        noises = np.repeat(noise_variance, self.num_data_chirps)

        return Symbols(symbols), channel_state, noises

    def unmap(self, data_symbols: Symbols) -> np.ndarray:

        bits_per_symbol = self.bits_per_symbol
        bits = np.empty(data_symbols.num_symbols * self.bits_per_symbol)

        for s, symbol in enumerate(data_symbols.raw[0, :]):

            symbol_bits = [int(x) for x in list(np.binary_repr(int(symbol.real), width=bits_per_symbol))]
            bits[s*bits_per_symbol:(s+1)*bits_per_symbol] = symbol_bits

        return bits

    @property
    def bandwidth(self) -> float:

        # The bandwidth is identical to the chirp bandwidth
        return self.chirp_bandwidth

    def _calculate_frequency_offsets(self, data_bits: np.ndarray) -> np.ndarray:
        """Calculates the frequency offsets on frame creation.

        Args:
            data_bits (np.ndarray): Data bits to calculate the offsets for.

        Returns:
            np.array: Array of length `number_data_chirps`.
        """
        # convert bits to integer frequency offsets
        # e.g. [8, 4, 2, 1]
        power_of_2 = 2 ** np.arange(self.bits_per_symbol - 1, -1, -1)
        bits = np.reshape(data_bits, (self.bits_per_symbol, -1), order='F')

        # generate offset according to bits
        offset = np.matmul(power_of_2, bits)
        return offset

    @property
    def power(self) -> float:
        return self.symbol_energy / self.samples_in_chirp

    @lru_cache(maxsize=1, typed=True)
    def _prototypes(self) -> Tuple[np.array, float]:
        """Generate chirp prototypes.

        This method generates the prototype chirps for all possible modulation symbols, that will be correlated with the
        received signal for detection.

        Since the computation is quite costly, the most recent output will be cached.

        Returns: Tuple[np.array, np.array, float]
            np.array:
                Prototype.
            float:
                Symbol energy.
        """

        # Chirp parameter inference
        slope = self.chirp_bandwidth / self.chirp_duration
        chirp_time = np.arange(self.samples_in_chirp) / self.sampling_rate
        f0 = -.5 * self.chirp_bandwidth
        f1 = -f0

        # non-coherent detection
        prototypes = np.zeros((2 ** self.bits_per_symbol, self.samples_in_chirp), dtype=complex)

        for idx in range(self.modulation_order):
            initial_frequency = f0 + idx * self.freq_difference
            frequency = chirp_time * slope + initial_frequency
            frequency[frequency > f1] -= self.chirp_bandwidth

            phase = 2 * np.pi * integrate.cumtrapz(frequency, dx=1 / self.sampling_rate, initial=0)
            prototypes[idx, :] = np.exp(1j * phase)

        symbol_energy = sum(abs(prototypes[0, :])**2)
        return prototypes, symbol_energy

    @property
    def sampling_rate(self) -> float:

        # Sampling rate scales with the chirp bandwidth
        return self.oversampling_factor * self.__chirp_bandwidth

    @property
    def pilot(self) -> Signal:
        """Samples of the frame's pilot section.

        Returns:
            samples (np.ndarray): Pilot samples.
        """

        # Generate single pilot chirp prototype
        prototypes, _ = self._prototypes()

        pilot_samples = np.empty(self.samples_in_chirp * self.num_pilot_chirps, dtype=complex)
        for pilot_idx in range(self.num_pilot_chirps):

            pilot_samples[pilot_idx*self.samples_in_chirp:(1+pilot_idx)*self.samples_in_chirp] = \
                prototypes[pilot_idx % len(prototypes)]

        return Signal(pilot_samples, self.sampling_rate)

    def _clear_cache(self) -> None:
        """Clear cached properties because a parameter has changed."""
        pass


class ChirpFskSynchronization(Synchronization[WaveformGeneratorChirpFsk]):
    """Synchronization for chirp-based frequency shift keying communication waveforms."""

    def __init__(self,
                 waveform_generator: Optional[WaveformGeneratorChirpFsk] = None) -> None:
        """
        Args:

            waveform_generator (WaveformGenerator, optional):
                The waveform generator this synchronization routine is attached to.
        """

        Synchronization.__init__(self, waveform_generator)


class ChirpFskCorrelationSynchronization(CorrelationSynchronization[WaveformGeneratorChirpFsk]):
    """Correlation-based clock-synchronization for Chirp-FSK waveforms."""
    ...
