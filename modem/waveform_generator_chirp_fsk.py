from __future__ import annotations
from typing import Tuple, List
from functools import lru_cache
import numpy as np
from scipy import integrate

from modem import Modem
from modem.waveform_generator import WaveformGenerator


class WaveformGeneratorChirpFsk(WaveformGenerator):
    """ Implements a chirp FSK waveform generator."""

    # Modulation parameters
    __modulation_order: int
    __chirp_duration: float
    __chirp_bandwidth: float
    __freq_difference: float

    __oversampling_factor: int

    # Frame parameters
    __num_pilot_chirps: int
    __num_data_chirps: int
    __guard_interval: float

    def __init__(self,
                 modem: Modem = None,
                 sampling_rate: float = None,
                 modulation_order: int = None,
                 chirp_duration: float = None,
                 chirp_bandwidth: float = None,
                 freq_difference: float = None,
                 oversampling_factor: int = None,
                 num_pilot_chirps: int = None,
                 num_data_chirps: int = None,
                 guard_interval: float = None) -> None:
        """Object initialization."""

        WaveformGenerator.__init__(self,
                                   modem=modem,
                                   sampling_rate=sampling_rate)

        # Default parameters
        # TODO: Insert a valid default configuration
        self.__modulation_order = 0
        self.__chirp_duration = 0.0
        self.__chirp_bandwidth = 0.0
        self.__freq_difference = 0.0
        self.__oversampling_factor = 0
        self.__num_pilot_chirps = 0
        self.__num_data_chirps = 0
        self.__guard_interval = 0.0
        self.__bits_per_symbol = 0
        self.__bits_in_frame = 0

        if modulation_order is not None:
            self.modulation_order = modulation_order

        if chirp_duration is not None:
            self.chirp_duration = chirp_duration

        if chirp_bandwidth is not None:
            self.chirp_bandwidth = chirp_bandwidth

        if freq_difference is not None:
            self.freq_difference = freq_difference

        if oversampling_factor is not None:
            self.oversampling_factor = oversampling_factor

        if num_pilot_chirps is not None:
            self.num_pilot_chirps = num_pilot_chirps

        if num_data_chirps is not None:
            self.num_data_chirps = num_data_chirps

        if guard_interval is not None:
            self.guard_interval = guard_interval

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
            order (int):w
                The new modulation order.

        Raises:
            ValueError:
                If `order` is not a positive power of two.
        """

        if order <= 0 or (order & (order - 1)) != 0:
            raise ValueError("Modulation order must be a positive power of two")

        self.__modulation_order = order

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

        if duration < 0:
            raise ValueError("Chirp duration must be greater than zero")

        self.__chirp_duration = duration

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
    def bits_in_frame(self) -> int:
        """The number of bits per generated frame.

        Returns:
            int:
                The number of bits.
        """

        return self.num_data_chirps * self.bits_per_symbol

    @property
    def samples_in_chirp(self) -> int:
        """The number of discrete samples per generated chirp.

        Returns:
            int:
                The number of samples.
        """

        return int(np.around(self.chirp_duration * self.sampling_rate))

    @property
    def chirps_in_frame(self) -> int:
        """The number of chirps per generated chirp.

        Returns:
            int:
                The number of chirps.
        """

        return self.num_pilot_chirps + self.num_data_chirps

    @property
    def chirp_time(self) -> np.array:
        """Chirp timestamps.

        Returns:
            array:
                Chirp timestamps.
        """

        return np.arange(self.samples_in_chirp) / self.__sampling_rate

    @property
    def samples_in_frame(self) -> int:
        """The number of discrete samples per generated frame.

        Returns:
            int:
                The number of samples.
        """

        return (self.samples_in_chirp * self.chirps_in_frame +
                int((np.around(self.__guard_interval * self.__sampling_rate))))

    @property
    def symbol_energy(self) -> float:
        """The theoretical average symbol (discrete-time) energy of the modulated signal.

        Energy of signal x[k] is defined as \\sum{|x[k]}^2
        Only data bits are considered, i.e., reference, guard intervals are ignored.

        Returns:
            The average symbol energy in UNIT.
        """

        _, _, energy = self._prototypes(self.__bits_per_symbol,
                                        self.samples_in_chirp,
                                        self.__modulation_order,
                                        self.sampling_rate,
                                        self.__chirp_bandwidth,
                                        self.__chirp_duration,
                                        self.__freq_difference)
        return energy

    def create_frame(self, timestamp: int,
                     data_bits: np.array) -> Tuple[np.ndarray, int, int]:

        offset = self._calculate_frequency_offsets(data_bits)
        f0 = -.5 * self.chirp_bandwidth

        # calculate initial frequencies of the chirps
        initial_frequency = f0 + offset * self.__freq_difference
        initial_frequency = np.concatenate(
            (np.ones(self.num_pilot_chirps) * f0, initial_frequency))
        frequency, amplitude = self._calculate_chirp_frequencies(
            initial_frequency)

        phase = 2 * np.pi * \
            integrate.cumtrapz(
                frequency,
                dx=1 /
                self.__sampling_rate,
                initial=0)

        output_signal = amplitude * np.exp(1j * phase)

        initial_sample_num = timestamp
        timestamp += self.samples_in_frame

        return output_signal[np.newaxis, :], timestamp, initial_sample_num

    def _calculate_frequency_offsets(
            self, data_bits: List[np.array]) -> np.ndarray:
        """Calculates the frequency offsets on frame creation.

        Args:
            data_bits(List[np.array]): Data bits to calculate the offsets for.

        Returns:
            np.array: Array of length `number_data_chirps`.
        """
        # convert bits to integer frequency offsets
        # e.g. [8, 4, 2, 1]
        power_of_2 = 2 ** np.arange(self.__bits_per_symbol - 1, -1, -1)
        bits = np.reshape(
            data_bits,
            (self.bits_per_symbol,
             self.num_data_chirps),
            order='F')
        # generate offset according to bits
        offset = np.matmul(power_of_2, bits)

        return offset

    def _calculate_chirp_frequencies(
            self, initial_frequency: np.array) -> Tuple[np.array, np.array]:
        """Calculates the chirp frequencies.

        Args:
            initial_frequency (np.array): Initial frquencies of chirps.

        Returns:
            (np.array, np.array):
                `np.array`: complex array containing samples as frequencies of chirps.
                `np.array`: corresponding amplitudes.
        """

        amplitude = np.zeros(self.samples_in_frame, dtype=complex)
        frequency = np.zeros(self.samples_in_frame, dtype=complex)
        slope = self.__chirp_bandwidth / self.__chirp_duration

        for symbol_index, f0 in enumerate(initial_frequency):
            first_sample = symbol_index * self.samples_in_chirp
            last_sample = first_sample + self.samples_in_chirp

            amplitude[first_sample:last_sample] = 1
            frequency[first_sample:last_sample] = f0 + \
                slope * self.chirp_time  # set (modulated) chirp

        frequency[frequency > self._f1] -= self.__chirp_bandwidth  # wrap

        return frequency, amplitude

    def receive_frame(self,
                      rx_signal: np.ndarray,
                      timestamp_in_samples: int,
                      noise_var: float) -> Tuple[List[np.array], np.ndarray]:

        useful_signal_length = self.samples_in_chirp * self.chirps_in_frame

        if rx_signal.shape[1] < useful_signal_length:
            bits = None
            rx_signal = np.array([])
        else:
            bits = np.zeros(
                (self.num_data_chirps, self.__bits_per_symbol), dtype=int)
            frame_signal = rx_signal[0, :useful_signal_length]

            # remove pilots
            frame_signal = frame_signal[self.samples_in_chirp *
                                        self.num_pilot_chirps:]

            cos_prototype, sin_prototype, _ = self._prototypes(self.__bits_per_symbol,
                                                               self.samples_in_chirp,
                                                               self.__modulation_order,
                                                               self.sampling_rate,
                                                               self.__chirp_bandwidth,
                                                               self.__chirp_duration,
                                                               self.__freq_difference)

            for symbol_idx in range(self.num_data_chirps):
                symbol_signal = frame_signal[:self.samples_in_chirp]
                frame_signal = frame_signal[self.samples_in_chirp:]
                symbol_metric = np.zeros(self.__modulation_order)

                for signal_idx in range(self.__modulation_order):
                    real_signal = np.real(symbol_signal)
                    imag_signal = np.imag(symbol_signal)
                    cos_metric = np.sum(
                        real_signal *
                        np.real(
                            cos_prototype[signal_idx]) +
                        imag_signal *
                        np.imag(
                            cos_prototype[signal_idx])) ** 2
                    sin_metric = np.sum(
                        real_signal *
                        np.real(
                            sin_prototype[signal_idx]) +
                        imag_signal *
                        np.imag(
                            sin_prototype[signal_idx]))**2
                    symbol_metric[signal_idx] = cos_metric + sin_metric

                symbol_est = np.argmax(symbol_metric)
                bits[symbol_idx, :] = [int(x) for x in list(
                    np.binary_repr(symbol_est, width=self.__bits_per_symbol))]

            rx_signal = rx_signal[:, self.samples_in_frame:]

        bits = np.ravel(bits)

        return list([bits]), rx_signal

    def get_bit_energy(self) -> float:
        return self.symbol_energy / self.__bits_per_symbol

    def get_power(self) -> float:
        return self.symbol_energy / self.samples_in_chirp

    @lru_cache(maxsize=1, typed=True)
    def _prototypes(self,
                    bits_per_symbol: int,
                    samples_in_chirp: int,
                    modulation_order: int,
                    sampling_rate: float,
                    chirp_bandwidth: float,
                    chirp_duration: float,
                    freq_difference: float
                    ) -> Tuple[np.array, np.array, float]:
        """Generate chirp prototypes.

        This method generates the prototype chirps for all possible modulation symbols, that will be correlated with the
        received signal for detection.

        Since the computation is quite costly, the most recent output will be cached.

        Returns: Tuple[np.array, np.array, float]
            np.array:
                Cosine prototype.
            np.array:
                Sine prototype.
            float:
                Symbol energy.
        """

        # Chirp parameter inference
        slope = chirp_bandwidth / chirp_duration
        chirp_time = np.arange(samples_in_chirp) / sampling_rate
        f0 = -.5 * chirp_bandwidth
        f1 = -f0

        # non-coherent detection
        cos_signal = np.zeros(
            (2 ** bits_per_symbol,
             samples_in_chirp),
            dtype=complex)
        sin_signal = np.zeros(
            (2 ** bits_per_symbol,
             samples_in_chirp),
            dtype=complex)

        for idx in range(modulation_order):
            initial_frequency = f0 + idx * freq_difference
            frequency = initial_frequency + chirp_time * slope
            frequency[frequency > f1] -= chirp_bandwidth

            phase = 2 * np.pi * \
                integrate.cumtrapz(
                    frequency,
                    dx=1 /
                    sampling_rate,
                    initial=0)
            cos_signal[idx, :] = np.exp(1j * phase)
            sin_signal[idx, :] = np.exp(1j * (phase - np.pi / 2))

        symbol_energy = sum(abs(cos_signal[0, :])**2)
        return cos_signal, sin_signal, symbol_energy
