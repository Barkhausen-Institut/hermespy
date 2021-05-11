from typing import Dict, Tuple, Any, List

import numpy as np
from scipy import integrate

from modem.waveform_generator import WaveformGenerator
from parameters_parser.parameters_chirp_fsk import ParametersChirpFsk


class WaveformGeneratorChirpFsk(WaveformGenerator):
    """ Implements a chirp FSK waveform generator."""

    def __init__(self, param: ParametersChirpFsk) -> None:
        super().__init__(param)
        self.param = param

        # derived parameters

        self._samples_in_chirp = int(
            np.around(
                self.param.chirp_duration *
                self.param.sampling_rate))
        self._chirps_in_frame = self.param.number_pilot_chirps + \
            self.param.number_data_chirps
        self._samples_in_frame = (self._samples_in_chirp *
                                  self._chirps_in_frame +
                                  int((np.around(self.param.guard_interval *
                                                 self.param.sampling_rate))))

        self._f0 = - self.param.chirp_bandwidth / 2
        self._f1 = -self._f0
        self._slope = self.param.chirp_bandwidth / self.param.chirp_duration
        self._chirp_time = np.arange(
            self._samples_in_chirp) / self.param.sampling_rate

        self._prototype_function: Dict[str, Any] = {}
        self._symbol_energy: float = 0

        self._build_prototype_functions()

    def create_frame(self, timestamp: int,
                     data_bits: np.array) -> Tuple[np.ndarray, int, int]:
        offset = self._calculate_frequency_offsets(data_bits)

        # calculate initial frequencies of the chirps
        initial_frequency = self._f0 + offset * self.param.freq_difference
        initial_frequency = np.concatenate(
            (np.ones(self.param.number_pilot_chirps) * self._f0, initial_frequency))
        frequency, amplitude = self._calculate_chirp_frequencies(
            initial_frequency)

        phase = 2 * np.pi * \
            integrate.cumtrapz(
                frequency,
                dx=1 /
                self.param.sampling_rate,
                initial=0)

        output_signal = amplitude * np.exp(1j * phase)

        initial_sample_num = timestamp
        timestamp += self._samples_in_frame

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
        power_of_2 = 2 ** np.arange(self.param.bits_per_symbol - 1, -1, -1)
        bits = np.reshape(
            data_bits,
            (self.param.bits_per_symbol,
             self.param.number_data_chirps),
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

        amplitude = np.zeros(self._samples_in_frame, dtype=complex)
        frequency = np.zeros(self._samples_in_frame, dtype=complex)

        for symbol_index, f0 in enumerate(initial_frequency):
            first_sample = symbol_index * self._samples_in_chirp
            last_sample = first_sample + self._samples_in_chirp

            amplitude[first_sample:last_sample] = 1
            frequency[first_sample:last_sample] = f0 + \
                self._slope * self._chirp_time  # set (modulated) chirp

        frequency[frequency > self._f1] -= self.param.chirp_bandwidth  # wrap

        return frequency, amplitude

    def receive_frame(self,
                      rx_signal: np.ndarray,
                      timestamp_in_samples: int,
                      noise_var: float) -> Tuple[List[np.array],
                                                       np.ndarray]:
        useful_signal_length = self._samples_in_chirp * self._chirps_in_frame

        if rx_signal.shape[1] < useful_signal_length:
            bits = None
            rx_signal = np.array([])
        else:
            bits = np.zeros(
                (self.param.number_data_chirps, self.param.bits_per_symbol), dtype=int)
            frame_signal = rx_signal[0, :useful_signal_length]

            # remove pilots
            frame_signal = frame_signal[self._samples_in_chirp *
                                        self.param.number_pilot_chirps:]

            for symbol_idx in range(self.param.number_data_chirps):
                symbol_signal = frame_signal[:self._samples_in_chirp]
                frame_signal = frame_signal[self._samples_in_chirp:]
                symbol_metric = np.zeros(self.param.modulation_order)

                for signal_idx in range(self.param.modulation_order):
                    real_signal = np.real(symbol_signal)
                    imag_signal = np.imag(symbol_signal)
                    cos_metric = np.sum(
                        real_signal *
                        np.real(
                            self._prototype_function['cos'][signal_idx]) +
                        imag_signal *
                        np.imag(
                            self._prototype_function['cos'][signal_idx])) ** 2
                    sin_metric = np.sum(
                        real_signal *
                        np.real(
                            self._prototype_function['sin'][signal_idx]) +
                        imag_signal *
                        np.imag(
                            self._prototype_function['sin'][signal_idx]))**2
                    symbol_metric[signal_idx] = cos_metric + sin_metric

                symbol_est = np.argmax(symbol_metric)
                bits[symbol_idx, :] = [int(x) for x in list(
                    np.binary_repr(symbol_est, width=self.param.bits_per_symbol))]

            rx_signal = rx_signal[:, self._samples_in_frame:]

        bits = np.ravel(bits)

        return list([bits]), rx_signal

    def get_bit_energy(self) -> float:
        return self._symbol_energy / self.param.bits_per_symbol

    def get_symbol_energy(self) -> float:
        return self._symbol_energy

    def get_power(self) -> float:
        return self._symbol_energy / self._samples_in_chirp

    def _build_prototype_functions(self) -> None:
        """
        modem._build_prototype_functions
        This method generates the prototype chirps for all possible modulation symbols, that will be correlated with the
        received signal for detection.
        These are stored in the variable 'self._prototype_function'. This function also calculates the energy of a
        modulation symbol in 'self._symbol_energy'
        """
        # non-coherent detection
        cos_signal = np.zeros(
            (2 ** self.param.bits_per_symbol,
             self._samples_in_chirp),
            dtype=complex)
        sin_signal = np.zeros(
            (2 ** self.param.bits_per_symbol,
             self._samples_in_chirp),
            dtype=complex)

        for idx in range(self.param.modulation_order):
            initial_frequency = self._f0 + idx * self.param.freq_difference
            frequency = initial_frequency + self._chirp_time * self._slope
            frequency[frequency > self._f1] -= self.param.chirp_bandwidth

            phase = 2 * np.pi * \
                integrate.cumtrapz(
                    frequency,
                    dx=1 /
                    self.param.sampling_rate,
                    initial=0)
            cos_signal[idx, :] = np.exp(1j * phase)
            sin_signal[idx, :] = np.exp(1j * (phase - np.pi / 2))

        self._symbol_energy = sum(abs(cos_signal[0, :])**2)
        self._prototype_function = {'cos': cos_signal, 'sin': sin_signal}
