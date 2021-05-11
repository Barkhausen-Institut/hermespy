import numpy as np


class ShapingFilter:
    """Implements a shaping/reception filter for a communications link.

    Currently, raised-cosine, root-raised-cosine, rectangular and FMCW
    (frequency modulated continuous wave) filters are implemented.
    An FIR filter with truncated impulse response is created.
    The filter is normalized, i.e., the impulse response has unit energy.

    Attributes:
        samples_per_symbol (int): samples per modulation symbol
        number_of_samples (int): filter length in samples
        delay_in_samples (int): delay introduced by filter
        impulse_response (numpy.array): filter impulse response
    """

    def __init__(
            self,
            filter_type: str,
            samples_per_symbol: int,
            length_in_symbols: float = 16,
            bandwidth_factor: float = 1.0,
            roll_off: float = 0,
            is_matched: bool = False):
        """
        Creates an object for a transmission/reception filter.

        Args:
            filter_type (str): Determines filter, currently supported:
                - RAISED_COSINE
                - ROOT_RAISED_COSINE
                - RECTANGULAR
                - FMCW
                - NONE
            samples_per_symbol (int): number of samples per modulation symbol.
            length_in_symbols (int): filter length in modulation symbols.
            bandwidth_factor (float): filter bandwidth can be expanded/reduced by this factor
                                     (default = 1), relatively to the symbol rate.
                                     For (root)-raised cosine, the Nyquist symbol rate will be
                                     multiplied by this factor
                                     For rectangular pulses, the pulse width in time will be divided by this factor.
                                     For FMCW, the sweep bandwidth will be given by the symbol rate multiplied by this
                                     factor.
            roll_off (float): Roll off factor between 0 and 1. Only relevant for (root)-raised cosine filters.
            is_matched (bool): if True, then a matched filter is considered.
        """

        self.samples_per_symbol = samples_per_symbol
        self.number_of_samples = None
        self.delay_in_samples = None
        self.impulse_response = None

        if filter_type == "NONE":
            self.impulse_response = 1.0
            self.delay_in_samples = 0
            self.number_of_samples = 1

        elif filter_type == "RECTANGULAR":
            self.number_of_samples = np.int(
                np.round(
                    self.samples_per_symbol /
                    bandwidth_factor))
            self.delay_in_samples = int(self.number_of_samples / 2)
            self.impulse_response = np.ones(self.number_of_samples)

            if is_matched:
                self.delay_in_samples -= 1

        elif filter_type == "RAISED_COSINE" or filter_type == "ROOT_RAISED_COSINE":
            self.number_of_samples = int(
                self.samples_per_symbol * length_in_symbols)
            delay_in_symbols = int(
                np.floor(
                    self.number_of_samples /
                    2) /
                samples_per_symbol)
            self.delay_in_samples = delay_in_symbols * samples_per_symbol

            self.impulse_response = self._get_raised_cosine(
                filter_type, roll_off, bandwidth_factor)

        elif filter_type == "FMCW":
            self.number_of_samples = int(
                np.ceil(
                    self.samples_per_symbol *
                    length_in_symbols))
            self.delay_in_samples = int(self.number_of_samples / 2)

            chirp_slope = bandwidth_factor / length_in_symbols
            self.impulse_response = self._get_fmcw(
                length_in_symbols, chirp_slope)

            if is_matched:
                self.impulse_response = np.flip(np.conj(self.impulse_response))
                self.delay_in_samples -= 1

        else:
            raise ValueError(f"Shaping filter {filter_type} not supported")

        # normalization (filter energy should be equal to one)
        self.impulse_response = self.impulse_response / \
            np.linalg.norm(self.impulse_response)

    def filter(self, input_signal: np.array) -> np.array:
        """Filters the input signal with the shaping filter.

        Args:
            input_signal (np.array): Input signal with N samples to filter.

        Returns:
            np.array:
                Filtered signal with  `N + samples_per_symbol*length_in_symbols - 1` samples.
        """
        output = np.convolve(input_signal, self.impulse_response)
        return output

    def _get_raised_cosine(self, filter_type: str, roll_off: float,
                           bandwidth_expansion: float) -> np.ndarray:
        """Returns a raised-cosine or root-raised-cosine impulse response

        Args:
            filter_type (str): either 'RAISED_COSINE' or 'ROOT_RAISED_COSINE'
            roll_off (float): filter roll-off factor, between 0 and 1
            bandwidth_expansion (float): bandwidth scaling factor, relative to the symbol rate. If equal to one, then
                the filter is  built for a symbol rate 1/'self.samples_per_symbol'

        Returns:
            impulse_response (np.array): filter impulse response
        """
        delay_in_symbols = self.delay_in_samples / self.samples_per_symbol

        impulse_response = np.zeros(self.number_of_samples)

        # create time reference
        t_min = -delay_in_symbols
        t_max = self.number_of_samples / self.samples_per_symbol + t_min
        time = np.arange(
            t_min,
            t_max,
            1 / self.samples_per_symbol) * bandwidth_expansion

        if filter_type == "RAISED_COSINE":
            if roll_off != 0:
                # indices with division of zero by zero
                idx_0_by_0 = (abs(time) == 1 / (2 * roll_off))
            else:
                idx_0_by_0 = np.zeros_like(time, dtype=bool)
            idx = ~idx_0_by_0
            impulse_response[idx] = (np.sinc(time[idx]) * np.cos(np.pi * roll_off * time[idx])
                                     / (1 - (2 * roll_off * time[idx]) ** 2))
            if np.any(idx_0_by_0):
                impulse_response[idx_0_by_0] = np.pi / \
                    4 * np.sinc(1 / (2 * roll_off))
        else:  # ROOT_RAISED_COSINE
            idx_0_by_0 = (time == 0)  # indices with division of zero by zero

            if roll_off != 0:
                # indices with division by zero
                idx_x_by_0 = (abs(time) == 1 / (4 * roll_off))
            else:
                idx_x_by_0 = np.zeros_like(time, dtype=bool)
            idx = (~idx_0_by_0) & (~idx_x_by_0)

            impulse_response[idx] = ((np.sin(np.pi * time[idx] * (1 - roll_off)) +
                                      4 * roll_off * time[idx] * np.cos(np.pi * time[idx] * (1 + roll_off))) /
                                     (np.pi * time[idx] * (1 - (4 * roll_off * time[idx])**2)))
            if np.any(idx_x_by_0):
                impulse_response[idx_x_by_0] = roll_off / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(
                    np.pi / (4 * roll_off)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * roll_off)))
            impulse_response[idx_0_by_0] = 1 + roll_off * (4 / np.pi - 1)

        return impulse_response

    def _get_fmcw(self, chirp_duration_in_symbols: float, chirp_slope: float):
        """Returns an FMCW impulse response

        Args:
            chirp_duration_in_symbols (float):
            chirp_slope (float): chirp bandwidth / chirp duration

        Returns:
            impulse_response (np.array): filter impulse response
        """
        time = np.arange(self.number_of_samples) / self.samples_per_symbol

        bandwidth = np.abs(chirp_duration_in_symbols * chirp_slope)
        sign = np.sign(chirp_slope)

        impulse_response = np.exp(
            1j * np.pi * (-sign * bandwidth * time + chirp_slope * time**2))

        return impulse_response
