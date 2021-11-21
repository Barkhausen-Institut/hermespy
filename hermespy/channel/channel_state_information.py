# -*- coding: utf-8 -*-
"""Channel State Information model for wireless transmission links."""

from __future__ import annotations
from typing import Generator, Optional, List, Union
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from sparse import COO, diagonal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class ChannelStateFormat(Enum):
    """Format flag for wireless transmission link states."""

    IMPULSE_RESPONSE = 0        # Channel state in impulse response format
    FREQUENCY_SELECTIVITY = 1   # Channel state in frequency selectivity format


class ChannelStateDimension(Enum):
    """Dimension selection of channel state information."""

    RECEIVE_STREAMS = 0
    TRANSMIT_STREAMS = 1
    SAMPLES = 2
    INFORMATION = 3


class ChannelStateInformation:
    """State of a single wireless link between a transmitting and receiving modem.

    Attributes:

        __state_format(ChannelStateFormat):
            The current format of the channel state information.
            The format may change depending on the most recent format requests.

        __state (np.ndarray):
            The current channel state.
            A numpy tensor of dimension `num_receive_streams`x`num_transmit_streams`x`num_samples`x`state_information`.

            If the state is currently in the format impulse response,
            `num_samples` is the time domain of the channel state and `state_information` the delay taps.

            If the state is currently in frequency selectivity format,
            `num_samples` discrete frequency domain bins and `state_information` is of length one, containing the
            respective Fourier weights.

        __num_delay_taps (int):
            Number of delay taps in impulse-response mode.
            Recovers the 4th matrix dimension during conversions.

        __num_frequency_bins (int):
            Number of discrete frequency bins in frequency-selectivity mode.
            Recovers the 4th matrix dimension during conversions.
    """

    __state_format: ChannelStateFormat
    __state: np.ndarray
    __num_delay_taps: int
    __num_frequency_bins: int

    def __init__(self,
                 state_format: ChannelStateFormat,
                 state: np.ndarray,
                 num_delay_taps: Optional[int] = None,
                 num_frequency_bins: Optional[int] = None) -> None:
        """Channel State Information object initialization.

        Args:
            state_format (ChannelStateFormat):
                Format of the `state` from which to initialize the channel state information.

            state (np.ndarray):
                Channel state matrix.
                A numpy tensor of dimension
                `num_receive_streams`x`num_transmit_streams`x`num_samples`x`state_information`.

            num_delay_taps (int, optional):
                Number of delay taps in impulse-response mode.

            num_frequency_bins (int):
                Number of discrete frequency bins in frequency-selectivity mode..

        Raises:
            ValueError:
                If `state` dimensions are invalid.
        """

        self.set_state(state_format, state, num_delay_taps, num_frequency_bins)

    @property
    def state_format(self) -> ChannelStateFormat:
        """Current channel state format.

        Returns:
            ChannelStateFormat: The current channel state format.
        """

        return self.__state_format

    @property
    def state(self) -> np.ndarray:
        """Current channel state tensor.

        Returns:
            np.ndarray: The current channel state tensor.
        """

        return self.__state

    @state.setter
    def state(self, new_state: np.ndarray) -> None:
        """Modify the channel state tensor.

        Args:
            new_state (np.ndarray): The new channel state.

        Raises:
            ValueError: On invalid state dimensions.
        """

        self.set_state(self.__state_format, new_state)

    def set_state(self,
                  state_format: ChannelStateFormat,
                  state: np.ndarray,
                  num_delay_taps: Optional[int] = None,
                  num_frequency_bins: Optional[int] = None) -> None:
        """Set a new channel state.

        Args:
            state_format (ChannelStateFormat):
                Format of the `state` from which to initialize the channel state information.

            state (np.ndarray):
                Channel state matrix.
                A numpy tensor of dimension
                `num_receive_streams`x`num_transmit_streams`x`num_samples`x`state_information`.

            num_delay_taps (int, optional):
                Number of delay taps.

            num_frequency_bins (int, optional):
                Number of discrete frequency bins.

        Raises:
            ValueError:
                If `state` dimensions are invalid.
        """

        if state_format not in ChannelStateFormat:
            raise ValueError("Unknown channel state format flag")

        if state.ndim != 4:
            raise ValueError("Channel state tensor must be 4-dimensional")

        if num_delay_taps is None:
            num_delay_taps = state.shape[3]

        if num_frequency_bins is None:
            num_frequency_bins = state.shape[3]

        if num_delay_taps < 1:
            raise ValueError("Number of delay taps must be greater or equal to one")

        if num_frequency_bins < 1:
            raise ValueError("Number of frequency bins must be greater or equal to one")

        if state_format == ChannelStateFormat.IMPULSE_RESPONSE and num_delay_taps != state.shape[3]:
            raise ValueError("Number of delay taps must be equal to the last dimension of the impulse response")

#        if state_format == ChannelStateFormat.FREQUENCY_SELECTIVITY and state.shape[3] != 1:
#            raise ValueError("In frequency selectivity mode,"
#                             "the fourth channel state matrix dimension must be of size one")

        self.__state_format = state_format
        self.__state = state
        self.__num_delay_taps = num_delay_taps
        self.__num_frequency_bins = num_frequency_bins

    def to_impulse_response(self) -> ChannelStateInformation:
        """Access the channel state in time-domain.

        May convert the internal state format via FFT.

        Returns:
            ChannelStateInformation:
                The current channel tensor of dimensions
                `num_receive_streams`x`num_transmit_streams`x`num_samples`x`num_delay_taps`.
        """

        if self.__state_format == ChannelStateFormat.FREQUENCY_SELECTIVITY:

            self.__state = ifft(self.__state, axis=3, norm='ortho')
            self.__state_format = ChannelStateFormat.IMPULSE_RESPONSE

        return self

    def to_frequency_selectivity(self, num_bins: Optional[int] = None) -> ChannelStateInformation:
        """Access the channel state in frequency-domain.

        May convert the internal state format via FFT.

        Args:
            num_bins (int, optional):
                Number of discrete frequency bins.
                By default, this will be the number of time samples,
                i.e. a FFT without zero-padding will be performed.

        Returns:
            ChannelStateInformation:
                The current channel tensor of dimensions
                `num_receive_streams`x`num_transmit_streams`x`num_samples`x`num_frequency_bins`.
        """

        if self.__state_format == ChannelStateFormat.IMPULSE_RESPONSE:

            if num_bins is None:
                num_bins = self.__num_frequency_bins

            else:
                self.__num_frequency_bins = num_bins

            self.__state = fft(self.__state, axis=3, n=num_bins)
            self.__state = self.__state.reshape((self.num_receive_streams, self.num_transmit_streams, -1, 1))

            self.__state_format = ChannelStateFormat.FREQUENCY_SELECTIVITY

        return self

    @property
    def num_receive_streams(self) -> int:
        """Number of receive streams within this channel state.

        Returns:
            int: Number of receive streams.
        """

        return self.__state.shape[0]

    @property
    def num_transmit_streams(self) -> int:
        """Number of transmit streams within this channel state.

        Returns:
            int: Number of transmit streams.
        """

        return self.__state.shape[1]

    @property
    def num_samples(self) -> int:
        """Number of time-domain samples within this channel state.

        Returns:
            int: Number of samples.
        """

        return self.__state.shape[2]

    @property
    def num_symbols(self) -> int:
        """Number of symbols considered within this channel state.

        Returns:
            int: Number of symbols.
        """

        if self.__state_format == ChannelStateFormat.IMPULSE_RESPONSE:
            return self.__state.shape[2]

        if self.__state_format == ChannelStateFormat.FREQUENCY_SELECTIVITY:
            return self.__state.shape[2] * self.__state.shape[3]

    @property
    def num_delay_taps(self) -> int:
        """Number of taps within the delay response of the channel state.

        Returns:
            int: Number of taps.
        """

        return self.__num_delay_taps

    @property
    def linear(self) -> COO:
        """Convert the channel state to a linear transformation tensor.

        Returns:
            Sparse linear transformation tensor of dimension N_Rx x N_Tx x N_out x N_in.
        """

        if self.__state_format == ChannelStateFormat.IMPULSE_RESPONSE:
            return self.__impulse_response_transformation()

        if self.__state_format == ChannelStateFormat.FREQUENCY_SELECTIVITY:
            return self.__frequency_response_transformation()

        raise RuntimeError("To linear CSI conversion encountered invalid internal state format")

    @linear.setter
    def linear(self, transformation: Union[COO, np.ndarray]) -> None:
        """Set the channel state from a linear transformation tensor.

        Args:
            transformation (Union[COO, np.ndarray]):
                Linear transformation tensor.
        """

        if self.__state_format == ChannelStateFormat.IMPULSE_RESPONSE:
            self.__from_impulse_response(transformation)

        elif self.__state_format == ChannelStateFormat.FREQUENCY_SELECTIVITY:
            self.__from_frequency_selectivity(transformation)

        else:
            raise RuntimeError("To linear CSI conversion encountered invalid internal state format")

    def __impulse_response_transformation(self) -> COO:
        """Convert a channel impulse response to a linear transformation tensor.


        Returns:
            COO:
                Sparse linear transformation tensor of dimension N_Rx x N_Tx x T+L x T.
                Note that the slice over the last two dimensions will form a lower triangular matrix.
        """

        num_rx = self.num_receive_streams
        num_tx = self.num_transmit_streams
        num_taps = self.__state.shape[3]
        num_s = self.num_samples
        num_out = num_s + num_taps - 1
        num_in = num_s

        in_ids = np.repeat(np.arange(num_in), num_taps)
        out_ids = np.array([np.arange(num_taps) + t for t in range(num_in)]).flatten()
        rx_ids = np.arange(num_rx)
        tx_ids = np.arange(num_tx)

        coordinates = [rx_ids.repeat(num_tx * num_taps * num_in),
                       tx_ids.repeat(num_rx * num_taps * num_in).reshape((num_tx, -1), order='F').flatten(),
                       np.tile(out_ids, num_rx * num_tx),
                       np.tile(in_ids, num_rx * num_tx)]

        transformation = COO(coordinates, self.__state.flatten(), shape=(num_rx, num_tx, num_out, num_in))
        return transformation

    def __frequency_response_transformation(self) -> COO:
        """Convert a channel frequency response to a linear transformation tensor.


        Returns:
            COO:
                Sparse linear transformation tensor of dimension N_Rx x N_Tx x F*T x F*T.
                Note that the slice over the first and last dimension will be a diagonal matrix.
        """

        num_rx = self.num_receive_streams
        num_tx = self.num_transmit_streams
        num_s = self.num_samples
        num_frequencies = self.__state.shape[3]

        num_symbols = num_s * num_frequencies

        diagonal_ids = np.arange(num_symbols)
        rx_ids = np.arange(num_rx)
        tx_ids = np.arange(num_tx)

        coordinates = [rx_ids.repeat(num_tx * num_symbols),
                       np.tile(tx_ids.repeat(num_symbols), num_rx),     # ToDo: This is probably not completely correct
                       np.tile(diagonal_ids, num_rx * num_tx),
                       np.tile(diagonal_ids, num_rx * num_tx)]

        transformation = COO(coordinates, self.__state.flatten(),
                             shape=(num_rx, num_tx, num_symbols, num_symbols))
        return transformation

    def __from_impulse_response(self, transformation: Union[COO, np.ndarray]) -> None:

        for delay_idx in range(self.__num_delay_taps):

            diagonal_elements = diagonal(transformation, axis1=3, axis2=2, offset=delay_idx)
            self.__state[:, :, :diagonal_elements.shape[2], delay_idx] = diagonal_elements.todense()

    def __from_frequency_selectivity(self, transformation: Union[COO, np.ndarray]) -> None:

        diagonal_elements = diagonal(transformation, axis1=2, axis2=3)
        self.__state[:, :, :diagonal_elements.shape[2], :].flat = diagonal_elements.todense()

    @staticmethod
    def Ideal(num_receive_streams: int,
              num_transmit_streams: int,
              num_samples: int) -> ChannelStateInformation:
        """Initialize an ideal channel state.

        Args:
            num_receive_streams (int):
                Number of emerging data streams after channel propagation.

            num_transmit_streams (int):
                Number of data streams feeding into the channel before propagation.

            num_samples (int):
                Number of timestamps at which the channel state has been sampled.

        Returns:
            ChannelStateInformation:
                Ideal channel state information of a non-distorting channel.
        """

        state = np.ones((num_receive_streams, num_transmit_streams, num_samples, 1), dtype=complex)
        return ChannelStateInformation(ChannelStateFormat.IMPULSE_RESPONSE, state)

    def received_streams(self) -> Generator[ChannelStateInformation, ChannelStateInformation, None]:
        """Iterate over the received streams slices within this channel state.

        Returns:
            Generator: Generator.
        """

        for stream_idx, received_stream in enumerate(self.__state):

            updated_stream = yield ChannelStateInformation(self.__state_format, received_stream[np.newaxis, ...])

            if updated_stream is not None:
                self[stream_idx, ::] = updated_stream

    def samples(self) -> Generator[ChannelStateInformation, ChannelStateInformation, None]:
        """Iterate over the sample slices within this channel state.

        Returns:
            Generator: Generator.
        """

        for sample_idx in range(self.num_samples):
            yield ChannelStateInformation(self.__state_format, self.__state[:, :, [sample_idx], :],
                                          self.__num_delay_taps, self.__num_frequency_bins)

    def __getitem__(self, section: slice) -> ChannelStateInformation:
        """Slice the channel state information.

        Args:
            section (slice):
                Slice of the channel state.

        Returns:
            ChannelStateInformation:
                New channel state with a section according to `value` slice.
        """

        state_section = self.__state[section]
        num_delay_taps = self.__num_delay_taps if state_section.shape[3] == self.__state.shape[3] else None

        return ChannelStateInformation(self.__state_format, state_section, num_delay_taps)

    def __setitem__(self, key: slice, value: ChannelStateInformation) -> None:
        """Update the channel state information.

        Args:
            key (slice):
                Section of the channel state to be set.

            value (ChannelStateInformation):
                The information to be set.

        Raises:
            NotImplementedError:
                If the formats of `value` and this channel do not match.
        """

        if value.state_format != self.__state_format:
            raise NotImplementedError("Setting CSIs of a different type is not yet supported")

        self.__state[key] = value.__state

    @staticmethod
    def concatenate(elements: List[ChannelStateInformation],
                    dimension: ChannelStateDimension) -> ChannelStateInformation:

        states = [element.__state for element in elements]
        stack = np.concatenate(states, axis=dimension.value)

        # ToDo: Make this smarter, it's not generally correct
        state_format = elements[0].__state_format if len(elements) > 0 else ChannelStateFormat.IMPULSE_RESPONSE
        num_delay_taps = elements[0].__num_delay_taps if len(elements) > 0 else None

        return ChannelStateInformation(state_format, stack, num_delay_taps)

    def plot(self) -> None:
        """Visualize the internal channel state information.

        Plots the absolute values of all channel state weights.
        """

        fig, axes = plt.subplots(self.__state.shape[0], self.__state.shape[1], squeeze=False)
        for rx_id, receive_states in enumerate(self.__state):
            for tx_id, transmit_states in enumerate(receive_states):

                axes[rx_id, tx_id].imshow(abs(transmit_states))
