# -*- coding: utf-8 -*-
"""
===============================
Channel State Information Model
===============================
"""

from __future__ import annotations
from itertools import product
from typing import Generator, List, SupportsIndex, Tuple, Type
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
from h5py import Group
from scipy.fft import fft, ifft
from sparse import COO, SparseArray  # type: ignore

from .factory import HDFSerializable
from .signal_model import Signal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class ChannelStateFormat(Enum):
    """Format flag for wireless transmission link states."""

    IMPULSE_RESPONSE = 0  # Channel state in impulse response format
    FREQUENCY_SELECTIVITY = 1  # Channel state in frequency selectivity format


class ChannelStateDimension(Enum):
    """Dimension selection of channel state information."""

    RECEIVE_STREAMS = 0
    TRANSMIT_STREAMS = 1
    SAMPLES = 2
    INFORMATION = 3


class ChannelStateInformation(HDFSerializable):
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
    __state: np.ndarray | SparseArray
    __num_delay_taps: int
    __num_frequency_bins: int

    def __init__(self, state_format: ChannelStateFormat, state: np.ndarray | SparseArray = None, num_delay_taps: int | None = None, num_frequency_bins: int | None = None) -> None:
        """Channel State Information object initialization.

        Args:
            state_format (ChannelStateFormat):
                Format of the `state` from which to initialize the channel state information.

            state (np.ndarray | SparseArray, optional):
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
    def state(self) -> np.ndarray | SparseArray:
        """Current channel state tensor.

        Returns:
            np.ndarray: The current channel state tensor.
        """

        return self.__state

    @state.setter
    def state(self, new_state: np.ndarray | SparseArray) -> None:
        """Modify the channel state tensor.

        Args:
            new_state (np.ndarray | SparseArray): The new channel state.

        Raises:
            ValueError: On invalid state dimensions.
        """

        self.set_state(self.__state_format, new_state)

    def dense_state(self) -> np.ndarray:
        """Return the channel state in dense format.

        Note that this method will convert the channel state to dense format if it is currently in sparse format.
        This operation may be computationally expensive and should be avoided if possible.

        Returns: The channel state tensor in dense format.
        """

        return self.__state.todense() if isinstance(self.__state, (SparseArray)) else self.__state

    def set_state(self, state_format: ChannelStateFormat, state: np.ndarray | SparseArray = None, num_delay_taps: int | None = None, num_frequency_bins: int | None = None) -> None:
        """Set a new channel state.

        Args:
            state_format (ChannelStateFormat):
                Format of the `state` from which to initialize the channel state information.

            state (np.ndarray | SparseArray, optional):
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

        state = np.empty((0, 0, 0, 1), dtype=complex) if state is None else state

        if state.ndim != 4:
            raise ValueError("Channel state tensor must be 4-dimensional")

        if num_delay_taps is None:
            num_delay_taps = state.shape[3]

        if num_frequency_bins is None:
            num_frequency_bins = state.shape[3]

        # if num_delay_taps < 1:
        #    raise ValueError("Number of delay taps must be greater or equal to one")

        # if num_frequency_bins < 1:
        #    raise ValueError("Number of frequency bins must be greater or equal to one")

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
            self.__state = ifft(self.__state, axis=3)
            self.__state_format = ChannelStateFormat.IMPULSE_RESPONSE

        return self

    def to_frequency_selectivity(self, num_bins: int | None = None) -> ChannelStateInformation:
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

            self.__state = fft(self.dense_state()[:, :, :num_bins, :], axis=3, n=num_bins)

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

        else:  # Channel estate is in frequency selectivity format
            return self.__state.shape[2] * self.__state.shape[3]

    @property
    def num_delay_taps(self) -> int:
        """Number of taps within the delay response of the channel state.

        Returns:
            int: Number of taps.
        """

        return self.__num_delay_taps

    @property
    def linear(self) -> SparseArray:
        """Convert the channel state to a linear transformation tensor.

        Returns:
            Sparse linear transformation tensor of dimension N_Rx x N_Tx x N_out x N_in.
        """

        if self.__state_format == ChannelStateFormat.IMPULSE_RESPONSE:
            return self.__impulse_response_transformation()

        else:  # Channel estate is in frequency selectivity format
            return self.__frequency_response_transformation()

    def __impulse_response_transformation(self) -> SparseArray:
        """Convert a channel impulse response to a linear transformation tensor.


        Returns:
            SparseArray:
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

        coordinates = [rx_ids.repeat(num_tx * num_taps * num_in), tx_ids.repeat(num_rx * num_taps * num_in).reshape((num_tx, -1), order="F").flatten(), np.tile(out_ids, num_rx * num_tx), np.tile(in_ids, num_rx * num_tx)]

        transformation = COO(coordinates, self.__state.flatten(), shape=(num_rx, num_tx, num_out, num_in))
        return transformation

    def __frequency_response_transformation(self) -> SparseArray:
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

        coordinates = [
            rx_ids.repeat(num_tx * num_symbols),
            # ToDo: This is probably not completely correct
            np.tile(tx_ids.repeat(num_symbols), num_rx),
            np.tile(diagonal_ids, num_rx * num_tx),
            np.tile(diagonal_ids, num_rx * num_tx),
        ]

        transformation = COO(coordinates, self.__state.flatten(), shape=(num_rx, num_tx, num_symbols, num_symbols))
        return transformation

    @staticmethod
    def Ideal(num_samples: int, num_receive_streams: int = 1, num_transmit_streams: int = 1) -> ChannelStateInformation:
        """Initialize an ideal channel state.

        Args:

            num_samples (int):
                Number of timestamps at which the channel state has been sampled.

            num_receive_streams (int, optional):
                Number of emerging data streams after channel propagation.

            num_transmit_streams (int, optional):
                Number of data streams feeding into the channel before propagation.

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

        for received_stream in self.__state:
            yield ChannelStateInformation(self.__state_format, received_stream[np.newaxis, ...])

    def samples(self) -> Generator[ChannelStateInformation, ChannelStateInformation, None]:
        """Iterate over the sample slices within this channel state.

        Returns:
            Generator: Generator.
        """

        for sample_idx in range(self.num_samples):
            yield ChannelStateInformation(self.__state_format, self.__state[:, :, [sample_idx], :], self.__num_delay_taps, self.__num_frequency_bins)

    def __getitem__(self, section: SupportsIndex | Tuple[SupportsIndex | slice, ...] | slice) -> ChannelStateInformation:
        """Slice the channel state information.

        Args:
            section (slice):
                Slice of the channel state.

        Returns:
            ChannelStateInformation:
                New channel state with a section according to `value` slice.
        """

        state_section = self.__state[section]

        for s, sec in enumerate(section):  # type: ignore
            if isinstance(sec, int):
                state_section = np.expand_dims(state_section, axis=s)

        num_delay_taps = self.__num_delay_taps if state_section.shape[3] == self.__state.shape[3] else None

        return ChannelStateInformation(self.__state_format, state_section, num_delay_taps)

    def __setitem__(self, key: SupportsIndex | slice | Tuple[SupportsIndex | slice, ...], value: ChannelStateInformation) -> None:
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
    def concatenate(elements: List[ChannelStateInformation], dimension: ChannelStateDimension) -> ChannelStateInformation:
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

    def propagate(self, signal: Signal) -> Signal:
        """Propagate a single signal model over this channel state information.

        This method should generally be avoided, since it's computationally costly.
        Instead, whenever there is access to a :class:`ChannelRealization`,
        :meth:`ChannelRealization.propagate` should always be preferred.

        Args:

            signal (Signal):
                Signal model to be propagated.

        Returns: Propagated signal model.
        """

        # Make sure the accessed state is in impulse response format
        state = self.to_impulse_response().dense_state()

        # Propagate the signal
        propagated_samples = np.zeros((state.shape[0], signal.num_samples + state.shape[3] - 1), dtype=np.complex_)

        for delay_index in range(state.shape[3]):
            for tx_idx, rx_idx in product(range(state.shape[1]), range(state.shape[0])):
                delayed_signal = state[rx_idx, tx_idx, : signal.num_samples, delay_index] * signal.samples[tx_idx, :]
                propagated_samples[rx_idx, delay_index : delay_index + signal.num_samples] += delayed_signal

        return Signal(propagated_samples, sampling_rate=signal.sampling_rate, carrier_frequency=signal.carrier_frequency, delay=signal.delay)

    def reciprocal(self) -> ChannelStateInformation:
        """Compute the reciprocal channel state.

        Returns: The reciprocal channel state information.
        """

        reciprocal_state = self.__state.transpose((1, 0, 2, 3))
        return ChannelStateInformation(self.__state_format, reciprocal_state, self.num_delay_taps, self.__num_frequency_bins)

    @classmethod
    def from_HDF(cls: Type[ChannelStateInformation], group: Group) -> ChannelStateInformation:
        # Recall datasets
        state = np.array(group["state"], dtype=complex)

        # Recall attributes
        format = ChannelStateFormat[group.attrs.get("format", "IMPULSE_RESPONSE")]

        # Initialize object from recalled state
        return cls(state=state, state_format=format)

    def to_HDF(self, group: Group) -> None:
        # Serialize datasets
        group.create_dataset("state", data=self.dense_state())

        # Serialize attributes
        group.attrs["num_transmit_streams"] = self.num_transmit_streams
        group.attrs["num_receive_streams"] = self.num_receive_streams
        group.attrs["num_symbols"] = self.num_symbols
        group.attrs["num_taps"] = self.num_delay_taps
        group.attrs["num_samples"] = self.num_samples
        group.attrs["format"] = self.state_format.name
