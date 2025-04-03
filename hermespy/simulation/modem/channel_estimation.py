# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Generic

import numpy as np
from scipy.signal import convolve
from sparse import COO  # type: ignore

from hermespy.core import ChannelStateInformation, Serializable
from hermespy.channel import Channel, ChannelSample
from hermespy.modem import (
    ChannelEstimation,
    FilteredSingleCarrierWaveform,
    GridSection,
    OFDMWaveform,
    ReferencePosition,
    StatedSymbols,
    Symbols,
    CWT,
)
from ..simulated_device import SimulatedDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class IdealChannelEstimation(Generic[CWT], ChannelEstimation[CWT]):
    """Channel estimation accessing the ideal channel state informaion.

    This type of channel estimation is only available during simulation runtime.
    """

    __cached_sample: ChannelSample | None = None

    def __init__(
        self,
        channel: Channel,
        transmitter: SimulatedDevice,
        receiver: SimulatedDevice,
        waveform: CWT | None = None,
    ) -> None:
        # Initialize base class
        ChannelEstimation.__init__(self, waveform)

        # Register a hook to cache the channel sample
        channel.add_sample_hook(self.__channel_sample_callback, transmitter, receiver)

        # Store class attributes
        self.__channel = channel
        self.__transmitter = transmitter
        self.__receiver = receiver

    @property
    def channel(self) -> Channel:
        """Channel to estimate."""

        return self.__channel

    @property
    def transmitter(self) -> SimulatedDevice:
        """Transmitting device."""

        return self.__transmitter

    @property
    def receiver(self) -> SimulatedDevice:
        """Receiving device."""

        return self.__receiver

    def __channel_sample_callback(self, sample: ChannelSample) -> None:
        """Callback to cache the channel sample."""

        self.__cached_sample = sample

    def _csi(
        self, delay: float, sampling_rate: float | None = None, num_samples: int | None = None
    ) -> ChannelStateInformation:
        """Query the ideal channel state information.

        Args:

            delay:
                The considered frame's delay offset to the drop start in seconds.

            sampling_rate:
                Sampling rate of the generated CSI.
                If not provided, the waveform's sampling rate will be assumed.

            num_samples:
                Number of samples within the generated CSI.
                If not provided, the waveform's frame sample count will be assumed.

        Returns: Ideal channel state information of the most recent reception.

        Raises:

            RuntimeError: If the estimation routine is not attached.
            RuntimeError: If no channel state is available.
        """

        if self.waveform is None:
            raise RuntimeError("Ideal channel state estimation routine floating")

        if self.__cached_sample is None:
            raise RuntimeError(
                "No channel sample available from which to estimate the ideal channel state information"
            )

        sampling_rate = self.waveform.sampling_rate if sampling_rate is None else sampling_rate
        num_samples = self.waveform.samples_per_frame if num_samples is None else num_samples

        channel_state_information = self.__cached_sample.state(num_samples, num_samples)
        return channel_state_information


class SingleCarrierIdealChannelEstimation(
    IdealChannelEstimation[FilteredSingleCarrierWaveform], Serializable
):
    """Ideal channel estimation for single carrier waveforms"""

    def estimate_channel(self, symbols: Symbols, frame_delay: float = 0.0) -> StatedSymbols:
        oversampling_factor = self.waveform.oversampling_factor
        num_symbols = self.waveform._num_frame_symbols
        filter_delay = int(0.5 * self.waveform._filter_delay)
        # sync_delay = int(frame_delay * self.waveform.sampling_rate)

        # Compute the CSI including inter-symbol interference
        filter_characteristics = self.waveform._transmit_filter() * self.waveform._receive_filter()
        state = (
            self._csi(frame_delay, self.waveform.sampling_rate, self.waveform.samples_per_frame)
            .to_impulse_response()
            .dense_state()
        )
        filtered_state = convolve(
            state, filter_characteristics[None, None, :, None], "full", "direct"
        )

        summed_state = filtered_state[:, :, :, 0]
        for d in range(1, min(oversampling_factor, filtered_state.shape[3])):
            summed_state[:, :, d:] += filtered_state[:, :, :-d, d]

        # Extract the symbol CSI
        symbol_csi = summed_state[
            :,
            :,
            filter_delay : filter_delay + num_symbols * oversampling_factor : oversampling_factor,
            None,
        ]

        # Convert to sparse representation
        sparse_csi = COO.from_numpy(symbol_csi)
        return StatedSymbols(symbols.raw, sparse_csi)


class OFDMIdealChannelEstimation(IdealChannelEstimation[OFDMWaveform], Serializable):
    """Ideal channel state estimation for OFDM waveforms."""

    serialized_attributes = {"reference_position"}

    reference_position: ReferencePosition
    """Assumed position of the reference symbol within the frame."""

    def __init__(
        self,
        channel: Channel,
        transmitter: SimulatedDevice,
        receiver: SimulatedDevice,
        reference_position: ReferencePosition = ReferencePosition.IDEAL,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:

            transmitter:
                Device transmitting into the channel to be estimated.

            receiver:
                Device receiving from the channel to be estimated.

            reference_position:
                Assumed location of the reference symbols within the ofdm frame.
        """

        self.reference_position = reference_position
        IdealChannelEstimation.__init__(self, channel, transmitter, receiver, *args, **kwargs)  # type: ignore

    def __extract_channel(
        self, section: GridSection, csi: np.ndarray, reference_position: ReferencePosition
    ) -> np.ndarray:
        # Remove the cyclic prefixes before transformation into the symbol's domain
        _csi = section.pick_samples(csi.transpose((0, 1, 3, 2)))

        """if reference_position == ReferencePosition.IDEAL:
            selected_csi = np.mean(_csi, axis=3, keepdims=False)

        else:
            reference_index = 0

            if reference_position == ReferencePosition.IDEAL_MIDAMBLE:
                reference_index = _csi.shape[3] // 2

            elif reference_position == ReferencePosition.IDEAL_POSTAMBLE:
                reference_index = _csi.shape[3] - 1

            selected_csi = _csi[:, :, :, reference_index, :]"""

        # We assume the channel to be constant within a single symbol duration.
        # Therefore, we average the channel state over the symbol duration.
        average_symbol_csi = np.mean(_csi, axis=-1, keepdims=False)

        # The CSI is the Fourier transform of the channel state
        transformed_csi = self.waveform._backward_transformation(
            average_symbol_csi.transpose((0, 1, 3, 2)), normalize=False
        )

        # Transform the channel state into the frequency domain
        return transformed_csi

    def estimate_channel(self, symbols: Symbols, delay: float = 0.0) -> StatedSymbols:
        # Query and densify the channel state
        ideal_csi = self._csi(delay=delay).dense_state()[: symbols.num_streams, ...]

        symbol_csi = np.zeros(
            (symbols.num_streams, ideal_csi.shape[1], symbols.num_blocks, symbols.num_symbols),
            dtype=np.complex128,
        )

        sample_index = 0
        word_index = 0

        # If the frame contains a pilot section, skip the respective samples
        if self.waveform.pilot_section:
            sample_index += self.waveform.pilot_section.num_samples

        for section in self.waveform.grid_structure:
            num_samples = section.num_samples
            num_words = section.num_words

            if num_words > 0:
                csi = self.__extract_channel(
                    section,
                    ideal_csi[:, :, sample_index : sample_index + num_samples, :],
                    self.reference_position,
                )
                symbol_csi[:, :, word_index : word_index + section.num_words, :] = csi

            sample_index += num_samples
            word_index += num_words

        # Correct the FFT normalization
        # symbol_csi *= np.sqrt(self.waveform.num_subcarriers)

        return StatedSymbols(symbols.raw, symbol_csi)
