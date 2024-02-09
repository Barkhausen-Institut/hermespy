# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Generic

import numpy as np
from scipy.signal import convolve
from sparse import COO  # type: ignore

from hermespy.core import ChannelStateInformation, Serializable
from hermespy.modem import (
    ChannelEstimation,
    FilteredSingleCarrierWaveform,
    OFDMWaveform,
    ReferencePosition,
    StatedSymbols,
    Symbols,
    WaveformType,
)
from ..simulated_device import SimulatedDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class IdealChannelEstimation(Generic[WaveformType], ChannelEstimation[WaveformType]):
    """Channel estimation accessing the ideal channel state informaion.

    This type of channel estimation is only available during simulation runtime.
    """

    yaml_tag = "IdealChannelEstimation"

    __transmitter: SimulatedDevice
    __receiver: SimulatedDevice

    def __init__(
        self,
        transmitter: SimulatedDevice,
        receiver: SimulatedDevice,
        waveform: WaveformType | None = None,
    ) -> None:
        # Initialize base class
        ChannelEstimation.__init__(self, waveform)

        # Initialize class attributes
        self.__transmitter = transmitter
        self.__receiver = receiver

    @property
    def transmitter(self) -> SimulatedDevice:
        """Device transmitting into the channel to be estimated."""

        return self.__transmitter

    @property
    def receiver(self) -> SimulatedDevice:
        """Device receiving from the channel to be estimated."""

        return self.__receiver

    def _csi(
        self, delay: float, sampling_rate: float | None = None, num_samples: int | None = None
    ) -> ChannelStateInformation:
        """Query the ideal channel state information.

        Args:

            delay (float):
                The considered frame's delay offset to the drop start in seconds.

            sampling_rate (float, optional):
                Sampling rate of the generated CSI.
                If not provided, the waveform's sampling rate will be assumed.

            num_samples (int, optional):
                Number of samples within the generated CSI.
                If not provided, the waveform's frame sample count will be assumed.

        Returns: Ideal channel state information of the most recent reception.

        Raises:

            RuntimeError: If the estimation routine is not attached.
            RuntimeError: If no channel state is available.
        """

        if self.waveform is None:
            raise RuntimeError("Ideal channel state estimation routine floating")

        if self.waveform.modem is None or self.waveform.modem.receiving_device is None:
            raise RuntimeError("Operating modem floating")

        cached_realization = self.receiver.channel_realization(self.transmitter)
        if cached_realization is None:
            raise RuntimeError(
                "No channel realization available from which to estimate the ideal channel state information"
            )

        sampling_rate = self.waveform.sampling_rate if sampling_rate is None else sampling_rate
        num_samples = self.waveform.samples_per_frame if num_samples is None else num_samples

        channel_state_information = cached_realization.state(
            delay, sampling_rate, num_samples, num_samples
        )
        return channel_state_information


class SingleCarrierIdealChannelEstimation(
    IdealChannelEstimation[FilteredSingleCarrierWaveform], Serializable
):
    """Ideal channel estimation for single carrier waveforms"""

    yaml_tag = "SC-Ideal"

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

    yaml_tag = "OFDM-Ideal"
    serialized_attributes = {"reference_position"}

    reference_position: ReferencePosition
    """Assumed position of the reference symbol within the frame."""

    def __init__(
        self,
        transmitter: SimulatedDevice,
        receiver: SimulatedDevice,
        reference_position: ReferencePosition = ReferencePosition.IDEAL,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:

            transmitter (SimulatedDevice):
                Device transmitting into the channel to be estimated.

            receiver (SimulatedDevice):
                Device receiving from the channel to be estimated.

            reference_position (ReferencPosition, optional):
                Assumed location of the reference symbols within the ofdm frame.
        """

        self.reference_position = reference_position
        IdealChannelEstimation.__init__(self, transmitter, receiver, *args, **kwargs)  # type: ignore

    def estimate_channel(self, symbols: Symbols, delay: float = 0.0) -> StatedSymbols:
        # Query and densify the channel state
        ideal_csi = self._csi(delay=delay).dense_state()[: symbols.num_streams, ::]

        symbol_csi = np.zeros(
            (symbols.num_streams, ideal_csi.shape[1], symbols.num_blocks, symbols.num_symbols),
            dtype=np.complex_,
        )

        sample_index = 0
        word_index = 0

        # If the frame contains a pilot section, skip the respective samples
        if self.waveform.pilot_section:
            sample_index += self.waveform.pilot_section.num_samples

        for section in self.waveform.structure:
            num_samples = section.num_samples
            csi = section.extract_channel(
                ideal_csi[:, :, sample_index : sample_index + num_samples, :],
                self.reference_position,
            )
            symbol_csi[:, :, word_index : word_index + section.num_words, :] = csi

            sample_index += num_samples
            word_index += section.num_words

        # Corret the FFT normalization
        symbol_csi *= np.sqrt(self.waveform.num_subcarriers)

        return StatedSymbols(symbols.raw, symbol_csi)
