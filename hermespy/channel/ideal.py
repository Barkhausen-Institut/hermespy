# -*- coding: utf-8 -*-

from __future__ import annotations
from typing_extensions import override

import numpy as np

from .channel import Channel, ChannelRealization, ChannelSample, LinkState, InterpolationMode
from hermespy.core import (
    ChannelStateInformation,
    ChannelStateFormat,
    SignalBlock,
    DeserializationProcess,
    SerializationProcess,
)

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class IdealChannelSample(ChannelSample):
    """Sample of an ideal channel realization.

    Generated by the :meth:`_sample<IdealChannelRealization._sample>` routine of :class:`IdealChannelRealization`.
    """

    def __init__(self, gain: float, state: LinkState) -> None:
        """
        Args:
            gain: Linear channel power factor.
            state: State of the channel at the time of sampling.
        """

        # Initialize base class
        ChannelSample.__init__(self, state)

        # Initialize class attributes
        self.__gain = gain

    @property
    def expected_energy_scale(self) -> float:
        return self.__gain

    def state(
        self,
        num_samples: int,
        max_num_taps: int,
        interpolation_mode: InterpolationMode = InterpolationMode.NEAREST,
    ) -> ChannelStateInformation:

        # MISO case
        if self.num_receive_antennas == 1:
            spatial_response = np.ones(
                (1, self.transmitter_state.antennas.num_transmit_antennas), dtype=np.complex128
            )

        # SIMO case
        elif self.num_transmit_antennas == 1:
            spatial_response = np.ones(
                (self.receiver_state.antennas.num_receive_antennas, 1), dtype=np.complex128
            )

        # MIMO case
        else:
            spatial_response = np.eye(
                self.num_receive_antennas, self.num_transmit_antennas, dtype=np.complex128
            )

        # Scale response by channel gain
        spatial_response *= np.sqrt(self.__gain)

        sampled_state = np.expand_dims(
            np.repeat(spatial_response[:, :, np.newaxis], num_samples, 2), axis=3
        )
        return ChannelStateInformation(ChannelStateFormat.IMPULSE_RESPONSE, sampled_state)

    def _propagate(self, signal: SignalBlock, interpolation: InterpolationMode) -> SignalBlock:
        # Single antenna transmitter case
        if self.num_transmit_antennas == 1:
            propagated_samples = np.repeat(signal[[0], :], self.num_receive_antennas, axis=0)

        # Single antenna receiver case
        elif self.num_receive_antennas == 1:
            propagated_samples = np.sum(signal, axis=0, keepdims=True)

        # No receiving antenna case
        elif self.num_receive_antennas == 0:
            propagated_samples = np.empty((0, signal.num_samples), dtype=np.complex128)

        # MIMO case
        else:
            propagated_samples = signal[: self.num_receive_antennas]
            if self.num_transmit_antennas < self.num_receive_antennas:
                propagated_samples = np.append(
                    propagated_samples,
                    np.zeros(
                        (
                            self.num_receive_antennas - self.num_transmit_antennas,
                            signal.num_samples,
                        ),
                        dtype=np.complex128,
                    ),
                    axis=0,
                )

        # Apply channel gain
        propagated_samples *= np.sqrt(self.__gain)
        return SignalBlock(propagated_samples, signal._offset)


class IdealChannelRealization(ChannelRealization[IdealChannelSample]):
    """Realization of an ideal channel.

    Generated by the :meth:`_realize()<IdealChannel._realize>` routine of :class:`IdealChannels<IdealChannel>`.
    """

    @override
    def _sample(self, state: LinkState) -> IdealChannelSample:
        # Since the ideal channel is deterministic, this is just a pass-through
        return IdealChannelSample(self.gain, state)

    @override
    def _reciprocal_sample(
        self, sample: IdealChannelSample, state: LinkState
    ) -> IdealChannelSample:
        return IdealChannelSample(self.gain, state)

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_floating(self.gain, "gain")

    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> IdealChannelRealization:
        return IdealChannelRealization(None, process.deserialize_floating("gain"))


class IdealChannel(Channel[IdealChannelRealization, IdealChannelSample]):
    """An ideal distortion-less channel model."""

    @override
    def _realize(self) -> IdealChannelRealization:
        return IdealChannelRealization(self.sample_hooks, self.gain)

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_floating(self.gain, "gain")

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> IdealChannel:
        return IdealChannel(process.deserialize_floating("gain"))
