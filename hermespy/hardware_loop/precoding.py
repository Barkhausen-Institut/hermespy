# -*- coding: utf-8 -*-

from __future__ import annotations
from typing_extensions import override

import numpy as np

from hermespy.core import (
    DeserializationProcess,
    DenseSignal,
    ReceiveState,
    ReceiveStreamDecoder,
    SerializationProcess,
    Signal,
    TransmitState,
    TransmitStreamEncoder,
)

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Yash Richhariya"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class IQSplitter(TransmitStreamEncoder):
    """Splits a complex base-band signal into its complex-valued and real-valued components."""

    def encode_streams(
        self, streams: Signal, num_output_streams: int, device: TransmitState
    ) -> Signal:
        samples = streams.view(np.ndarray)

        split_samples = np.empty((num_output_streams, streams.num_samples), dtype=np.complex128)
        split_samples[::2, :] = samples.real
        split_samples[1::2, :] = samples.imag

        return DenseSignal(
            num_output_streams,
            streams.num_samples,
            streams.sampling_rate,
            streams.carrier_frequency,
            streams.noise_power,
            streams.delay,
            split_samples.tobytes(),
        )

    @override
    def num_transmit_input_streams(self, num_output_streams: int) -> int:
        return num_output_streams // 2

    @override
    def serialize(self, process: SerializationProcess) -> None:
        return

    @classmethod
    @override
    def Deserialize(cls: type[IQSplitter], process: DeserializationProcess) -> IQSplitter:
        return cls()


class IQCombiner(ReceiveStreamDecoder):
    """Combines a complex-valued and a real-valued signal into a complex base-band signal."""

    def combineIQ(self, signalI: np.ndarray, signalQ: np.ndarray) -> np.ndarray:
        phi = np.mean(np.angle(signalQ / signalI))

        signalQ = np.exp(-1j * phi) * signalQ
        amp = np.sqrt(np.abs(signalI) ** 2 + np.abs(signalQ) ** 2)
        phase = np.angle(signalQ + signalI)
        combinedSamples = amp * np.exp(1j * phase)
        return combinedSamples

    @override
    def decode_streams(
        self, streams: Signal, num_output_streams: int, device: ReceiveState
    ) -> Signal:
        stream_samples = streams.view(np.ndarray)
        combined_samples = self.combineIQ(stream_samples[0::2], stream_samples[1::2])
        return DenseSignal(
            num_output_streams,
            streams.num_samples,
            streams.sampling_rate,
            streams.carrier_frequency,
            streams.noise_power,
            streams.delay,
            combined_samples.tobytes(),
        )

    @override
    def num_receive_output_streams(self, num_input_streams: int) -> int:
        return num_input_streams // 2

    @override
    def serialize(self, process: SerializationProcess) -> None:
        return

    @classmethod
    @override
    def Deserialize(cls: type[IQCombiner], process: DeserializationProcess) -> IQCombiner:
        return cls()
