# -*- coding: utf-8 -*-

from __future__ import annotations
from typing_extensions import override

from hermespy.core import (
    DeserializationProcess,
    ReceiveState,
    Reception,
    Serializable,
    SerializationProcess,
    Signal,
    SignalTransmitter,
    SignalReceiver,
    Transmission,
    TransmitState,
)
from .beamformer import TransmitBeamformer, ReceiveBeamformer

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class BeamformingTransmitter(SignalTransmitter):
    """Signal transmitter with digital beamforming capabilities."""

    __beamformer: TransmitBeamformer

    def __init__(self, signal: Signal, beamformer: TransmitBeamformer) -> None:
        """
        Args:

            signal (Signal):
                Signal to be transmitted.

            beamformer (TransmitBeamformer):
                Beamformer to be used.
        """

        # Initialize base class
        SignalTransmitter.__init__(self, signal)

        # Initialize class attributes
        self.beamformer = beamformer

    @property
    def beamformer(self) -> TransmitBeamformer:
        """Beamformer to be used.

        Returns: Beamformer.
        """

        return self.__beamformer

    @beamformer.setter
    def beamformer(self, value: TransmitBeamformer) -> None:
        self.__beamformer = value

    @override
    def _transmit(self, device: TransmitState, duration: float) -> Transmission:
        # Generate base transmission
        base_transmission = SignalTransmitter._transmit(self, device, duration)

        # Apply beamforming to the resulting port streams
        beamformed_signal = self.beamformer.encode_streams(
            base_transmission.signal, device.antennas.num_transmit_antennas, device
        )

        # Return transmission
        return Transmission(beamformed_signal)

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_object(self.beamformer, "beamformer")
        process.serialize_object(self.signal, "signal")

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> BeamformingTransmitter:
        return BeamformingTransmitter(
            process.deserialize_object("signal"), process.deserialize_object("beamformer")
        )


class BeamformingReceiver(SignalReceiver, Serializable):
    """Signal receiver with digital beamforming capabilities."""

    __beamformer: ReceiveBeamformer

    def __init__(
        self,
        beamformer: ReceiveBeamformer,
        num_samples: int,
        sampling_rate: float,
        expected_power: float = 0.0,
    ) -> None:
        """
        Args:

            beamformer (ReceiveBeamformer):
                Beamformer to be used.

            num_samples (int):
                Number of samples per reception.

            sampling_rate (float):
                Required sampling rate in Hz.

            expected_power (float):
                Expected power of the received signal in W.
        """

        # Initialize base class
        SignalReceiver.__init__(self, num_samples, sampling_rate, expected_power)

        # Initialize class attributes
        self.beamformer = beamformer

    @property
    def beamformer(self) -> ReceiveBeamformer:
        """Beamformer to be used.

        Returns: Beamformer.
        """

        return self.__beamformer

    @beamformer.setter
    def beamformer(self, value: ReceiveBeamformer) -> None:
        self.__beamformer = value

    @override
    def _receive(self, signal: Signal, device: ReceiveState) -> Reception:
        # Receive base reception
        base_reception = SignalReceiver._receive(self, signal, device)

        # Apply beamforming to the resulting port streams
        beamformed_signal = self.beamformer.decode_streams(
            base_reception.signal,
            self.beamformer.num_receive_output_streams(signal.num_streams),
            device,
        )

        # Return reception
        return Reception(beamformed_signal)

    @override
    def serialize(self, process: SerializationProcess) -> None:
        SignalReceiver.serialize(self, process)
        process.serialize_object(self.beamformer, "beamformer")

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> BeamformingReceiver:
        return BeamformingReceiver(
            process.deserialize_object("beamformer"),
            process.deserialize_integer("num_samples"),
            process.deserialize_floating("sampling_rate"),
            process.deserialize_floating("expected_power"),
        )
