# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Sequence
from typing_extensions import override

import numpy as np

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
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
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
    def _transmit(self, state: TransmitState, duration: float) -> Transmission:
        # Generate base transmission
        base_transmission = SignalTransmitter._transmit(self, state, duration)

        # Apply beamforming to the resulting port streams
        beamformed_signal = self.beamformer.encode_streams(
            base_transmission.signal, state.antennas.num_transmit_antennas, state
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
        selected_receive_ports: Sequence[int] | None = None,
        expected_power: float = 0.0,
    ) -> None:
        """
        Args:

            beamformer: Beamformer to be used.
            num_samples: Number of samples to be received.
            selected_receive_ports: Ports to be used for reception. If None, all ports are used.
            expected_power: Expected power of the received signal.
        """

        # Initialize base class
        SignalReceiver.__init__(self, num_samples, selected_receive_ports, expected_power)

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
    def _receive(self, signal: Signal, state: ReceiveState) -> Reception:
        # Receive base reception
        base_reception = SignalReceiver._receive(self, signal, state)

        # Apply beamforming to the resulting port streams
        beamformed_signal = self.beamformer.decode_streams(
            base_reception.signal,
            self.beamformer.num_receive_output_streams(signal.num_streams),
            state,
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
        selected_receive_ports = process.deserialize_array("selected_receive_ports", np.int64, None)
        return BeamformingReceiver(
            process.deserialize_object("beamformer"),
            process.deserialize_integer("num_samples"),
            (
                selected_receive_ports.flatten().tolist()
                if selected_receive_ports is not None
                else None
            ),
            process.deserialize_floating("expected_power"),
        )
