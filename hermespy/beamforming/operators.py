# -*- coding: utf-8 -*-

from __future__ import annotations

from hermespy.core import (
    Reception,
    Serializable,
    Signal,
    SignalTransmitter,
    SignalReceiver,
    Transmission,
)
from .beamformer import TransmitBeamformer, ReceiveBeamformer

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
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
        self.__beamformer.operator = self

    def _transmit(self, duration: float = 0) -> Transmission:
        # Generate base transmission
        base_transmission = SignalTransmitter._transmit(self, duration)

        # Apply beamforming to the resulting port streams
        beamformed_signal = self.beamformer.transmit(base_transmission.signal)

        # Return transmission
        return Transmission(beamformed_signal)


class BeamformingReceiver(SignalReceiver, Serializable):
    """Signal receiver with digital beamforming capabilities."""

    yaml_tag = "BeamformingReceiver"
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
        self.__beamformer.operator = self

    def _receive(self, *args) -> Reception:
        # Receive base reception
        base_reception = SignalReceiver._receive(self, *args)

        # Apply beamforming to the resulting port streams
        beamformed_signal = self.beamformer.receive(base_reception.signal)

        # Return reception
        return Reception(beamformed_signal)
