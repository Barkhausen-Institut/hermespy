# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import TYPE_CHECKING
from abc import abstractmethod

from hermespy.core import Signal, SignalBlock, Serializable, SparseSignal

if TYPE_CHECKING:
    from ..simulated_device import SimulatedDeviceState  # pragma: no cover

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Coupling(Serializable):
    """Base class for mutual coupling model implementations."""

    def transmit(self, signal: Signal, state: SimulatedDeviceState) -> Signal:
        """Apply the mutual coupling model during signal transmission.

        Args:

            signal: The signal to be transmitted.
            state: The current state of the simulated device.

        Returns: The signal resulting from coupling modeling.

        Raises:

            ValueError: If the number of signal streams does not match the number of transmitting antennas.
        """

        if state.antennas.num_transmit_antennas != signal.num_streams:
            raise ValueError(
                f"Number of signal streams ({signal.num_streams}) does not match the number of transmitting antenna ports ({state.antennas.num_transmit_antennas})"
            )

        coupled_blocks: list[SignalBlock] = [self._transmit(b, state) for b in signal.blocks]
        return SparseSignal.Create(
            coupled_blocks,
            signal.sampling_rate,
            signal.carrier_frequency,
            signal.noise_power,
            signal.delay,
        )

    @abstractmethod
    def _transmit(self, signal: SignalBlock, state: SimulatedDeviceState) -> SignalBlock:
        """Apply the mutual coupling model during signal transmission.

        Args:

            signal: The signal to be transmitted.
            state: The current state of the simulated device.

        Returns: The signal resulting from coupling modeling.
        """
        ...  # pragma: no cover

    def receive(self, signal: Signal, state: SimulatedDeviceState) -> Signal:
        """Apply the mutual coupling model during signal reception.

        Args:
            signal: The signal to be received.
            state: The current state of the simulated device.

        Returns:
            The signal resulting from coupling modeling.

        Raises:
            ValueError: If the number of signal streams does not match the number of receive antennas.
        """

        if state.antennas.num_receive_antennas != signal.num_streams:
            raise ValueError(
                f"Number of signal streams ({signal.num_streams}) does not match the number of receive antennas ({state.antennas.num_receive_antennas})"
            )

        coupled_blocks: list[SignalBlock] = [self._receive(b, state) for b in signal.blocks]
        return SparseSignal.Create(
            coupled_blocks,
            signal.sampling_rate,
            signal.carrier_frequency,
            signal.noise_power,
            signal.delay,
        )

    @abstractmethod
    def _receive(self, signal: SignalBlock, state: SimulatedDeviceState) -> SignalBlock:
        """Apply the mutual coupling model during signal reception.

        Args:

            signal: The signal to be received.
            state: The current state of the simulated device.

        Returns: The signal resulting from coupling modeling.
        """
        ...  # pragma: no cover
