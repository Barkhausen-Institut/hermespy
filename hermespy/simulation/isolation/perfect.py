# -*- coding: utf-8 -*-

from __future__ import annotations
from typing_extensions import override

from hermespy.core import Signal, SerializationProcess, DeserializationProcess
from .isolation import Isolation

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class PerfectIsolation(Isolation):
    """Perfect isolation model without leakage between RF chains."""

    @override
    def _leak(self, signal: Signal | None) -> Signal:
        if signal is None:
            return Signal.Empty(
                self.device.sampling_rate,
                self.device.antennas.num_receive_antennas,
                carrier_frequency=self.device.carrier_frequency,
            )

        else:
            return Signal.Empty(
                signal.sampling_rate,
                self.device.antennas.num_receive_antennas,
                carrier_frequency=signal.carrier_frequency,
            )

    @override
    def serialize(self, process: SerializationProcess) -> None:
        return

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> PerfectIsolation:
        return cls()
