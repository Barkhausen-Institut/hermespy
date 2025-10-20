# -*- coding: utf-8 -*-
"""
================
Perfect Coupling
================
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing_extensions import override

from hermespy.core import SignalBlock, SerializationProcess, DeserializationProcess
from .coupling import Coupling

if TYPE_CHECKING:
    from hermespy.simulation import SimulatedDeviceState  # pragma: no cover

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class PerfectCoupling(Coupling):
    """Ideal mutual coupling between two antenna arrays."""

    @override
    def _transmit(self, signal: SignalBlock, state: SimulatedDeviceState) -> SignalBlock:
        return signal  # Just a pass-through

    @override
    def _receive(self, signal: SignalBlock, state: SimulatedDeviceState) -> SignalBlock:
        return signal  # Just a pass-through

    @override
    def serialize(self, process: SerializationProcess) -> None:
        return

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> PerfectCoupling:
        return cls()
