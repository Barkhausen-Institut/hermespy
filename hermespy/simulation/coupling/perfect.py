# -*- coding: utf-8 -*-
"""
================
Perfect Coupling
================
"""

from __future__ import annotations
from typing_extensions import override

from hermespy.core import Signal, SerializationProcess, DeserializationProcess
from .coupling import Coupling

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class PerfectCoupling(Coupling):
    """Ideal mutual coupling between two antenna arrays."""

    def _transmit(self, signal: Signal) -> Signal:
        return signal

    def _receive(self, signal: Signal) -> Signal:
        return signal

    @override
    def serialize(self, serialization_process: SerializationProcess) -> None:
        return

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> PerfectCoupling:
        return cls()
