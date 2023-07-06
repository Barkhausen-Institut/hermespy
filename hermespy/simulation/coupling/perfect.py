# -*- coding: utf-8 -*-
"""
================
Perfect Coupling
================
"""

from __future__ import annotations

from hermespy.core import Serializable, Signal
from .coupling import Coupling

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class PerfectCoupling(Serializable, Coupling):
    """Ideal mutual coupling between two antenna arrays."""

    yaml_tag = "Perfect-Coupling"

    def _transmit(self, signal: Signal) -> Signal:
        return signal

    def _receive(self, signal: Signal) -> Signal:
        return signal
