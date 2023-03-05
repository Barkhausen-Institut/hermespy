# -*- coding: utf-8 -*-
"""
=================
Perfect Isolation
=================
"""

from __future__ import annotations

from hermespy.core import FloatingError, Serializable, Signal
from .isolation import Isolation

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class PerfectIsolation(Serializable, Isolation):
    """Perfect isolation model without leakage between RF chains."""

    yaml_tag = "PerfectIsolation"

    def leak(self, signal: Signal | None) -> Signal:
        if self.device is None:
            raise FloatingError("Error trying to simulate leakage of a floating model")

        if signal is None:
            return self._leak(None)

        if self.device.antennas.num_transmit_antennas != signal.num_streams:
            raise ValueError("Number of signal streams ({signal.num_streams}) does not match the number of transmitting antennas ({self.device.antennas.num_transmit_antennas})")

        return self._leak(signal)

    def _leak(self, signal: Signal | None) -> Signal:
        if signal is None:
            return Signal.empty(self.device.sampling_rate, self.device.antennas.num_receive_antennas, carrier_frequency=self.device.carrier_frequency)

        else:
            return Signal.empty(signal.sampling_rate, self.device.antennas.num_receive_antennas, carrier_frequency=signal.carrier_frequency)
