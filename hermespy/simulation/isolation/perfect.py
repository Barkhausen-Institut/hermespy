# -*- coding: utf-8 -*-
"""
=================
Perfect Isolation
=================
"""

from hermespy.core import Serializable, Signal
from .isolation import Isolation

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class PerfectIsolation(Serializable, Isolation):
    """Perfect isolation model without leakage between RF chains."""

    yaml_tag = u'Perfect'

    def _leak(self, signal: Signal) -> Signal:

        # No leakage at all, therefore an empty signal is sufficient
        return Signal.empty(signal.sampling_rate,
                            self.device.antennas.num_receive_antennas,
                            carrier_frequency=signal.carrier_frequency)
