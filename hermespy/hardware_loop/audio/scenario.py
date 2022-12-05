# -*- coding: utf-8 -*-
"""
=====================
Audio Device Scenario
=====================
"""

from ..scenario import PhysicalScenario
from .device import AudioDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class AudioScenario(PhysicalScenario[AudioDevice]):
    """Scenario of phyical device bindings to sound cards."""

    ...  # pragma no cover
