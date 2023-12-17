# -*- coding: utf-8 -*-
"""
===================
General Definitions
===================
"""

from .factory import SerializableEnum

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class ConsoleMode(SerializableEnum):
    """Printing behaviour of the simulation during runtime"""

    INTERACTIVE = 0
    """Interactive refreshing of the shell information"""

    LINEAR = 1
    """Linear appending of the shell information"""

    SILENT = 2
    """No prints exept errors"""


class SNRType(SerializableEnum):
    """Supported types of signal-to-noise ratios."""

    EBN0 = 0
    """Bit energy to noise power ratio."""

    ESN0 = 1
    """Symbol energy to noise power ratio."""

    PN0 = 2
    """Signal power to noise power ratio."""

    EN0 = 3
    """Signal energy to noise power ratio."""

    N0 = 4
    """Noise power."""

    CUSTOM = 3
    """Custom snr definition."""
