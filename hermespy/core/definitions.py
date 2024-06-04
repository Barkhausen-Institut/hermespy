# -*- coding: utf-8 -*-
"""
===================
General Definitions
===================
"""

from .factory import SerializableEnum

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
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
