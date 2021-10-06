# -*- coding: utf-8 -*-
"""HermesPy hardware loop configuration.
"""

from __future__ import annotations

from . import Executable

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class HardwareLoop(Executable):
    """Hermespy hardware loop configuration.
    """

    yaml_tag = u'Loop'

    def __init__(self) -> None:
        """Object initialization.
        """
        pass

    def run(self) -> None:
        """Run the full loop configuration.
        """
        pass