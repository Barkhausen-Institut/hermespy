# -*- coding: utf-8 -*-

try:  # pragma: no cover
    from .usrp import UsrpAntennas, UsrpDevice
    from .system import UsrpSystem

except ImportError:  # pragma: no cover
    UsrpaAntennas, UsrpDevice, UsrpSystem = None, None, None  # type: ignore

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


__all__ = ["UsrpAntennas", "UsrpDevice", "UsrpSystem"]
