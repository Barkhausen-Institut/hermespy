# -*- coding: utf-8 -*-

try:  # pragma: no cover
    from .usrp import UsrpDevice
    from .system import UsrpSystem

except ImportError:  # pragma: no cover
    UsrpDevice, UsrpSystem = None, None  # type: ignore

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


__all__ = ["UsrpDevice", "UsrpSystem"]
