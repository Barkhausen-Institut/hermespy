# -*- coding: utf-8 -*-

from .correlation import DeviceType, CorrelationType, StandardAntennaCorrelation
from .cost259 import Cost259Type, Cost259
from .exponential import Exponential
from .fading import (
    MultipathFadingChannel,
    MultipathFadingRealization,
    MultipathFadingSample,
    AntennaCorrelation,
    CustomAntennaCorrelation,
)
from .tdl import TDLType, TDL

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


__all__ = [
    "DeviceType",
    "CorrelationType",
    "StandardAntennaCorrelation",
    "Cost259Type",
    "Cost259",
    "Exponential",
    "MultipathFadingChannel",
    "MultipathFadingRealization",
    "MultipathFadingSample",
    "AntennaCorrelation",
    "CustomAntennaCorrelation",
    "TDLType",
    "TDL",
]
