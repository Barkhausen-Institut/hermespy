# -*- coding: utf-8 -*-

from .channel import (
    Channel,
    ChannelRealization,
    ChannelSample,
    ChannelSampleHook,
    CRT,
    InterpolationMode,
    LinkState,
)
from .consistent import (
    ConsistentGenerator,
    ConsistentBoolean,
    ConsistentGaussian,
    ConsistentUniform,
)
from .cdl import (
    DelayNormalization,
    LOSState,
    O2IState,
    CDL,
    CDLType,
    IndoorFactory,
    FactoryType,
    IndoorOffice,
    OfficeType,
    RuralMacrocells,
    UrbanMacrocells,
    UrbanMicrocells,
)
from .delay import SpatialDelayChannel, RandomDelayChannel
from .fading import (
    DeviceType,
    CorrelationType,
    StandardAntennaCorrelation,
    Cost259Type,
    Cost259,
    Exponential,
    MultipathFadingChannel,
    MultipathFadingRealization,
    MultipathFadingSample,
    AntennaCorrelation,
    CustomAntennaCorrelation,
    TDLType,
    TDL,
)
from .quadriga import QuadrigaChannel
from .ideal import IdealChannel, IdealChannelRealization
from .radar import (
    MultiTargetRadarChannel,
    FixedCrossSection,
    VirtualRadarTarget,
    PhysicalRadarTarget,
    SingleTargetRadarChannel,
)

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


__all__ = [
    "ChannelStateDimension",
    "ChannelStateFormat",
    "Channel",
    "ChannelRealization",
    "ChannelRealization",
    "ChannelSample",
    "ChannelSampleHook",
    "CRT",
    "InterpolationMode",
    "LinkState",
    "SpatialDelayChannel",
    "RandomDelayChannel",
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
    "ConsistentGenerator",
    "ConsistentBoolean",
    "ConsistentGaussian",
    "ConsistentUniform",
    "DelayNormalization",
    "LOSState",
    "O2IState",
    "CDL",
    "CDLType",
    "IndoorFactory",
    "FactoryType",
    "IndoorOffice",
    "OfficeType",
    "RuralMacrocells",
    "UrbanMacrocells",
    "UrbanMicrocells",
    "IdealChannel",
    "IdealChannelRealization",
    "QuadrigaChannel",
    "FixedCrossSection",
    "MultiTargetRadarChannel",
    "VirtualRadarTarget",
    "PhysicalRadarTarget",
    "SingleTargetRadarChannel",
]
