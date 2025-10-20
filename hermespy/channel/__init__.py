# -*- coding: utf-8 -*-

from .channel import Channel, ChannelRealization, ChannelSample, ChannelSampleHook, CRT, LinkState
from .consistent import (
    ConsistentGenerator,
    ConsistentBoolean,
    ConsistentGaussian,
    ConsistentUniform,
    DualConsistentRealization,
    StaticConsistentRealization,
)
from .cdl import (
    ClusterDelayLineRealizationParameters,
    DelayNormalization,
    LOSState,
    O2IState,
    CDL,
    CDLRealization,
    CDLType,
    IndoorFactory,
    IndoorFactoryRealization,
    FactoryType,
    IndoorOffice,
    IndoorOfficeRealization,
    OfficeType,
    RuralMacrocells,
    RuralMacrocellsRealization,
    UrbanMacrocells,
    UrbanMacrocellsRealization,
    UrbanMicrocells,
    UrbanMicrocellsRealization,
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
    MultiTargetRadarChannelRealization,
    FixedCrossSection,
    VirtualRadarTarget,
    PhysicalRadarTarget,
    SingleTargetRadarChannel,
    SingleTargetRadarChannelRealization,
)

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


__all__ = [
    "Channel",
    "ChannelRealization",
    "ChannelRealization",
    "ChannelSample",
    "ChannelSampleHook",
    "CRT",
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
    "DualConsistentRealization",
    "StaticConsistentRealization",
    "ClusterDelayLineRealizationParameters",
    "DelayNormalization",
    "LOSState",
    "O2IState",
    "CDL",
    "CDLRealization",
    "CDLType",
    "IndoorFactory",
    "IndoorFactoryRealization",
    "FactoryType",
    "IndoorOffice",
    "IndoorOfficeRealization",
    "OfficeType",
    "RuralMacrocells",
    "RuralMacrocellsRealization",
    "UrbanMacrocells",
    "UrbanMacrocellsRealization",
    "UrbanMicrocells",
    "UrbanMicrocellsRealization",
    "IdealChannel",
    "IdealChannelRealization",
    "QuadrigaChannel",
    "FixedCrossSection",
    "MultiTargetRadarChannel",
    "MultiTargetRadarChannelRealization",
    "VirtualRadarTarget",
    "PhysicalRadarTarget",
    "SingleTargetRadarChannel",
    "SingleTargetRadarChannelRealization",
]
