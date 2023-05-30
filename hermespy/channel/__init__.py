# -*- coding: utf-8 -*-

from .channel import Channel, ChannelRealization, CRT
from .cluster_delay_line_indoor_factory import IndoorFactoryLineOfSight, IndoorFactoryNoLineOfSight
from .cluster_delay_line_indoor_office import IndoorOfficeLineOfSight, IndoorOfficeNoLineOfSight
from .cluster_delay_line_rural_macrocells import RuralMacrocellsLineOfSight, RuralMacrocellsNoLineOfSight, RuralMacrocellsOutsideToInside
from .cluster_delay_line_street_canyon import StreetCanyonLineOfSight, StreetCanyonNoLineOfSight, StreetCanyonOutsideToInside
from .cluster_delay_line_urban_macrocells import UrbanMacrocellsLineOfSight, UrbanMacrocellsNoLineOfSight, UrbanMacrocellsOutsideToInside
from .cluster_delay_lines import ClusterDelayLine, DelayNormalization
from .delay import DelayChannelBase, SpatialDelayChannel, RandomDelayChannel
from .ideal import IdealChannel, IdealChannelRealization
from .multipath_fading_channel import MultipathFadingChannel, AntennaCorrelation, CustomAntennaCorrelation
from .multipath_fading_templates import MultipathFadingCost256, Cost256Type, MultipathFading5GTDL, TDLType, MultipathFadingExponential, StandardAntennaCorrelation, DeviceType, CorrelationType
from .radar_channel import RadarChannelBase, SingleTargetRadarChannel, RadarTarget, RadarCrossSectionModel, FixedCrossSection, MultiTargetRadarChannel, VirtualRadarTarget, PhysicalRadarTarget, MultiTargetRadarChannelRealization

from .quadriga_interface_matlab import MatlabEngine
from .quadriga_interface_octave import Oct2Py

if MatlabEngine is not None:  # pragma: no cover
    from .quadriga_interface_matlab import QuadrigaMatlabInterface as QuadrigaInterface  # type: ignore
elif Oct2Py is not None:  # pragma: no cover
    from .quadriga_interface_octave import QuadrigaOctaveInterface as QuadrigaInterface  # type: ignore
else:  # pragma: no cover
    from .quadriga_interface import QuadrigaInterface  # type: ignore

from .quadriga_channel import QuadrigaChannel

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


__all__ = [
    "ChannelStateDimension",
    "ChannelStateFormat",
    "Channel",
    "ChannelRealization",
    "CRT",
    "DelayChannelBase",
    "SpatialDelayChannel",
    "RandomDelayChannel",
    "IndoorFactoryLineOfSight",
    "IndoorFactoryNoLineOfSight",
    "IndoorOfficeLineOfSight",
    "IndoorOfficeNoLineOfSight",
    "RuralMacrocellsLineOfSight",
    "RuralMacrocellsNoLineOfSight",
    "RuralMacrocellsOutsideToInside",
    "StreetCanyonLineOfSight",
    "StreetCanyonNoLineOfSight",
    "StreetCanyonOutsideToInside",
    "UrbanMacrocellsLineOfSight",
    "UrbanMacrocellsNoLineOfSight",
    "UrbanMacrocellsOutsideToInside",
    "ClusterDelayLine",
    "DelayNormalization",
    "IdealChannel",
    "IdealChannelRealization",
    "MultipathFadingChannel",
    "AntennaCorrelation",
    "CustomAntennaCorrelation",
    "MultipathFading5GTDL",
    "TDLType",
    "MultipathFadingCost256",
    "Cost256Type",
    "MultipathFadingExponential",
    "StandardAntennaCorrelation",
    "DeviceType",
    "CorrelationType",
    "QuadrigaChannel",
    "QuadrigaInterface",
    "RadarChannelBase",
    "SingleTargetRadarChannel",
    "RadarTarget",
    "RadarCrossSectionModel",
    "FixedCrossSection",
    "MultiTargetRadarChannel",
    "VirtualRadarTarget",
    "PhysicalRadarTarget",
    "MultiTargetRadarChannelRealization",
]
