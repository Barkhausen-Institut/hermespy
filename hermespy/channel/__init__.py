from .channel import Channel, ChannelRealization, ChannelRealizationType
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

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


try:
    from .quadriga_interface_matlab import QuadrigaMatlabInterface as QuadrigaInterface

except ImportError:
    try:
        from .quadriga_interface_octave import QuadrigaOctaveInterface as QuadrigaInterface  # type: ignore

    except ImportError:
        from .quadriga_interface import QuadrigaInterface  # type: ignore

from .quadriga_channel import QuadrigaChannel

__all__ = [
    "ChannelStateDimension",
    "ChannelStateFormat",
    "Channel",
    "ChannelRealization",
    "ChannelRealizationType",
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
