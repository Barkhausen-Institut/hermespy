from hermespy.core.channel_state_information import ChannelStateDimension, ChannelStateFormat, ChannelStateInformation
from .channel import Channel
from .cluster_delay_line_indoor_factory import IndoorFactoryLineOfSight, IndoorFactoryNoLineOfSight
from .cluster_delay_line_indoor_office import IndoorOfficeLineOfSight, IndoorOfficeNoLineOfSight
from .cluster_delay_line_rural_macrocells import RuralMacrocellsLineOfSight, RuralMacrocellsNoLineOfSight, RuralMacrocellsOutsideToInside
from .cluster_delay_line_street_canyon import StreetCanyonLineOfSight, StreetCanyonNoLineOfSight, StreetCanyonOutsideToInside
from .cluster_delay_line_urban_macrocells import UrbanMacrocellsLineOfSight, UrbanMacrocellsNoLineOfSight, UrbanMacrocellsOutsideToInside
from .cluster_delay_lines import ClusterDelayLine, DelayNormalization
from .multipath_fading_channel import MultipathFadingChannel, AntennaCorrelation, CustomAntennaCorrelation
from .multipath_fading_templates import MultipathFadingCost256, Cost256Type, MultipathFading5GTDL, TDLType, MultipathFadingExponential, StandardAntennaCorrelation, DeviceType, CorrelationType
from .radar_channel import RadarChannel

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


try:
    from .quadriga_interface_matlab import QuadrigaMatlabInterface as QuadrigaInterface

except ImportError:
    try:

        from .quadriga_interface_octave import QuadrigaOctaveInterface as QuadrigaInterface

    except ImportError:
        from .quadriga_interface import QuadrigaInterface

from .quadriga_channel import QuadrigaChannel

__all__ = [
    "ChannelStateDimension",
    "ChannelStateFormat",
    "ChannelStateInformation",
    "Channel",
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
    "RadarChannel",
]
