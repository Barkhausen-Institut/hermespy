from .channel_state_information import ChannelStateDimension, ChannelStateFormat, ChannelStateInformation
from .channel import Channel
from .multipath_fading_channel import MultipathFadingChannel
from .multipath_fading_templates import MultipathFadingCost256, MultipathFading5GTDL, MultipathFadingExponential

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.2"
__maintainer__ = "Tobias Kronauer"
__email__ = "tobias.kronaue@barkhauseninstitut.org"
__status__ = "Prototype"


try:
    from .quadriga_interface_matlab import QuadrigaMatlabInterface as QuadrigaInterface

except ImportError:
    try:

        from .quadriga_interface_octave import QuadrigaOctaveInterface as QuadrigaInterface

    except ImportError as error:
        from .quadriga_interface import QuadrigaInterface

from .quadriga_channel import QuadrigaChannel

__all__ = ['ChannelStateDimension', 'ChannelStateFormat', 'ChannelStateInformation', 'Channel',
           'MultipathFadingChannel', 'MultipathFading5GTDL', 'MultipathFadingCost256', 'MultipathFadingExponential',
           'QuadrigaChannel', 'QuadrigaInterface']
