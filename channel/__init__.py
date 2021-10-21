from .channel import Channel
from .multipath_fading_channel import MultipathFadingChannel
from .multipath_fading_templates import MultipathFadingCost256, MultipathFading5GTDL

try:
    from .quadriga_interface_matlab import QuadrigaMatlabInterface as QuadrigaInterface

except ImportError:
    try:

        from .quadriga_interface_octave import QuadrigaOctaveInterface as QuadrigaInterface

    except ImportError as error:
        from .quadriga_interface import QuadrigaInterface

from .quadriga_channel import QuadrigaChannel

__all__ = ['Channel', 'MultipathFadingChannel', 'MultipathFading5GTDL', 'MultipathFadingCost256',
           'QuadrigaChannel', 'QuadrigaInterface']
