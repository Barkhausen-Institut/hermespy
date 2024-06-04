from os import path
from typing import List, Type

import matplotlib.pyplot as plt

from hermespy.simulation.antenna import Antenna, IdealAntenna, PatchAntenna, Dipole

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.7"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


image_dir = path.join(path.dirname(__file__), '..', 'images')
antenna_types: List[Type[Antenna]] = [IdealAntenna, PatchAntenna, Dipole]

for antenna_type in antenna_types:

    antenna = antenna_type()
    figure = antenna.plot_gain()

    filename = 'api_antenna_' + antenna_type.__name__.lower() + '_gain.png'
    figure.savefig(path.join(image_dir, filename))

plt.show()
