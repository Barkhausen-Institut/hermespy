from .factory import SerializableClasses, Factory
from .executable import Executable, Verbosity
from .simulation import Simulation, SNRType
from .hardware_loop import HardwareLoop

__author__ = "André Noll Barreto"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["André Barreto", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.2"
__maintainer__ = "André Noll Barreto"
__email__ = "andre.nollbarreto@barkhauseninstitut.org"
__status__ = "Prototype"


__all__ = ['Executable', 'Verbosity', 'Simulation', 'HardwareLoop', 'SNRType', 'Factory', 'SerializableClasses']

import sys
from inspect import getmembers
modules = ['channel', 'modem', 'scenario', 'simulator_core', 'source', 'coding', 'precoding', 'modem.tools',
           'noise', 'modem.rf_chain_models']
for module in modules:
    for _, member in getmembers(sys.modules['hermespy.' + module]):

        if hasattr(member, 'yaml_tag'):
            SerializableClasses.add(member)
