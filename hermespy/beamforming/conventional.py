# -*- coding: utf-8 -*-
"""
========================
Conventional Beamforming
========================

Also refererd to as Delay and Sum Beamformer.
"""

from hermespy.core import Serializable
from .beamformer import Beamformer


__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.7"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class ConventionalBeamformer(Serializable):
    """Conventional beamforming."""
    
    yaml_tag = u'ConventionalBeamformer'
    """YAML serialization tag."""
    
    
    