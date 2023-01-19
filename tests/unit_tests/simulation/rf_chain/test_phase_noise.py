# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal

from hermespy.core import Signal
from hermespy.simulation.rf_chain.phase_noise import NoPhaseNoise

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestNoPhaseNoise(TestCase):
    """Test the phase noise stub"""
    
    def setUp(self) -> None:
        
        self.pn = NoPhaseNoise()
    
    def test_add_noise(self) -> None:
        """Adding noise should actually do nothing"""
        
        signal = Signal(np.random.standard_normal((3, 10)), 1)
        noisy_signal = self.pn.add_noise(signal)
        
        assert_array_equal(signal.samples, noisy_signal.samples)
