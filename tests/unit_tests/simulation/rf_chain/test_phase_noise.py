# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from hermespy.core import Signal
from hermespy.simulation.rf_chain.phase_noise import NoPhaseNoise, PowerLawPhaseNoise


__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
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


class TestPowerLawPhaseNoise(TestCase):
    """Test power law phase noise model implementation"""
    
    def setUp(self) -> None:
        
        k1 = 1e-3
        k2 = 1e-2
        k3 = 1e-1
        k4 = 1
        cutoff = 1e-3
        vicinity = 1e-3
        num_flicker_subterms = 11
        
        self.pn = PowerLawPhaseNoise(k1, k2, k3, k4, cutoff, vicinity, num_flicker_subterms)
        
    def test_vicinity_validation(self) -> None:
        """Vicinity property should raise ValueError on negative arguments"""
        
        with self.assertRaises(ValueError):
            self.pn.vicinity = -1.
            
    def test_flicker_scale_validation(self) -> None:
        """Flicker scale property should raise ValueError on negative arguments"""
        
        with self.assertRaises(ValueError):
            self.pn.flicker_scale = -1.
            
    def test_flicker_num_subterms_validation(self) -> None:
        """Number of flicker subterms property should raise ValueError on invalid arguments"""
        
        with self.assertRaises(ValueError):
            self.pn.flicker_num_subterms = 0
            
        with self.assertRaises(ValueError):
            self.pn.flicker_num_subterms = -1
            
    def test_white_fm_validation(self) -> None:
        """White FM scale property should raise ValueError on negative arguments"""
        
        with self.assertRaises(ValueError):
            self.pn.white_fm_scale = -1.

    def test_flicker_fm_validation(self) -> None:
        """Flicker FM scale property should raise ValueError on negative arguments"""
        
        with self.assertRaises(ValueError):
            self.pn.flicker_fm_scale = -1.
            
    def test_gauss_cutoff_validation(self) -> None:
        """Gauss cutoff property should raise ValueError on negative arguments"""
        
        with self.assertRaises(ValueError):
            self.pn.gauss_cutoff = -1.
            
    def test_random_walk_fm_validation(self) -> None:
        """Random walk FM scale property should raise ValueError on negative arguments"""
        
        with self.assertRaises(ValueError):
            self.pn.random_walk_fm_scale = -1.
            
    def test_plot_psds(self) -> None:
        """Plotting the PSD charactersitics should return a Figure handle"""
        
        with patch('matplotlib.pyplot.figure') as figure_patch:
            
            _ = self.pn.plot_psds(100, 10)
            figure_patch.assert_called()
            
    def test_add_noise(self) -> None:
        """Adding noise should properly distort the input signal"""
        
        signal = Signal(np.random.standard_normal((3, 10)), 1)
        noisy_signal = self.pn.add_noise(signal)
        
        self.assertCountEqual(signal.samples.shape, noisy_signal.samples.shape)
        assert_array_almost_equal(signal.power, noisy_signal.power)
        self.assertRaises(AssertionError, assert_array_equal, signal.samples, noisy_signal.samples)

    def test_psd_caching(self) -> None:
        """The power spectral density subroutine should correctly cache its output"""

        num_samples = 100
        sampling_rate = 1e3

        psd = self.pn._psd(sampling_rate, num_samples)

        self.pn.random_walk_fm_scale = 10
        new_psd = self.pn._psd(sampling_rate, num_samples)
        self.assertRaises(AssertionError, assert_array_equal, psd, new_psd)

    def test_psd_validation(self) -> None:
        """PSD subroutine should raise a ValueError on number of samples smaller than two"""

        with self.assertRaises(ValueError):
            _ = self.pn._psd(100, 1)

        with self.assertRaises(ValueError):
            _ = self.pn._psd(100, -1)
