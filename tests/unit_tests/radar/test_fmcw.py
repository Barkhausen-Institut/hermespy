# -*- coding: utf-8 -*-

from unittest import TestCase

from hermespy.radar import FMCW
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestFMCW(TestCase):
    """Test the FMCW radar waveform"""

    def setUp(self) -> None:

        self.num_chirps = 9
        self.bandwidth = 1.4e9
        self.chirp_duration = 2e-6
        self.pulse_rep_interval = 2e-6
        self.sampling_rate = 4 * self.bandwidth

        self.fmcw = FMCW(num_chirps=self.num_chirps, bandwidth=self.bandwidth,
                         chirp_duration=self.chirp_duration, pulse_rep_interval=self.pulse_rep_interval,
                         sampling_rate=self.sampling_rate)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""

        self.assertEqual(self.num_chirps, self.fmcw.num_chirps)
        self.assertEqual(self.bandwidth, self.fmcw.bandwidth)
        self.assertEqual(self.chirp_duration, self.fmcw.chirp_duration)
        self.assertEqual(self.pulse_rep_interval, self.fmcw.pulse_rep_interval)
        self.assertEqual(self.sampling_rate, self.fmcw.sampling_rate)
    
    def test_ping_estimate(self) -> None:
        """Pinging and estimating should result in a valid velocity-range profile"""

        signal = self.fmcw.ping()
        estimate = self.fmcw.estimate(signal)

        self.assertEqual(len(self.fmcw.velocity_bins), estimate.shape[0])
        self.assertEqual(len(self.fmcw.range_bins), estimate.shape[1])

    def test_num_chirps_setget(self) -> None:
        """Number of chirps property getter should return setter argument"""

        num_chirps = 15
        self.fmcw.num_chirps = num_chirps

        self.assertEqual(num_chirps, self.fmcw.num_chirps)

    def test_num_chirps_validation(self) -> None:
        """Number of chirps property setter should raise ValueError on arguments samller than one"""

        with self.assertRaises(ValueError):
            self.fmcw.num_chirps = 0

        with self.assertRaises(ValueError):
            self.fmcw.num_chirps = -1

    def test_bandwidth_setget(self) -> None:
        """Bandwidth property getter should return setter argument"""

        bandwidth = 10.
        self.fmcw.bandwidth = 10.

        self.assertEqual(bandwidth, self.fmcw.bandwidth)

    def test_bandwidth_validation(self) -> None:
        """Bandwidth property setter should raise ValueError on arguments smaller or equal to zero"""

        with self.assertRaises(ValueError):
            self.fmcw.bandwidth = 0.

        with self.assertRaises(ValueError):
            self.fmcw.bandwidth = -1.

    def test_sampling_rate_setget(self) -> None:
        """Sampling rate property getter should return setter argument"""

        sampling_rate = 10.
        self.fmcw.sampling_rate = 10.

        self.assertEqual(sampling_rate, self.fmcw.sampling_rate)

    def test_sampling_rate_validation(self) -> None:
        """Sampling rate property setter should raise ValueError on arguments smaller or equal to zero"""

        with self.assertRaises(ValueError):
            self.fmcw.sampling_rate = 0.

        with self.assertRaises(ValueError):
            self.fmcw.sampling_rate = -1.
    
    def test_power(self) -> None:
        """A single chirp should have unit signal power"""
        
        self.fmcw.pulse_rep_interval = self.fmcw.chirp_duration
        self.fmcw.num_chirps = 1
        
        pulse = self.fmcw.ping()
        self.assertEqual(self.fmcw.power, pulse.power)

    def test_energy(self) -> None:
        """A single chirp should have the correct energy"""
        
        self.fmcw.pulse_rep_interval = self.fmcw.chirp_duration
        self.fmcw.num_chirps = 1
        
        pulse = self.fmcw.ping()
        self.assertEqual(self.fmcw.energy, pulse.energy)

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.fmcw, {'max_velocity',})
