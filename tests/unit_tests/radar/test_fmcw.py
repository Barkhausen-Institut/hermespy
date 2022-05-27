# -*- coding: utf-8 -*-

from unittest import TestCase

from hermespy.radar import FMCW

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.7"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestFMCW(TestCase):
    """Test the FMCW radar waveform"""

    def setUp(self) -> None:

        self.num_chirps = 9
        self.bandwidth = 1.4e9
        self.sampling_rate = 1.4e9
        self.max_range = 20

        self.fmcw = FMCW(num_chirps=self.num_chirps, bandwidth=self.bandwidth, sampling_rate=self.sampling_rate, max_range=self.max_range)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""

        self.assertEqual(self.num_chirps, self.fmcw.num_chirps)
        self.assertEqual(self.bandwidth, self.fmcw.bandwidth)
        self.assertEqual(self.sampling_rate, self.fmcw.sampling_rate)
        self.assertEqual(self.max_range, self.fmcw.max_range)
    
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

    def test_range_resolution_setget(self) -> None:
        """Range resolution property getter should return setter argument"""

        range_resolution = 10.
        self.fmcw.range_resolution = 10.

        self.assertEqual(range_resolution, self.fmcw.range_resolution)

    def test_range_resolution_validation(self) -> None:
        """Range resolution property setter should raise ValueError on arguments smaller or equal to zero"""

        with self.assertRaises(ValueError):
            self.fmcw.range_resolution = 0.

        with self.assertRaises(ValueError):
            self.fmcw.range_resolution = -1.

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

    def test_max_range_setget(self) -> None:
        """Maximum range property getter should return setter argument"""

        max_range = 10.
        self.fmcw.max_range = 10.

        self.assertEqual(max_range, self.fmcw.max_range)

    def test_max_range_validation(self) -> None:
        """Maximum range property setter should raise ValueError on arguments smaller or equal to zero"""

        with self.assertRaises(ValueError):
            self.fmcw.max_range = 0.

        with self.assertRaises(ValueError):
            self.fmcw.max_range = -1.
