# -*- coding: utf-8 -*-

from unittest import TestCase

from hermespy.radar import FMCW
from hermespy.simulation import SimulatedDevice
from unit_tests.core.test_factory import test_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
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
        self.oversampling_factor = 4
        self.sampling_rate = self.oversampling_factor * self.bandwidth

        self.fmcw = FMCW(num_chirps=self.num_chirps, chirp_duration=self.chirp_duration, pulse_rep_interval=self.pulse_rep_interval)
        self.device = SimulatedDevice(bandwidth=self.bandwidth, oversampling_factor=self.oversampling_factor, carrier_frequency=1e9)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""

        self.assertEqual(self.num_chirps, self.fmcw.num_chirps)
        self.assertEqual(self.chirp_duration, self.fmcw.chirp_duration)
        self.assertEqual(self.pulse_rep_interval, self.fmcw.pulse_rep_interval)

    def test_ping_estimate(self) -> None:
        """Pinging and estimating should result in a valid velocity-range profile"""

        state = self.device.state()
        signal = self.fmcw.ping(state.transmit_state())
        estimate = self.fmcw.estimate(signal, state.receive_state())

        self.assertEqual(len(self.fmcw.relative_doppler_bins), estimate.shape[0])
        self.assertEqual(len(self.fmcw.range_bins(state.bandwidth)), estimate.shape[1])

    def test_ping_estimate_downsampling(self) -> None:
        """Pinging and estimating should result in a valid velocity-range profile when downsampled"""

        self.fmcw.adc_sampling_rate = 0.5 * self.bandwidth

        signal = self.fmcw.ping(self.device.state().transmit_state())
        estimate = self.fmcw.estimate(signal, self.device.state().receive_state())

        self.assertEqual(len(self.fmcw.relative_doppler_bins), estimate.shape[0])
        self.assertEqual(len(self.fmcw.range_bins(self.bandwidth)), estimate.shape[1])

    def test_frame_duration(self) -> None:
        """Frame duration should be the pulse repetition interval times the number of chirps"""

        self.assertEqual(self.fmcw.frame_duration(self.bandwidth), self.fmcw.pulse_rep_interval * self.fmcw.num_chirps)

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

    def test_chirp_duration_validation(self) -> None:
        """Chirp duration property setter should raise ValueError on arguments smaller or equal to zero"""

        with self.assertRaises(ValueError):
            self.fmcw.chirp_duration = 0.0

        with self.assertRaises(ValueError):
            self.fmcw.chirp_duration = -1.0

    def test_chirp_duration_setget(self) -> None:
        """Chirp duration property getter should return setter argument"""

        chirp_duration = 10.0
        self.fmcw.chirp_duration = 10.0

        self.assertEqual(chirp_duration, self.fmcw.chirp_duration)

    def test_adc_sampling_rate_validation(self) -> None:
        """ADC sampling rate property setter should raise ValueError on arguments smaller or equal to zero"""

        with self.assertRaises(ValueError):
            self.fmcw.adc_sampling_rate = -1.0

    def test_adc_sampling_rate_setget(self) -> None:
        """ADC sampling rate property getter should return setter argument"""

        adc_sampling_rate = 10.0
        self.fmcw.adc_sampling_rate = 10.0
        self.assertEqual(adc_sampling_rate, self.fmcw.adc_sampling_rate)

    def test_pulse_rep_interval_validation(self) -> None:
        """Pulse repetition interval property setter should raise ValueError on arguments smaller or equal to zero"""

        with self.assertRaises(ValueError):
            self.fmcw.pulse_rep_interval = 0.0

        with self.assertRaises(ValueError):
            self.fmcw.pulse_rep_interval = -1.0

    def test_pulse_rep_interval_setget(self) -> None:
        """Pulse repetition interval property getter should return setter argument"""

        pulse_rep_interval = 10.0
        self.fmcw.pulse_rep_interval = 10.0

        self.assertEqual(pulse_rep_interval, self.fmcw.pulse_rep_interval)

    def test_power(self) -> None:
        """A single chirp should have unit signal power"""

        self.fmcw.pulse_rep_interval = self.fmcw.chirp_duration
        self.fmcw.num_chirps = 1

        pulse = self.fmcw.ping(self.device.state().transmit_state())
        self.assertEqual(self.fmcw.power, pulse.power)

    def test_energy(self) -> None:
        """A single chirp should have the correct energy"""

        self.fmcw.pulse_rep_interval = self.fmcw.chirp_duration
        self.fmcw.num_chirps = 1

        pulse = self.fmcw.ping(self.device.state().transmit_state())
        self.assertEqual(self.fmcw.energy(self.bandwidth, self.oversampling_factor), pulse.energy)

    def test_pulse_prototype_validation(self) -> None:
        """Pulse prototype property setter should raise RuntimeError on invalid parameterization"""

        self.fmcw.pulse_rep_interval = 0.5 * self.fmcw.chirp_duration

        with self.assertRaises(RuntimeError):
            _ = self.fmcw.ping(self.device.state().transmit_state())

    def test_serialization(self) -> None:
        """Test FMCW radar waveform serialization"""

        test_roundtrip_serialization(self, self.fmcw)
