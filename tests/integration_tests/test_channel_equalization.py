# -*- coding: utf-8 -*-
"""Test Channel Equalization."""

import unittest
from itertools import product
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.constants import pi

from hermespy.channel import MultipathFading5GTDL, MultipathFadingCost256
from hermespy.precoding import SymbolPrecoding, MMSETimeEqualizer, MMSESpaceEqualizer, ZFTimeEqualizer, ZFSpaceEqualizer
from hermespy.core.signal_model import Signal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "3.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestMeanSquareTimeEqualization(unittest.TestCase):
    """Test Minimum-Mean-Square-Error channel equalization in time-domain."""

    def setUp(self) -> None:

        self.precoding = SymbolPrecoding()
        self.equalizer = MMSETimeEqualizer()
        self.precoding[0] = self.equalizer

        self.transmitter = Mock()
        self.transmitter.antennas.num_antennas = 1
        self.receiver = Mock()
        self.receiver.antennas.num_antennas = 1

        self.rng = np.random.default_rng(42)
        self.random_mother = Mock()
        self.random_mother._rng = self.rng
        self.num_samples = 1000
        self.sampling_rate = 1e6
        self.samples = np.exp(2j * self.rng.uniform(0., pi, self.num_samples))
        self.signal = Signal(self.samples, self.sampling_rate)

        self.noise_variances = [0, 1e-4]
        self.rms_delays = [0., 1e-5]

    def test_equalize_5GTDL(self) -> None:
        """Test equalization of 5GTDL multipath fading channels."""

        for model_type, rms_delay, noise_variance in product(MultipathFading5GTDL.TYPE,
                                                             self.rms_delays,
                                                             self.noise_variances):

            channel = MultipathFading5GTDL(model_type=model_type,
                                           rms_delay=rms_delay,
                                           transmitter=self.transmitter,
                                           receiver=self.receiver)
            channel.random_mother = self.random_mother

            propagated_signal, _, channel_state = channel.propagate(self.signal)

            noise = (self.rng.normal(0., noise_variance ** .5, propagated_signal[0].samples.shape) +
                     1j * self.rng.normal(0., noise_variance ** .5, propagated_signal[0].samples.shape)) * 2 ** -.5
            noisy_signal = propagated_signal[0].samples + noise

            signal_power = np.var(self.signal.samples)
            equalized_signal = self.precoding.decode(noisy_signal, channel_state, noise_variance)
            equalized_signal_power = np.var(equalized_signal)

            self.assertAlmostEqual(signal_power, equalized_signal_power, places=1)

    def test_equalize_Cost256(self) -> None:
        """Test equalization of 5GTDL multipath fading channels."""

        for model_type, noise_variance in product(MultipathFadingCost256.TYPE, self.noise_variances):

            channel = MultipathFadingCost256(model_type=model_type,
                                             transmitter=self.transmitter,
                                             receiver=self.receiver)
            channel.random_mother = self.random_mother

            propagated_signal, _, channel_state = channel.propagate(self.signal)

            noise = (self.rng.normal(0., noise_variance ** .5, propagated_signal[0].samples.shape) +
                     1j * self.rng.normal(0., noise_variance ** .5, propagated_signal[0].samples.shape)) * 2 ** -.5
            noisy_signal = propagated_signal[0].samples + noise

            signal_power = np.var(self.signal.samples)
            equalized_signal = self.precoding.decode(noisy_signal, channel_state, noise_variance)
            equalized_signal_power = np.var(equalized_signal)

            self.assertAlmostEqual(signal_power, equalized_signal_power, places=1)


class TestMeanSquareSpaceEqualization(unittest.TestCase):
    """Test Zero-Forcing channel equalization in space-domain."""

    def setUp(self) -> None:

        self.precoding = SymbolPrecoding()
        self.equalizer = MMSESpaceEqualizer()
        self.precoding[0] = self.equalizer

        self.transmitter = Mock()
        self.transmitter.antennas.num_antennas = 1
        self.receiver = Mock()
        self.receiver.antennas.num_antennas = 1

        self.rng = np.random.default_rng(42)
        self.random_mother = Mock()
        self.random_mother._rng = self.rng
        self.num_samples = 100
        self.sampling_rate = 1e6

        self.num_antennas = [1, 2, 4]

    def test_equalize_5GTDL(self) -> None:
        """Test equalization of 5GTDL multipath fading channels."""

        for model_type, num_antennas in product(MultipathFading5GTDL.TYPE, self.num_antennas):

            self.transmitter.antennas.num_antennas = num_antennas
            self.receiver.antennas.num_antennas = num_antennas

            samples = np.exp(2j * self.rng.uniform(0., pi, (num_antennas, self.num_samples)))
            signal = Signal(samples, self.sampling_rate)
            noise = np.zeros((num_antennas, self.num_samples))

            channel = MultipathFading5GTDL(model_type=model_type,
                                           rms_delay=0.,
                                           transmitter=self.transmitter,
                                           receiver=self.receiver)
            channel.random_mother = self.random_mother

            propagated_signal, _, channel_state = channel.propagate(signal)

            equalized_signal, _, _ = self.equalizer.decode(propagated_signal[0].samples, channel_state, noise)
            assert_array_almost_equal(samples, equalized_signal)


class TestZeroForcingTimeEqualization(unittest.TestCase):
    """Test Zero-Forcing channel equalization in time-domain."""

    def setUp(self) -> None:

        self.precoding = SymbolPrecoding()
        self.equalizer = ZFTimeEqualizer()
        self.precoding[0] = self.equalizer

        self.transmitter = Mock()
        self.transmitter.antennas.num_antennas = 1
        self.receiver = Mock()
        self.receiver.antennas.num_antennas = 1

        self.generator = np.random.default_rng(42)
        self.random_mother = Mock()
        self.random_mother._rng = self.generator
        self.num_samples = 100
        self.sampling_rate = 1e6
        self.samples = np.exp(2j * self.generator.uniform(0., pi, self.num_samples))
        self.signal = Signal(self.samples, self.sampling_rate)

        self.rms_delays = [0., 1e-6, 1e-5]

    def test_equalize_5GTDL(self) -> None:
        """Test equalization of 5GTDL multipath fading channels."""

        for model_type, rms_delay in product(MultipathFading5GTDL.TYPE, self.rms_delays):

            channel = MultipathFading5GTDL(model_type=model_type,
                                           rms_delay=rms_delay,
                                           transmitter=self.transmitter,
                                           receiver=self.receiver)
            channel.random_mother = self.random_mother

            propagated_signal, _, channel_state = channel.propagate(self.signal)

            expected_propagated_samples = channel_state.linear[0, 0, :, :].todense() @ self.samples
            assert_array_almost_equal(expected_propagated_samples, propagated_signal[0].samples[0, :])

            equalized_signal = self.precoding.decode(propagated_signal[0].samples, channel_state, 0.)
            assert_array_almost_equal(self.samples, equalized_signal)

    def test_equalize_Cost256(self) -> None:
        """Test equalization of 5GTDL multipath fading channels."""

        for model_type in MultipathFadingCost256.TYPE:

            channel = MultipathFadingCost256(model_type=model_type,
                                             transmitter=self.transmitter,
                                             receiver=self.receiver)
            channel.random_mother = self.random_mother

            propagated_signal, _, channel_state = channel.propagate(self.signal)

            expected_propagated_samples = channel_state.linear[0, 0, :, :].todense() @ self.samples
            assert_array_almost_equal(expected_propagated_samples, propagated_signal[0].samples[0, :])

            equalized_signal = self.precoding.decode(propagated_signal[0].samples, channel_state, 0.)
            assert_array_almost_equal(self.samples, equalized_signal)


class TestZeroForcingSpaceEqualization(unittest.TestCase):
    """Test Zero-Forcing channel equalization in space-domain."""

    def setUp(self) -> None:

        self.precoding = SymbolPrecoding()
        self.equalizer = ZFSpaceEqualizer()
        self.precoding[0] = self.equalizer

        self.transmitter = Mock()
        self.transmitter.antennas.num_antennas = 1
        self.receiver = Mock()
        self.receiver.antennas.num_antennas = 1

        self.rng = np.random.default_rng(42)
        self.random_mother = Mock()
        self.random_mother._rng = self.rng
        self.num_samples = 100
        self.sampling_rate = 1e6

        self.num_antennas = [1, 2, 4]

    def test_equalize_5GTDL(self) -> None:
        """Test equalization of 5GTDL multipath fading channels."""

        for model_type, num_antennas in product(MultipathFading5GTDL.TYPE, self.num_antennas):

            self.transmitter.antennas.num_antennas = num_antennas
            self.receiver.antennas.num_antennas = num_antennas

            samples = np.exp(2j * self.rng.uniform(0., pi, (num_antennas, self.num_samples)))
            signal = Signal(samples, self.sampling_rate)
            noise = np.zeros((num_antennas, self.num_samples))

            channel = MultipathFading5GTDL(model_type=model_type,
                                           rms_delay=0.,
                                           transmitter=self.transmitter,
                                           receiver=self.receiver)
            channel.random_mother = self.random_mother

            propagated_signal, _, channel_state = channel.propagate(signal)

            equalized_signal, _, _ = self.equalizer.decode(propagated_signal[0].samples, channel_state, noise)
            assert_array_almost_equal(samples, equalized_signal)
