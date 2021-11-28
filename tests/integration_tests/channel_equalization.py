# -*- coding: utf-8 -*-
"""Test Channel Equalization."""

import unittest
from itertools import product
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.constants import pi

from hermespy.channel import MultipathFading5GTDL, MultipathFadingCost256
from hermespy.precoding import SymbolPrecoding, MMSETimeEqualizer, ZFTimeEqualizer
from hermespy.signal import Signal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestMMSEChannelEqualization(unittest.TestCase):

    def setUp(self) -> None:

        self.precoding = SymbolPrecoding()
        self.equalizer = MMSETimeEqualizer()
        self.precoding[0] = self.equalizer

        self.transmitter = Mock()
        self.transmitter.num_antennas = 1
        self.receiver = Mock()
        self.receiver.num_antennas = 1

        self.generator = np.random.default_rng(42)
        self.num_samples = 100
        self.sampling_rate = 1e6
        self.samples = np.exp(2j * self.generator.uniform(0., pi, self.num_samples))
        self.signal = Signal(self.samples, self.sampling_rate)

        self.noise_variances = [0.]  # [1e-4, 1e-3, 1e-2]
        self.rms_delays = [0., 1e-6, 1e-5]

    def test_equalize_5GTDL(self) -> None:
        """Test equalization of 5GTDL multipath fading channels."""

        for noise_variance, model_type, rms_delay in product(self.noise_variances,
                                                             MultipathFading5GTDL.TYPE,
                                                             self.rms_delays):

            channel = MultipathFading5GTDL(model_type=model_type,
                                           rms_delay=rms_delay,
                                           transmitter=self.transmitter,
                                           receiver=self.receiver,
                                           random_generator=self.generator)

            propagated_signal, channel_state = channel.propagate(self.signal)

            expected_propagated_samples = channel_state.linear[0, 0, :, :].todense() @ self.samples
            assert_array_almost_equal(expected_propagated_samples, propagated_signal.samples[0, :])

            cs = channel_state.linear[0, 0, :, :].todense()
            from scipy.linalg import inv

            corr = cs @ cs.T.conj()
            norm = np.diag(corr)
            #equalizer = cs.T.conj() @ inv(corr)
            equalizer = np.linalg.pinv(cs)

            equalized_signal = equalizer @ propagated_signal.samples[0, :]

            assert_array_almost_equal(self.samples, equalized_signal)

            #propagated_signal.samples += self.generator.normal(0, noise_variance, propagated_signal.num_samples)


            #equalized_signal = self.precoding.decode(propagated_signal.samples, channel_state, noise_variance)
            #assert_array_almost_equal(self.samples, equalized_signal, decimal=1)


class TestZeroForcingChannelEqualization(unittest.TestCase):

    def setUp(self) -> None:

        self.precoding = SymbolPrecoding()
        self.equalizer = ZFTimeEqualizer()
        self.precoding[0] = self.equalizer

        self.transmitter = Mock()
        self.transmitter.num_antennas = 1
        self.receiver = Mock()
        self.receiver.num_antennas = 1

        self.generator = np.random.default_rng(42)
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
                                           receiver=self.receiver,
                                           random_generator=self.generator)

            propagated_signal, channel_state = channel.propagate(self.signal)

            expected_propagated_samples = channel_state.linear[0, 0, :, :].todense() @ self.samples
            assert_array_almost_equal(expected_propagated_samples, propagated_signal.samples[0, :])

            equalized_signal = self.precoding.decode(propagated_signal.samples, channel_state, 0.)
            assert_array_almost_equal(self.samples, equalized_signal)

    def test_equalize_Cost256(self) -> None:
        """Test equalization of 5GTDL multipath fading channels."""

        for model_type in MultipathFadingCost256.TYPE:

            channel = MultipathFadingCost256(model_type=model_type,
                                             transmitter=self.transmitter,
                                             receiver=self.receiver,
                                             random_generator=self.generator)

            propagated_signal, channel_state = channel.propagate(self.signal)

            expected_propagated_samples = channel_state.linear[0, 0, :, :].todense() @ self.samples
            assert_array_almost_equal(expected_propagated_samples, propagated_signal.samples[0, :])

            equalized_signal = self.precoding.decode(propagated_signal.samples, channel_state, 0.)
            assert_array_almost_equal(self.samples, equalized_signal)
