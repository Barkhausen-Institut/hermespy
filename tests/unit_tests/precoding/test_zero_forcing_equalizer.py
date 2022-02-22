# -*- coding: utf-8 -*-
"""Test Minimum-Mean-Square channel equalization."""

import unittest
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.constants import pi

from hermespy.channel import ChannelStateFormat, ChannelStateInformation
from hermespy.precoding import ZFSpaceEqualizer

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestZeroForcingSpaceEqualizer(unittest.TestCase):

    def setUp(self) -> None:

        # Random rng
        self.generator = np.random.default_rng(42)

        # Precoder to be tested
        self.precoder = ZFSpaceEqualizer()

        # Mock the precoding configuration
        self.precoding = Mock()
        self.precoding.required_outputs = lambda precoder: 1
        self.precoding.required_inputs = lambda precoder: 1
        self.precoder.precoding = self.precoding

    def test_encode(self) -> None:
        """Calling encode should raise a NotImplementedError."""

        stream = self.generator.random((5, 10))
        assert_array_equal(stream, self.precoder.encode(stream))

    def test_decode_noiseless(self) -> None:
        """Decode should properly equalize the provided stream response in the absence of noise."""

        num_samples = 100
        num_streams = 4

        expected_symbols = np.exp(1j * self.generator.uniform(0, 2*pi, (num_streams, num_samples)))
        stream_responses = np.exp(1j * self.generator.uniform(0, 2*pi, (num_streams, num_streams, num_samples, 1)))
        channel_state = ChannelStateInformation(ChannelStateFormat.IMPULSE_RESPONSE, stream_responses)

        propagated_symbol_stream = np.empty((num_streams, num_samples), dtype=complex)
        expected_equalized_responses = np.zeros((num_streams, num_streams, num_samples, 1), dtype=complex)
        for symbol_idx, (response, symbol) in enumerate(zip(stream_responses.transpose((2, 0, 1, 3)),
                                                            expected_symbols.T)):

            propagated_symbol_stream[:, symbol_idx] = response[:, :, 0] @ symbol
            expected_equalized_responses[:, :, symbol_idx, 0] = np.identity(num_streams, dtype=complex)

        stream_noises = np.zeros((num_streams, num_samples), dtype=float)

        equalized_symbols, equalized_csi, equalized_noises = self.precoder.decode(propagated_symbol_stream,
                                                                                  channel_state,
                                                                                  stream_noises)

        assert_array_almost_equal(expected_symbols, equalized_symbols)
        assert_array_almost_equal(expected_equalized_responses, equalized_csi.state)
        assert_array_almost_equal(equalized_noises, np.zeros((num_streams, num_samples), dtype=float))

    def test_num_input_streams(self) -> None:
        """The number of input streams should always be equal to the number of required output streams."""

        for num_outputs in [1, 5, 10]:

            self.precoding.required_outputs = lambda precoder: num_outputs
            self.assertEqual(num_outputs, self.precoder.num_input_streams)

    def test_num_output_streams(self) -> None:
        """The number of output streams should always be equal to the number of required input streams."""

        for num_inputs in [1, 5, 10]:

            self.precoding.required_inputs = lambda precoder: num_inputs
            self.assertEqual(num_inputs, self.precoder.num_output_streams)
