# -*- coding: utf-8 -*-
"""Test Minimum-Mean-Square channel equalization."""

import unittest
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_equal

from hermespy.channel import ChannelStateInformation, ChannelStateFormat
from hermespy.precoding import SpatialMultiplexing

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestSpatialMultiplexing(unittest.TestCase):

    def setUp(self) -> None:

        # Random rng
        self.generator = np.random.default_rng(42)

        # Precoder to be tested
        self.precoder = SpatialMultiplexing()

        # Mock the precoding configuration
        self.precoding = Mock()
        self.precoding.required_outputs = lambda precoder: 4
        self.precoding.required_inputs = lambda precoder: 1
        self.precoder.precoding = self.precoding

    def test_encode_decode_circular(self) -> None:
        """Encoding and subsequently decoding a data stream should lead to identical symbols."""

        input_stream = self.generator.random((1, 400))
        channel_state = ChannelStateInformation(ChannelStateFormat.IMPULSE_RESPONSE,
                                                self.generator.random((4, 1, 100, 1)))
        stream_noise = self.generator.random((4, 100))

        encoded_stream = self.precoder.encode(input_stream)
        decoded_stream, decoded_responses, _ = self.precoder.decode(encoded_stream, channel_state, stream_noise)

        assert_array_equal(input_stream, decoded_stream)

    def test_num_input_streams(self) -> None:
        """The number of input streams is always one."""

        for num_outputs in [1, 5, 10]:

            self.precoding.required_outputs = lambda precoder: num_outputs
            self.assertEqual(1, self.precoder.num_input_streams)

    def test_num_output_streams(self) -> None:
        """The number of output streams should always be equal to the number of required output streams."""

        for num_outputs in [1, 5, 10]:

            self.precoding.required_outputs = lambda precoder: num_outputs
            self.assertEqual(num_outputs, self.precoder.num_output_streams)

    def test_rate(self) -> None:
        """The rate should always be the fraction between input and output streams."""

        for num_outputs in [1, 5, 10]:

            self.precoding.required_outputs = lambda precoder: num_outputs
            self.assertEqual(1/num_outputs, float(self.precoder.rate))
