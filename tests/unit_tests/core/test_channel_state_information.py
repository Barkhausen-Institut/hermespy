# -*- coding: utf-8 -*-
"""Test Channel State Information model for wireless transmission links."""

from unittest import TestCase
from unittest.mock import patch, PropertyMock

from numpy import exp
from numpy.random import default_rng
from numpy.testing import assert_array_equal  # , assert_array_almost_equal
from numpy.linalg import norm
from scipy.constants import pi

from hermespy.core.channel_state_information import ChannelStateInformation, ChannelStateFormat

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestChannelStateInformation(TestCase):
    """Test Channel State Information."""

    def setUp(self) -> None:

        self.generator = default_rng(42)

        self.num_rx_streams = 2
        self.num_tx_streams = 3
        self.num_samples = 10
        self.num_information = 10
        self.state_format = ChannelStateFormat.IMPULSE_RESPONSE
        self.state = exp(2j * self.generator.uniform(0, pi, (self.num_rx_streams, self.num_tx_streams,
                                                             self.num_samples, self.num_information)))

        self.csi = ChannelStateInformation(self.state_format, self.state)

    def test_init(self) -> None:
        """Init parameters should be properly stored as class attributes."""

        self.assertEqual(self.state_format, self.csi.state_format)
        assert_array_equal(self.state, self.csi.state)

    def test_state_format_get(self) -> None:
        """State format property should return the current state format enum."""

        with patch.object(self.csi, '_ChannelStateInformation__state_format', new_callable=PropertyMock) as mock:
            self.assertIs(mock, self.csi.state_format)

    def test_state_setget(self) -> None:
        """State property getter should return setter argument."""

        state = exp(2j * self.generator.uniform(0, pi, (self.num_rx_streams, self.num_tx_streams,
                                                        self.num_samples, self.num_information)))

        self.csi.state = state
        assert_array_equal(state, self.csi.state)

    def test_set_state(self) -> None:
        """Set state function should properly modify the channel state."""

        for state_format in [ChannelStateFormat.FREQUENCY_SELECTIVITY, ChannelStateFormat.IMPULSE_RESPONSE]:

            state = exp(2j * self.generator.uniform(0, pi, (self.num_rx_streams, self.num_tx_streams,
                                                            self.num_samples, self.num_information)))

            self.csi.set_state(state_format, state)
            self.assertEqual(state_format, self.csi.state_format)
            assert_array_equal(state, self.csi.state)

    def test_impulse_response_no_conversion(self) -> None:
        """Test impulse response property get without conversion."""

        state = exp(2j * self.generator.uniform(0, pi, (self.num_rx_streams, self.num_tx_streams,
                                                        self.num_samples, self.num_information)))
        self.csi.set_state(ChannelStateFormat.IMPULSE_RESPONSE, state)

        assert_array_equal(state, self.csi.to_impulse_response().state)
        self.assertEqual(ChannelStateFormat.IMPULSE_RESPONSE, self.csi.state_format)

    def test_impulse_response_conversion(self) -> None:
        """Test impulse response property get without conversion."""
        pass

    def test_frequency_selectivity_no_conversion(self) -> None:
        """Test impulse response property get without conversion."""

        state = exp(2j * self.generator.uniform(0, pi, (self.num_rx_streams, self.num_tx_streams,
                                                        self.num_samples, self.num_information)))
        self.csi.set_state(ChannelStateFormat.FREQUENCY_SELECTIVITY, state)

        assert_array_equal(state, self.csi.to_frequency_selectivity().state)
        self.assertEqual(ChannelStateFormat.FREQUENCY_SELECTIVITY, self.csi.state_format)

    def test_frequency_selectivity_conversion(self) -> None:
        """Test impulse response property get without conversion."""
        pass

#    def test_conversion_frequency_selectivity(self) -> None:
#        """Channel states should remain identical after a round-trip conversion
#        starting from frequency selectivity."""
#
#        state = exp(2j * self.generator.uniform(0, pi, (self.num_rx_streams, self.num_tx_streams,
#                                                        self.num_samples, self.num_information)))
#
#        self.csi.set_state(ChannelStateFormat.FREQUENCY_SELECTIVITY, state)
#        round_trip_state = self.csi.to_impulse_response().to_frequency_selectivity().state
#
#        assert_array_almost_equal(state, round_trip_state)

    def test_conversion_power_scaling(self) -> None:
        """Power of channel states should remain identical after a round-trip conversion."""

        state = exp(2j * self.generator.uniform(0, pi, (self.num_rx_streams, self.num_tx_streams,
                                                        self.num_samples, self.num_information)))

        self.csi.set_state(ChannelStateFormat.FREQUENCY_SELECTIVITY, state)
        round_trip_state = self.csi.to_impulse_response().to_frequency_selectivity().state

        self.assertAlmostEqual(norm(round_trip_state), norm(state))

    def test_num_receive_streams(self) -> None:
        """Number of receive streams property should report the correct matrix dimension."""

        num_rx_streams = 20
        state = exp(2j * self.generator.uniform(0, pi, (num_rx_streams, self.num_tx_streams,
                                                        self.num_samples, self.num_information)))

        self.csi.set_state(ChannelStateFormat.IMPULSE_RESPONSE, state)
        self.assertEqual(num_rx_streams, self.csi.num_receive_streams)

    def test_num_transmit_streams(self) -> None:
        """Number of transmit streams property should report the correct matrix dimension."""

        num_tx_streams = 20
        state = exp(2j * self.generator.uniform(0, pi, (self.num_rx_streams, num_tx_streams,
                                                        self.num_samples, self.num_information)))

        self.csi.set_state(ChannelStateFormat.IMPULSE_RESPONSE, state)
        self.assertEqual(num_tx_streams, self.csi.num_transmit_streams)

    def test_num_samples(self) -> None:
        """Number of samples property should report the correct matrix dimension."""

        num_samples = 20
        state = exp(2j * self.generator.uniform(0, pi, (self.num_rx_streams, self.num_tx_streams,
                                                        num_samples, self.num_information)))

        self.csi.set_state(ChannelStateFormat.IMPULSE_RESPONSE, state)
        self.assertEqual(num_samples, self.csi.num_samples)
