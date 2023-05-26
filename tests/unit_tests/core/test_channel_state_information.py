# -*- coding: utf-8 -*-
"""Test Channel State Information model for wireless transmission links"""

from os import path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
from h5py import File
from numpy import exp
from numpy.random import default_rng
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.constants import pi

from hermespy.core import ChannelStateDimension, ChannelStateInformation, ChannelStateFormat

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestChannelStateInformation(TestCase):
    """Test Channel State Information"""

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
        """Init parameters should be properly stored as class attributes"""

        self.assertEqual(self.state_format, self.csi.state_format)
        assert_array_equal(self.state, self.csi.state)

    def test_state_format_get(self) -> None:
        """State format property should return the current state format enum"""

        with patch.object(self.csi, '_ChannelStateInformation__state_format', new_callable=PropertyMock) as mock:
            self.assertIs(mock, self.csi.state_format)

    def test_state_setget(self) -> None:
        """State property getter should return setter argument"""

        state = exp(2j * self.generator.uniform(0, pi, (self.num_rx_streams, self.num_tx_streams,
                                                        self.num_samples, self.num_information)))

        self.csi.state = state
        assert_array_equal(state, self.csi.state)
        
    def test_set_state_validation(self) -> None:
        """Set state function should raise exceptions on invalid arguments"""

        with self.assertRaises(ValueError):
            self.csi.set_state(ChannelStateFormat.IMPULSE_RESPONSE, exp(2j * self.generator.uniform(0, pi, (1, 1, 1))))

        with self.assertRaises(ValueError):
            self.csi.set_state(ChannelStateFormat.IMPULSE_RESPONSE, exp(2j * self.generator.uniform(0, pi, (1, 1, 1, 1))), num_delay_taps=10)

    def test_set_state(self) -> None:
        """Set state function should properly modify the channel state"""

        for state_format in [ChannelStateFormat.FREQUENCY_SELECTIVITY, ChannelStateFormat.IMPULSE_RESPONSE]:

            state = exp(2j * self.generator.uniform(0, pi, (self.num_rx_streams, self.num_tx_streams,
                                                            self.num_samples, self.num_information)))

            self.csi.set_state(state_format, state)
            self.assertEqual(state_format, self.csi.state_format)
            assert_array_equal(state, self.csi.state)

    def test_impulse_response_no_conversion(self) -> None:
        """Test impulse response property get without conversion"""

        state = exp(2j * self.generator.uniform(0, pi, (self.num_rx_streams, self.num_tx_streams,
                                                        self.num_samples, self.num_information)))
        self.csi.set_state(ChannelStateFormat.IMPULSE_RESPONSE, state)

        assert_array_equal(state, self.csi.to_impulse_response().state)
        self.assertEqual(ChannelStateFormat.IMPULSE_RESPONSE, self.csi.state_format)

    def test_impulse_response_conversion(self) -> None:
        """Test impulse response property get without conversion"""
        
        expected_response = self.csi.state.copy()
        
        self.csi.to_frequency_selectivity()
        self.csi.to_impulse_response()
        
        assert_array_almost_equal(expected_response, self.csi.state)

    def test_frequency_selectivity_no_conversion(self) -> None:
        """Test impulse response property get without conversion"""

        state = exp(2j * self.generator.uniform(0, pi, (self.num_rx_streams, self.num_tx_streams,
                                                        self.num_samples, self.num_information)))
        self.csi.set_state(ChannelStateFormat.FREQUENCY_SELECTIVITY, state)

        assert_array_equal(state, self.csi.to_frequency_selectivity().state)
        self.assertEqual(ChannelStateFormat.FREQUENCY_SELECTIVITY, self.csi.state_format)

    def test_frequency_selectivity_conversion(self) -> None:
        """Test impulse response property conversion"""
        
        self.csi.to_frequency_selectivity()
        expected_response = self.csi.state.copy()
        
        self.csi.to_impulse_response()
        self.csi.to_frequency_selectivity(num_bins=expected_response.shape[2])
        
        assert_array_almost_equal(expected_response, self.csi.state)

    def test_num_receive_streams(self) -> None:
        """Number of receive streams property should report the correct matrix dimension"""

        num_rx_streams = 20
        state = exp(2j * self.generator.uniform(0, pi, (num_rx_streams, self.num_tx_streams,
                                                        self.num_samples, self.num_information)))

        self.csi.set_state(ChannelStateFormat.IMPULSE_RESPONSE, state)
        self.assertEqual(num_rx_streams, self.csi.num_receive_streams)

    def test_num_transmit_streams(self) -> None:
        """Number of transmit streams property should report the correct matrix dimension"""

        num_tx_streams = 20
        state = exp(2j * self.generator.uniform(0, pi, (self.num_rx_streams, num_tx_streams,
                                                        self.num_samples, self.num_information)))

        self.csi.set_state(ChannelStateFormat.IMPULSE_RESPONSE, state)
        self.assertEqual(num_tx_streams, self.csi.num_transmit_streams)

    def test_num_samples(self) -> None:
        """Number of samples property should report the correct matrix dimension"""

        num_samples = 20
        state = exp(2j * self.generator.uniform(0, pi, (self.num_rx_streams, self.num_tx_streams,
                                                        num_samples, self.num_information)))

        self.csi.set_state(ChannelStateFormat.IMPULSE_RESPONSE, state)
        self.assertEqual(num_samples, self.csi.num_samples)
        
    def test_num_symbols(self) -> None:
        """Number of symbols property should report the correct matrix dimension"""
        
        num_symbols = 20
        state = exp(2j * self.generator.uniform(0, pi, (self.num_rx_streams, self.num_tx_streams,
                                                        num_symbols, 1)))

        self.csi.set_state(ChannelStateFormat.IMPULSE_RESPONSE, state)
        self.assertEqual(num_symbols, self.csi.num_symbols)
        
        self.csi.set_state(ChannelStateFormat.FREQUENCY_SELECTIVITY, state)
        self.assertEqual(num_symbols, self.csi.num_symbols)
        
    def test_num_delay_taps(self) -> None:
        """Number of delay taps property should report the correct dimension"""
        
        self.assertEqual(self.num_samples, self.csi.num_delay_taps)
        
    def test_linear(self) -> None:
        """Test linear channel matrix generation"""
        
        self.csi.to_impulse_response()
        impulse_linear_csi = self.csi.linear
        
        self.csi.to_frequency_selectivity()
        frequency_linear_csi = self.csi.linear
        
        self.assertSequenceEqual((self.num_rx_streams, self.num_tx_streams, self.num_samples + self.num_information - 1, self.num_samples), impulse_linear_csi.shape)
        self.assertSequenceEqual((self.num_rx_streams, self.num_tx_streams, 100, 100), frequency_linear_csi.shape)


    def test_ideal_initialization(self) -> None:
        """Test ideal channel state initialization"""
        
        ideal_csi = ChannelStateInformation.Ideal(self.num_samples, self.num_rx_streams, self.num_tx_streams)
        assert_array_equal(np.ones((self.num_rx_streams, self.num_tx_streams, self.num_samples, 1)), ideal_csi.state)

    def test_received_streams(self) -> None:
        """Test received streams generator"""
        
        for stream_idx, received_stream in enumerate(self.csi.received_streams()):
            
            # Assert that the slice was selected correctly
            assert_array_equal(self.csi.state[stream_idx, :, :, :], received_stream.state[0, ::])

    def test_samples(self) -> None:
        """Test the samples generator"""

        for sample_idx, samples in enumerate(self.csi.samples()):
            
            # Assert that the slice was selected correctly
            assert_array_equal(self.csi.state[:, :, sample_idx, :], samples.state[:, :, 0, :])

    def test_item_getset(self) -> None:

        for rx, tx in np.ndindex(self.num_rx_streams, self.num_tx_streams):
            
            # Assert that the slice was selected correctly
            assert_array_equal(self.csi.state[rx, tx], self.csi[rx, tx, :, :].state[0, 0, :, :])
            
            # Assert that the slice was set correctly
            old_state = self.csi[rx, tx].state.copy()
            self.csi[rx, tx] = ChannelStateInformation(self.csi.state_format, -old_state)
            assert_array_equal(self.csi[rx, tx].state, -old_state)

    def test_set_item_validation(self) -> None:
        """Setting an item of invalid state format should raise a NotImplementedError"""
        
        with self.assertRaises(NotImplementedError):
            self.csi[0, 0] = ChannelStateInformation(ChannelStateFormat.FREQUENCY_SELECTIVITY, np.ones((1, 1, 1, 1)))

    def test_concatenate(self) -> None:
        """Test concatenating two channel state information objects"""
        
        csi = ChannelStateInformation(self.state_format, self.state)
        csi2 = ChannelStateInformation(self.state_format, self.state)
        
        concatenated_csi = ChannelStateInformation.concatenate([csi, csi2], dimension=ChannelStateDimension.SAMPLES)
        assert_array_equal(np.concatenate((csi.state, csi2.state), axis=2), concatenated_csi.state)

    @patch('matplotlib.pyplot.subplots')
    def test_plot(self, subplots_patch: MagicMock) -> None:
        
        figure_mock = MagicMock()
        axes_mock = MagicMock()
        subplots_patch.return_value = figure_mock, axes_mock
        
        self.csi.plot()
        subplots_patch.assert_called_once()
        
    def test_reciprocal(self) -> None:
        """Test reciprocal channel state information generation"""
        
        reciprocal_csi = self.csi.reciprocal()
        expected_csi = reciprocal_csi.reciprocal()
        
        assert_array_equal(expected_csi.state, self.csi.state)

    def test_hdf_serialization(self) -> None:
        """Serialization to and from HDF5 should yield the correct object reconstruction"""
        
        csi: ChannelStateInformation = None
        
        with TemporaryDirectory() as tempdir:
            
            file_location = path.join(tempdir, 'testfile.hdf5')
            
            with File(file_location, 'a') as file:
                
                group = file.create_group('testgroup')
                self.csi.to_HDF(group)
                
            with File(file_location, 'r') as file:
                
                group = file['testgroup']
                csi = self.csi.from_HDF(group)
                
        assert_array_equal(self.csi.state, csi.state)
        self.assertEqual(self.csi.state_format, csi.state_format)
