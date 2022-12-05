# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

from numpy.random import default_rng
from numpy.testing import assert_array_equal
from uhd_wrapper.utils.config import MimoSignal

from hermespy.core import Signal
from hermespy.hardware_loop.uhd import UsrpDevice
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestUsrpDevice(TestCase):

    def setUp(self) -> None:

        self.rng = default_rng(42)

        self.zerorpc_patch = patch('hermespy.hardware_loop.uhd.usrp.Client')
        self.client_patch = patch('hermespy.hardware_loop.uhd.usrp.UsrpClient')

        self.zerorpc_mock: MagicMock = self.zerorpc_patch.start()
        self.client_mock: MagicMock = self.client_patch.start()
        self.zerorpc_mock.return_value = self.zerorpc_mock
        self.client_mock.return_value = self.client_mock
        self.client_mock.getSupportedSamplingRates.return_value = [1., 2., 3., 4.]
        
        self.ip = '192.168.0.1'
        self.port = '9999'
        self.carrier_frequency = 1.123e4
        self.usrp = UsrpDevice(ip=self.ip, port=self.port, carrier_frequency=self.carrier_frequency)
        
    def tearDown(self) -> None:

        self.zerorpc_patch.stop()
        self.client_patch.stop()

    def test_init(self) -> None:
        """Test initialization routine"""
        
        self.zerorpc_mock.connect.assert_called_once()
        
        self.assertEqual(self.ip, self.usrp.ip)
        self.assertEqual(self.port, self.usrp.port)
        self.assertEqual(self.carrier_frequency, self.usrp.carrier_frequency)
        
    def test_reconfiguration(self) -> None:
        """Configuration should only be called if required"""
        
        self.usrp._configure_device()
        
        self.client_mock.configureRfConfig.assert_called_once()

    def test_upload(self) -> None:
        """Test the device upload subroutine"""

        transmitted_signal = Signal(self.rng.normal(size=(self.usrp.num_antennas, 11)),
                                    sampling_rate=self.usrp.sampling_rate,
                                    carrier_frequency=self.usrp.carrier_frequency)

        self.usrp._upload(transmitted_signal)

        self.client_mock.configureTx.assert_called_once()
        self.client_mock.configureRx.assert_called_once()

    def test_trigger(self) -> None:
        """Test the individual device trigger"""

        self.usrp.trigger()
        self.client_mock.execute.assert_called_once()

    def test_download(self) -> None:
        """Test the device download subroutine"""

        received_signal = Signal(self.rng.normal(size=(self.usrp.num_antennas, 11)),
                                 sampling_rate=self.usrp.sampling_rate,
                                 carrier_frequency=self.usrp.carrier_frequency)

        self.usrp._UsrpDevice__collection_enabled= True
        self.client_mock.collect.return_value = [MimoSignal([s for s in received_signal.samples])]
        signal = self.usrp._download()

        assert_array_equal(received_signal.samples, signal.samples)

    def test_client(self) -> None:
        """Test access to the UHD client"""

        self.assertIs(self.client_mock, self.usrp._client)

    def test_tx_gain_setget(self) -> None:
        """Transmit gain property getter should return setter argument"""

        gain = 1.23456
        self.usrp.tx_gain = gain

        self.assertEqual(gain, self.usrp.tx_gain)

    def test_rx_gain_setget(self) -> None:
        """Receive gain property getter should return setter argument"""

        gain = 1.23456
        self.usrp.rx_gain = gain

        self.assertEqual(gain, self.usrp.rx_gain)

    def test_max_sampling_rate(self) -> None:
        """Max sampling rate property should return the correct sampling rate"""

        self.assertEqual(4., self.usrp.max_sampling_rate)

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.usrp, {'antenna_positions'})
