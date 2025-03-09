# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import MagicMock, patch

from numpy.random import default_rng
from numpy.testing import assert_array_equal
from uhd_wrapper.utils.config import MimoSignal
from zerorpc.exceptions import LostRemote, RemoteError

from hermespy.core import Signal, SignalTransmitter, SignalReceiver
from hermespy.hardware_loop.uhd import UsrpAntennas, UsrpDevice
from unit_tests.core.test_factory import test_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestUsrpAntennas(TestCase):
    """Test the USRP antenna configuration class"""

    def setUp(self) -> None:
        self.device = MagicMock(spec=UsrpDevice)
        self.device.num_digital_transmit_ports = 3
        self.device.num_digital_receive_ports = 4
        self.antennas = UsrpAntennas(self.device)

    def test_init(self) -> None:
        """Test initialization routine"""

        self.assertEqual(self.device, self.antennas.device)
        self.assertEqual(self.device.num_digital_transmit_ports, self.antennas.num_transmit_ports)
        self.assertEqual(self.device.num_digital_receive_ports, self.antennas.num_receive_ports)

    def test_receive_ports(self) -> None:
        """Receive ports property should return the correct number of ports"""

        self.assertEqual(4, len(self.antennas.receive_ports))

    def test_transmit_ports(self) -> None:
        """Transmit ports property should return the correct number of ports"""

        self.assertEqual(3, len(self.antennas.transmit_ports))

    def test_ports(self) -> None:
        """Ports property should return the correct number of ports"""

        self.assertEqual(7, len(self.antennas.ports))

    def test_num_ports(self) -> None:
        """Port count properties should return the correct number of ports"""

        self.assertEqual(4, self.antennas.num_receive_ports)
        self.assertEqual(3, self.antennas.num_transmit_ports)
        self.assertEqual(7, self.antennas.num_ports)


class TestUsrpDevice(TestCase):
    def setUp(self) -> None:
        self.rng = default_rng(42)

        self.ip = "192.168.0.1"
        self.port = 9999
        self.carrier_frequency = 1.123e4

        self.client_mock = MagicMock()
        self.client_mock.getSupportedSamplingRates.return_value = [1.0, 2.0, 3.0, 4.0]
        self.client_mock.ip = self.ip
        self.client_mock.port = self.port
        self.client_mock.getNumAntennas.return_value = 4

        self.client_create_patch = patch("hermespy.hardware_loop.uhd.usrp.UsrpClient.create")
        self.client_create_mock: MagicMock = self.client_create_patch.start()
        self.client_create_mock.return_value = self.client_mock

        self.usrp = UsrpDevice(ip=self.ip, port=self.port, carrier_frequency=self.carrier_frequency, num_prepended_zeros=0, num_appended_zeros=0)

    def tearDown(self) -> None:
        self.client_create_patch.stop()

    def test_init(self) -> None:
        """Test initialization routine"""

        self.assertEqual(self.ip, self.usrp.ip)
        self.assertEqual(self.port, self.usrp.port)
        self.assertEqual(self.carrier_frequency, self.usrp.carrier_frequency)

    def test_reconfiguration(self) -> None:
        """Configuration should only be called if required"""

        self.usrp._configure_device()

        self.client_mock.configureRfConfig.assert_called_once()

    def test_error_handling(self) -> None:
        """Test error handling behaviour"""

        def raiseLostRemote(*args, **kwargs):
            raise LostRemote()

        def raiseRemoteError(*args, **kwargs):
            raise RemoteError("a", "b", "c")

        signal = Signal.Empty(self.usrp.sampling_rate, self.usrp.antennas.num_transmit_antennas, carrier_frequency=self.usrp.carrier_frequency)

        self.client_mock.configureTx.side_effect = raiseLostRemote
        with self.assertRaises(RuntimeError):
            self.usrp.trigger_direct(signal)

        self.client_mock.configureTx.side_effect = raiseRemoteError
        with self.assertRaises(RuntimeError):
            self.usrp.trigger_direct(signal)

    def test_upload(self) -> None:
        """Test the device upload subroutine"""

        # Enable transmission scaling for increased coverage
        self.usrp.scale_transmission = True

        transmitted_signal = Signal.Create(self.rng.normal(size=(self.usrp.num_digital_transmit_ports, 11)), sampling_rate=self.usrp.sampling_rate, carrier_frequency=self.usrp.carrier_frequency)

        self.usrp._upload(transmitted_signal)

        self.client_mock.configureTx.assert_called_once()
        self.client_mock.configureRx.assert_called_once()

    def test_transmit(self) -> None:
        """Test transmitting operator behaviour"""

        transmitted_signal = Signal.Create(self.rng.normal(size=(self.usrp.num_digital_transmit_ports, 11)), sampling_rate=self.usrp.sampling_rate, carrier_frequency=self.usrp.carrier_frequency)
        transmitter = SignalTransmitter(transmitted_signal)
        self.usrp.transmitters.add(transmitter)

        receiver = SignalReceiver(16, self.usrp.sampling_rate)
        self.usrp.receivers.add(receiver)

        transmission = self.usrp.transmit()

        self.client_mock.configureTx.assert_called_once()
        self.client_mock.configureRx.assert_called_once()
        assert_array_equal(transmitted_signal.getitem(), transmission.mixed_signal.getitem())

    def test_receive_no_collection(self) -> None:
        """Test reception without enabled collection"""

        reception = self.usrp.receive()
        self.assertEqual(0, reception.impinging_signals[0].num_samples)

    def test_trigger(self) -> None:
        """Test the individual device trigger"""

        self.usrp.trigger()
        self.client_mock.executeImmediately.assert_called_once()

    def test_download(self) -> None:
        """Test the device download subroutine"""

        received_signal = Signal.Create(self.rng.normal(size=(self.usrp.num_digital_receive_ports, 11)), sampling_rate=self.usrp.sampling_rate, carrier_frequency=self.usrp.carrier_frequency)

        self.usrp._UsrpDevice__collection_enabled = True
        self.client_mock.collect.return_value = [MimoSignal([s for s in received_signal.getitem()])]
        signal = self.usrp._download()

        assert_array_equal(received_signal.getitem(), signal.getitem())

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

    def test_num_prepended_zeros_setget(self) -> None:
        """Number of prepended zeros property getter should return setter argument"""

        expected_num_zeros = 1234
        self.usrp.num_prepeneded_zeros = expected_num_zeros

        self.assertEqual(expected_num_zeros, self.usrp.num_prepeneded_zeros)

    def test_num_prepended_zeros_validation(self) -> None:
        """Number of prepended zeros property should raise ValueError on negative arguments"""

        try:
            self.usrp.num_prepeneded_zeros = 0

        except ValueError:
            self.fail()

        with self.assertRaises(ValueError):
            self.usrp.num_prepeneded_zeros = -1

    def test_num_appended_zeros_setget(self) -> None:
        """Number of appended zeros property getter should return setter argument"""

        expected_num_zeros = 1234
        self.usrp.num_prepeneded_zeros = expected_num_zeros

        self.assertEqual(expected_num_zeros, self.usrp.num_prepeneded_zeros)

    def test_num_appended_zeros_validation(self) -> None:
        """Number of appended zeros property should raise ValueError on negative arguments"""

        try:
            self.usrp.num_appended_zeros = 0

        except ValueError:
            self.fail()

        with self.assertRaises(ValueError):
            self.usrp.num_appended_zeros = -1

    def test_max_sampling_rate(self) -> None:
        """Max sampling rate property should return the correct sampling rate"""

        self.assertEqual(4.0, self.usrp.max_sampling_rate)

    def test_serialization(self) -> None:
        """Test USRP device serialization"""

        test_roundtrip_serialization(self, self.usrp, {"antenna_positions"})
