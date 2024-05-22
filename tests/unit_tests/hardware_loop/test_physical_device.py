# -*- coding: utf-8 -*-
"""Test Physical Device functionalities."""

from os import path
from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch
from tempfile import TemporaryDirectory

import numpy as np
from h5py import File
from numpy.random import default_rng
from numpy.testing import assert_array_equal

from hermespy.core import DeviceInput, DeviceReception, ProcessedDeviceInput, Signal, SignalTransmitter
from hermespy.hardware_loop import DelayCalibration, NoDelayCalibration, NoLeakageCalibration, PhysicalDevice, SelectiveLeakageCalibration
from hermespy.simulation import SimulatedUniformArray, SimulatedIdealAntenna
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class PhysicalDeviceMock(PhysicalDevice):
    """Mock for a physical device."""

    __sampling_rate: float

    def __init__(self, sampling_rate: float, *args, **kwargs) -> None:
        PhysicalDevice.__init__(self, *args, **kwargs)
        self.__sampling_rate = sampling_rate
        self.__antennas = SimulatedUniformArray(SimulatedIdealAntenna, 1.0, (1, 1, 1))

    @property
    def antennas(self) -> SimulatedUniformArray:
        return self.__antennas

    @property
    def sampling_rate(self) -> float:
        return self.__sampling_rate

    @property
    def carrier_frequency(self) -> float:
        return 0.0

    def configure(self) -> None:
        pass

    def trigger(self) -> None:
        pass

    def fetch(self) -> None:
        pass

    @property
    def max_sampling_rate(self) -> float:
        return self.sampling_rate


class TestPhysicalDevice(TestCase):
    """Test the base class for all physical devices."""

    def setUp(self) -> None:
        self.rng = default_rng(42)
        self.sampling_rate = 1e6

        self.device = PhysicalDeviceMock(sampling_rate=self.sampling_rate)

    def test_calibration_init(self) -> None:
        """Calibration attributes should be properly initialized"""

        leakage_calibration = NoLeakageCalibration()
        delay_calibration = NoDelayCalibration()

        self.device = PhysicalDeviceMock(sampling_rate=self.sampling_rate, leakage_calibration=leakage_calibration, delay_calibration=delay_calibration)

        self.assertIs(leakage_calibration, self.device.leakage_calibration)
        self.assertIs(leakage_calibration.device, self.device)
        self.assertIs(delay_calibration, self.device.delay_calibration)
        self.assertIs(delay_calibration.device, self.device)

    def test_calibration_delay_setget(self) -> None:
        """Calibration delay property getter should return setter argument"""

        delay = 1.23456
        self.device.calibration_delay = delay

        self.assertEqual(delay, self.device.calibration_delay)

    def test_adaptive_sampling_setget(self) -> None:
        """Adaptive sampling property getter should return setter argument"""

        self.device.adaptive_sampling = True
        self.assertTrue(self.device.adaptive_sampling)

    def test_lowpass_filter_setget(self) -> None:
        """Lowpass filter property getter should return setter argument"""

        self.device.lowpass_filter = True
        self.assertTrue(self.device.lowpass_filter)

    def test_lowpass_bandwidth_setget(self) -> None:
        """Lopwass bandwidth property getter should return setter argument"""

        bandwidth = 3.45567
        self.device.lowpass_bandwidth = bandwidth

        self.assertEqual(bandwidth, self.device.lowpass_bandwidth)

    def test_lowpass_bandwidth_validation(self) -> None:
        """Lowpass bandwidth property setter should raise ValueError on invalid arguments"""

        try:
            self.device.lowpass_bandwidth = 0.0

        except ValueError:
            self.fail()

        with self.assertRaises(ValueError):
            self.device.lowpass_bandwidth = -1.0

    def test_max_receive_delay_setget(self) -> None:
        """Max receive delay property getter should return setter argument"""

        max_receive_delay = 3.45567
        self.device.max_receive_delay = max_receive_delay

        self.assertEqual(max_receive_delay, self.device.max_receive_delay)

    def test_max_receive_delay_validation(self) -> None:
        """Max receive delay property setter should raise ValueError on invalid arguments"""

        try:
            self.device.max_receive_delay = 0.0

        except ValueError:
            self.fail()

        with self.assertRaises(ValueError):
            self.device.max_receive_delay = -1.0

    def test_velocity(self) -> None:
        """Accessing the velocity property should raise a NotImplementedError"""

        with self.assertRaises(NotImplementedError):
            _ = self.device.velocity

    @patch.object(PhysicalDeviceMock, "_download")
    def test_estimate_noise_power(self, patch_download) -> None:
        """Noise power estimation should return the correct power estimate."""

        num_samples = 10000
        expected_noise_power = 0.1
        samples = 2**-0.5 * (self.rng.normal(size=num_samples, scale=expected_noise_power**0.5) + 1j * self.rng.normal(size=num_samples, scale=expected_noise_power**0.5))
        signal = Signal.Create(samples, sampling_rate=self.sampling_rate)
        patch_download.side_effect = lambda: signal

        noise_power = self.device.estimate_noise_power(num_samples)

        self.assertAlmostEqual(expected_noise_power, noise_power[0], places=2)
        assert_array_equal(noise_power, self.device.noise_power)

    def test_noise_power_setget(self) -> None:
        """Noise power property getter should return setter argument"""

        expected_noise_power = np.array([1])
        self.device.noise_power = expected_noise_power

        assert_array_equal(expected_noise_power, self.device.noise_power)

        self.device.noise_power = None
        self.assertIsNone(self.device.noise_power)

    def test_noise_power_validation(self) -> None:
        """Noise power property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.device.noise_power = np.array([-1, 0])

    @patch("hermespy.hardware_loop.physical_device.PhysicalDevice._upload")
    def test_transmit_no_adpative_sampling(self, _upload: MagicMock) -> None:
        """Test physical device extended transmit routine without adptive sampling"""

        transmitted_signal = Signal.Create(np.zeros((self.device.num_antennas, 10)), self.device.sampling_rate, self.device.carrier_frequency)
        transmitter = SignalTransmitter(transmitted_signal)
        self.device.transmitters.add(transmitter)

        self.device.adaptive_sampling = False
        transmission = self.device.transmit()

        _upload.assert_called_once()
        assert_array_equal(transmitted_signal[:, :], transmission.mixed_signal[:, :])

    @patch("hermespy.hardware_loop.physical_device.PhysicalDevice._upload")
    def test_transmit_adpative_sampling(self, _upload: MagicMock) -> None:
        """Test physical device extended transmit routine with adptive sampling"""

        transmitted_signal = Signal.Create(np.zeros((self.device.num_antennas, 10), dtype=complex), self.device.sampling_rate, self.device.carrier_frequency)

        transmitter_alpha = SignalTransmitter(transmitted_signal)
        transmitter_beta = SignalTransmitter(transmitted_signal)
        self.device.transmitters.add(transmitter_alpha)
        self.device.transmitters.add(transmitter_beta)

        self.device.adaptive_sampling = True
        transmission = self.device.transmit()

        _upload.assert_called_once()
        assert_array_equal(transmitted_signal[:, :], transmission.mixed_signal[:, :])

    def test_transmit_validation(self) -> None:
        """Phyiscal device extended transmit routine should raise RuntimeErrors on invalid configurations"""

        self.device.adaptive_sampling = True

        signal_alpha = Signal.Create(np.zeros((self.device.num_antennas, 10)), self.device.sampling_rate, self.device.carrier_frequency)
        transmitter_alpha = SignalTransmitter(signal_alpha)
        self.device.transmitters.add(transmitter_alpha)

        signal_beta = Signal.Create(np.zeros((self.device.num_antennas, 10)), 1 + self.device.sampling_rate, self.device.carrier_frequency)
        transmitter_beta = SignalTransmitter(signal_beta)
        self.device.transmitters.add(transmitter_beta)

        with self.assertRaises(RuntimeError):
            _ = self.device.transmit()

    @patch("hermespy.hardware_loop.physical_device.PhysicalDevice._download")
    def test_receive(self, _download: MagicMock) -> None:
        """Test physical device extended receive routine"""

        receiver = Mock()
        receiver.sampling_rate = self.device.sampling_rate
        receiver.selected_receive_ports = [i for i in range(self.device.num_receive_ports)]

        _download.return_value = Signal.Create(np.zeros((self.device.num_receive_ports, 10)), self.device.sampling_rate, self.device.carrier_frequency)
        self.device.lowpass_filter = True
        self.device.receivers.add(receiver)

        _ = self.device.process_input()
        receiver.cache_reception.assert_called_once()

        receiver.reset_mock()
        self.device.lowpass_bandwidth = 1.0

        _ = self.device.process_input()
        receiver.cache_reception.assert_called_once()

    @patch("hermespy.hardware_loop.physical_device.PhysicalDevice._download")
    def test_receive_validation(self, _download: MagicMock) -> None:
        """Receive routine should raise ValueErrors on invalid arguments"""

        with self.assertRaises(ValueError):
            _download.return_value = Signal.Create(np.zeros((3, 10)), self.device.sampling_rate, self.device.carrier_frequency)
            _ = self.device.process_input()

        with self.assertRaises(ValueError):
            _download.return_value = Signal.Create(np.zeros((self.device.num_receive_ports, 10)), self.device.sampling_rate + 1, self.device.carrier_frequency)
            _ = self.device.process_input()

    def test_download(self) -> None:
        """The download subroutine should raise a NotImplementedError"""

        with self.assertRaises(NotImplementedError):
            _ = self.device._download()

    def test_trigger_direct_validation(self) -> None:
        """Trigger routine should raise a ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.device.trigger_direct(Signal.Empty(self.device.sampling_rate, self.device.num_antennas, carrier_frequency=1.0))

        with self.assertRaises(ValueError):
            self.device.trigger_direct(Signal.Empty(566662, self.device.num_antennas, carrier_frequency=self.device.carrier_frequency))

    def test_process_device_input(self) -> None:
        """Test processing of device inputs"""

        input = DeviceInput(Signal.Empty(self.device.sampling_rate, self.device.num_antennas, carrier_frequency=self.device.carrier_frequency))
        processed_input = self.device.process_input(input)

        self.assertIsInstance(processed_input, ProcessedDeviceInput)

    def test_process_signal_sequence(self) -> None:
        """Test processing of a sequence of signals"""

        input = [Signal.Empty(self.device.sampling_rate, self.device.num_antennas, carrier_frequency=self.device.carrier_frequency)]
        processed_input = self.device.process_input(input)

        self.assertIsInstance(processed_input, ProcessedDeviceInput)

    def test_process_signal(self) -> None:
        """Test processing of a single signal"""

        input = Signal.Empty(self.device.sampling_rate, self.device.num_antennas, carrier_frequency=self.device.carrier_frequency)
        processed_input = self.device.process_input(input)

        self.assertIsInstance(processed_input, ProcessedDeviceInput)

    @patch("hermespy.hardware_loop.physical_device.PhysicalDevice._download")
    def test_lowpass_filter(self, _download: MagicMock) -> None:
        """Test lowpass filtering during input processing"""

        input_samples = self.device._rng.standard_normal((self.device.num_antennas, 512)) + 1j * self.device._rng.standard_normal((self.device.num_antennas, 512))
        _download.return_value = Signal.Create(input_samples, self.device.sampling_rate, self.device.carrier_frequency)

        # Enable lowpass filter
        self.device.lowpass_filter = True

        # Check with default cutoff frequency
        self.device.lowpass_bandwidth = 0.0
        default_filtered_samples = self.device.process_input().impinging_signals[0][:, :]

        # Check with specific cutoff frequency
        self.device.lowpass_bandwidth = 0.5 * self.device.sampling_rate
        filtered_samples = self.device.process_input().impinging_signals[0][:, :]

        assert_array_equal(default_filtered_samples, filtered_samples)

    def test_default_init(self) -> None:
        """Calibration attributes should be properly initialized with default values"""

        self.assertIsInstance(self.device.leakage_calibration, NoLeakageCalibration)
        self.assertIs(self.device, self.device.leakage_calibration.device)
        self.assertIsInstance(self.device.delay_calibration, NoDelayCalibration)
        self.assertIs(self.device, self.device.delay_calibration.device)

    def test_specific_init(self) -> None:
        """Calibration attributes should be properly initialized with specific values"""

        expected_leakage_calibration = NoLeakageCalibration()
        expected_delay_calibration = NoDelayCalibration()

        self.device = PhysicalDeviceMock(1.0, leakage_calibration=expected_leakage_calibration, delay_calibration=expected_delay_calibration)

        self.assertIs(self.device, self.device.leakage_calibration.device)
        self.assertIs(self.device, self.device.delay_calibration.device)

    def test_leakage_calibration_setget(self) -> None:
        """Leakage calibration property getter should return properly configured setter argument"""

        expected_leakage_calibration = NoLeakageCalibration()
        self.device.leakage_calibration = expected_leakage_calibration

        self.assertIs(expected_leakage_calibration, self.device.leakage_calibration)
        self.assertIs(self.device, expected_leakage_calibration.device)

    def test_delay_calibration_setget(self) -> None:
        """Delay calibration property getter should return properly configured setter argument"""

        expected_delay_calibration = NoDelayCalibration()
        self.device.delay_calibration = expected_delay_calibration

        self.assertIs(expected_delay_calibration, self.device.delay_calibration)
        self.assertIs(self.device, expected_delay_calibration.device)

    def test_save_load(self) -> None:
        """Test save and subsequential load functionality from HDF"""

        expected_leakage_response = np.array([[[0, 0, 1, 0, 1j]]], dtype=np.complex_)
        expected_leakage_calibration = SelectiveLeakageCalibration(expected_leakage_response, 1.0, 0.0)
        self.device.leakage_calibration = expected_leakage_calibration

        exepected_delay = 1.2345
        expected_delay_calibration = DelayCalibration(exepected_delay)
        self.device.delay_calibration = expected_delay_calibration

        recalled_device = PhysicalDeviceMock(1.0)
        patched_recalled_device = PhysicalDeviceMock(1.0)

        with TemporaryDirectory() as tmp_dir:
            file_path = path.join(tmp_dir, "test.hdf5")
            self.device.save_calibration(file_path)

            recalled_device.load_calibration(file_path)

            with patch("hermespy.hardware_loop.physical_device.DelayCalibrationBase.hdf_group_name", "xxxxxxx"):
                patched_recalled_device.load_calibration(file_path)

        self.assertIsInstance(recalled_device.leakage_calibration, SelectiveLeakageCalibration)
        assert_array_equal(recalled_device.leakage_calibration.leakage_response, expected_leakage_response)

        self.assertIsInstance(recalled_device.delay_calibration, DelayCalibration)
        self.assertEqual(recalled_device.delay_calibration.delay, exepected_delay)

        self.assertIsInstance(patched_recalled_device.delay_calibration, NoDelayCalibration)


class TestCalibration(TestCase):
    """Test calibration base class."""

    def setUp(self) -> None:
        self.calibration = NoDelayCalibration()

    def test_device_setget(self) -> None:
        """Calibaratable property getter should return properly configured setter argument"""

        expected_device = PhysicalDeviceMock(1.0)
        self.calibration.device = expected_device

        self.assertIs(expected_device, self.calibration.device)
        self.assertIs(self.calibration.device, expected_device)

        replaced_expected_device = PhysicalDeviceMock(1.0)
        self.calibration.device = replaced_expected_device

        self.assertIsNot(self.calibration, expected_device.delay_calibration)
        self.assertIs(self.calibration, replaced_expected_device.delay_calibration)

    def test_save_load(self) -> None:
        """Test saving and loading from and to HDF files"""

        with TemporaryDirectory() as tmp_dir:
            file_path = path.join(tmp_dir, "test.hdf5")
            self.calibration.save(file_path)

            recalled_file_calibration = self.calibration.Load(file_path)
            recalled_group_calibration = self.calibration.Load(File(file_path, "r"))

        self.assertIsInstance(recalled_file_calibration, NoDelayCalibration)
        self.assertIsInstance(recalled_group_calibration, NoDelayCalibration)


class TestNoDelayCalibration(TestCase):
    """Test delay calibration stub class."""

    def setUp(self) -> None:
        self.calibration = NoDelayCalibration()

    def test_delay_get(self) -> None:
        """Delay property should always return zero"""

        self.assertEqual(0.0, self.calibration.delay)

    def test_correct_transmit_delay(self) -> None:
        """Delays should be correctly corrected during transmission"""

        test_samples = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]], dtype=np.complex_)
        test_signal = Signal.Create(test_samples, 1.0)

        assert_array_equal(test_signal[:, :], self.calibration.correct_transmit_delay(test_signal)[:, :])

    def test_correct_receive_delay(self) -> None:
        """Delays should be correctly corrected during reception"""

        test_samples = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]], dtype=np.complex_)
        test_signal = Signal.Create(test_samples, 1.0)

        assert_array_equal(test_signal[:, :], self.calibration.correct_receive_delay(test_signal)[:, :])

    def test_yaml_serialization(self) -> None:
        """Test YAML serialization and deserialization"""

        test_yaml_roundtrip_serialization(self, self.calibration)


class TestNoLeakageCalibration(TestCase):
    """Test leakage calibration stub class."""

    def setUp(self) -> None:
        self.leakage_calibration = NoLeakageCalibration()

    def test_remove_leakage(self) -> None:
        """Nothing should be done to the signal during leakage removal"""

        transmitted_signal = Signal.Create(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]], dtype=np.complex_), 1.0)
        received_signal = Signal.Create(np.array([[1, 2, 3, 4, 2], [6, 7, 8, 1, 0]], dtype=np.complex_), 1.0)

        assert_array_equal(received_signal[:, :], self.leakage_calibration.remove_leakage(transmitted_signal, received_signal)[:, :])

    def test_yaml_serialization(self) -> None:
        """Test YAML serialization and deserialization"""

        test_yaml_roundtrip_serialization(self, self.leakage_calibration)

    def test_save_load(self) -> None:
        """Test saving and loading from and to HDF files"""

        with TemporaryDirectory() as tmp_dir:
            file_path = path.join(tmp_dir, "test.hdf5")
            self.leakage_calibration.save(file_path)

            recalled_file_calibration = self.leakage_calibration.Load(file_path)

        self.assertIsInstance(recalled_file_calibration, NoLeakageCalibration)
