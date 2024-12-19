# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np
from unittest.mock import Mock, patch
from numpy.testing import assert_array_almost_equal

from hermespy.core import Reception, Receiver, ReceiveState, Signal, Transmission, TransmitState, Transmitter
from hermespy.hardware_loop.audio import AudioDevice
from hermespy.hardware_loop.audio.device import AudioAntenna, AudioDeviceAntennas, AudioPlaybackPort, AudioRecordPort
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SineOperator(Transmitter[Transmission], Receiver[Reception]):
    """Operator transmitting a sine wave for testing purposes."""

    __duration: float
    __frequency: float

    def __init__(self, duration=3, frequency=-5e3) -> None:
        Transmitter.__init__(self)
        Receiver.__init__(self)
        self.__duration = duration
        self.__frequency = frequency

    @property
    def power(self) -> float:
        return 1.0

    def _transmit(self, device: TransmitState, duration: float) -> Transmission:
        sine = np.exp(2j * np.pi * np.arange(int(self.__duration * self.sampling_rate)) / self.sampling_rate * self.__frequency)
        signal = Signal.Create(sine[np.newaxis, :], device.sampling_rate, device.carrier_frequency)

        transmission = Transmission(signal)
        return transmission

    def _receive(self, signal: Signal, device: ReceiveState) -> Reception:
        reception = Reception(signal)
        return reception

    @property
    def sampling_rate(self) -> float:
        return 40e3

    @property
    def frame_duration(self) -> float:
        return self.__duration

    @property
    def energy(self) -> float:
        return 1.0

    def _noise_power(self, strength: float, snr_type=...) -> float:
        return strength

    def _recall_transmission(self, group) -> Transmission:
        return Transmission.from_HDF(group)

    def _recall_reception(self, group) -> Reception:
        return Reception.from_HDF(group)


class TestAudioAntenna(TestCase):
    """Test audio antenna model."""

    def setUp(self) -> None:
        self.antenna = AudioAntenna()

    def test_copy(self) -> None:
        """Test copying the antenna stub"""

        copy = self.antenna.copy()
        self.assertEqual(self.antenna.mode, copy.mode)
        assert_array_almost_equal(self.antenna.pose, copy.pose)

    def test_characteristics(self) -> None:
        """Audio device antenna should always return ideal characteristics"""

        self.assertCountEqual(np.array([2**0.5, 2**0.5], dtype=float), self.antenna.local_characteristics(0.0, 0.0))


class TestAudioPlaybackPort(TestCase):
    """Test audio playback port model."""

    def setUp(self) -> None:
        self.device = Mock()
        self.device.playback_channels = [1, 2, 3, 4, 5]
        self.device.record_channels = [1, 2, 3, 4, 5]
        self.antennas = AudioDeviceAntennas(self.device)

        self.channel = 5
        self.port = AudioPlaybackPort(self.antennas, self.channel)

    def test_properties(self) -> None:
        """Test audio playback port properties"""

        self.assertEqual(self.channel, self.port.channel)


class TestAudioRecordPort(TestCase):
    """Test audio record port model."""

    def setUp(self) -> None:

        self.device = Mock()
        self.device.playback_channels = [1, 2, 3, 4, 5]
        self.device.record_channels = [1, 2, 3, 4, 5]
        self.antennas = AudioDeviceAntennas(self.device)

        self.channel = 5
        self.port = AudioRecordPort(self.antennas, self.channel)

    def test_properties(self) -> None:
        """Test audio record port properties"""

        self.assertEqual(self.channel, self.port.channel)


class TestAudioDeviceAntennas(TestCase):
    def setUp(self) -> None:
        self.device = Mock()
        self.device.playback_channels = [1, 2, 3, 4, 5]
        self.device.record_channels = [1, 2, 3, 4, 5]

        self.antennas = AudioDeviceAntennas(self.device)

    def test_new_port(self) -> None:
        """Adding a new port should raise a RuntimeError"""

        with self.assertRaises(RuntimeError):
            self.antennas._new_port()

    def test_num_antennas(self) -> None:
        """Test numbero of transmit antennas calcualtion"""

        self.assertEqual(10, self.antennas.num_antennas)

    def test_num_receive_antennas(self) -> None:
        """Number of receive antennas property should return the correct number"""

        self.assertEqual(5, self.antennas.num_receive_antennas)

    def test_num_transmit_antennas(self) -> None:
        """Number of transmit antennas property should return the correct number"""

        self.assertEqual(5, self.antennas.num_transmit_antennas)

    def test_antennas(self) -> None:
        """Antennas property should alwys return a list of antenna instances"""

        self.assertEqual(10, len(self.antennas.antennas))

    def test_transmit_ports(self) -> None:
        """Transmit ports property should return the correct list of transmit ports"""

        self.assertEqual(5, len(self.antennas.transmit_ports))

    def test_receive_ports(self) -> None:
        """Receive ports property should return the correct list of receive ports"""

        self.assertEqual(5, len(self.antennas.receive_ports))


class TestAudioDevice(TestCase):
    def setUp(self) -> None:
        self.device = AudioDevice(6, 4, [1], [1])

    def test_playback_device_setget(self) -> None:
        """Playback device property getter should return setter argument"""

        device = 1
        self.device.playback_device = device

        self.assertEqual(device, self.device.playback_device)

    def test_playback_device_validation(self) -> None:
        """Playback device property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.device.playback_device = -1

    def test_record_device_setget(self) -> None:
        """Record device property getter should return setter argument"""

        device = 1
        self.device.record_device = device

        self.assertEqual(device, self.device.record_device)

    def test_record_device_validation(self) -> None:
        """Record device property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.device.record_device = -1

    def test_playback_channels_setget(self) -> None:
        """Playback channels property getter should return setter argument"""

        channels = [1, 2, 3, 4]
        self.device.playback_channels = channels

        self.assertCountEqual(channels, self.device.playback_channels)

    def test_record_channels_setget(self) -> None:
        """Record channels property getter should return setter argument"""

        channels = [1, 2, 3, 4]
        self.device.record_channels = channels

        self.assertCountEqual(channels, self.device.record_channels)

    def test_sampling_rate_setget(self) -> None:
        """Sampling rate property getter should return setter argument"""

        sampling_rate = 1.0
        self.device.sampling_rate = sampling_rate

        self.assertEqual(sampling_rate, self.device.sampling_rate)

    def test_sampling_rate_validation(self) -> None:
        """Sampling rate property setter should raise ValueErrors on invalid arguments"""

        with self.assertRaises(ValueError):
            self.device.sampling_rate = -1.0

        with self.assertRaises(ValueError):
            self.device.sampling_rate = 0.0

    def test_max_sampling_rate(self) -> None:
        """Maximal sampling rate property should return configured sampling rate"""

        self.device.sampling_rate = 1.0
        self.assertEqual(1.0, self.device.max_sampling_rate)

    def test_transmit_receive(self) -> None:
        """Test all device stages."""

        def side_effect(*args, **kwargs):
            self.device._AudioDevice__reception = args[0]

        sd_mock = Mock()
        sd_mock.playrec.side_effect = side_effect
        with patch.object(self.device, "_import_sd") as import_sd_mock:
            import_sd_mock.return_value = sd_mock

            state = self.device.state()
            dsp = SineOperator()
            self.device.transmitters.add(dsp)
            self.device.receivers.add(dsp)

            transmission = self.device.transmit(state)
            self.device.trigger()
            processed_input = self.device.process_input(state=state)
            reception = self.device.receive(processed_input.operator_inputs[0], state)

        assert_array_almost_equal(transmission.operator_transmissions[0].signal.getitem(), reception.operator_inputs[0].getitem())

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.device, {"antenna_positions"})
