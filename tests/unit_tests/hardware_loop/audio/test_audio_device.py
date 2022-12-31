from multiprocessing.sharedctypes import Value
from unittest import TestCase

import numpy as np
from unittest.mock import MagicMock, Mock, patch
from numpy.testing import assert_array_almost_equal

from hermespy.core import ChannelStateInformation, DuplexOperator, Reception, Signal, Transmission
from hermespy.hardware_loop.audio import AudioDevice
from hermespy.hardware_loop.audio.device import AudioDeviceAntennas
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SineOperator(DuplexOperator):
    """Operator transmitting a sine wave for testing purposes."""
    
    __duration: float
    __frequency: float
    
    def __init__(self, duration=3, frequency = -5e3) -> None:
    
        self.__duration = duration
        self.__frequency = frequency
        
        DuplexOperator.__init__(self)
        
    def transmit(self, duration: float = 0.) -> Transmission:
        
        sine = np.exp(2j * np.pi * np.arange(int(self.__duration * self.sampling_rate)) / self.sampling_rate * self.__frequency)
        signal = Signal(sine[np.newaxis, :], self.sampling_rate, self.device.carrier_frequency)
        
        transmission = Transmission(signal=signal)
        
        self.device.transmitters.add_transmission(self, transmission)
        return transmission

    def _receive(self, *args) -> Reception:
        
        reception = Reception(signal=self.signal)
        return reception

    @property
    def sampling_rate(self) -> float:
        
        return self.device.sampling_rate

    @property
    def frame_duration(self) -> float:
        
        return self.__duration

    @property
    def energy(self) -> float:

        return 1.
    
    def noise_power(self, strength: float, snr_type=...) -> float:
        return 0.
    
class TestAudioDeviceAntennas(TestCase):
    
    def setUp(self) -> None:
        
        self.device = Mock()
        self.device.playback_channels = [1, 2, 3, 4, 5]
        
        self.antennas = AudioDeviceAntennas(self.device)
        
    def test_num_antennas(self) -> None:
        """Test numbero of transmit antennas calcualtion."""
        
        self.assertEqual(5, self.antennas.num_antennas)


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
            
    def test_sampling_rate_setget(self) ->  None:
        """Sampling rate property getter should return setter argument"""
        
        sampling_rate = 1.
        self.device.sampling_rate = sampling_rate
        
        self.assertEqual(sampling_rate, self.device.sampling_rate)
        
    def test_sampling_rate_validation(self) -> None:
        """Sampling rate property setter should raise ValueErrors on invalid arguments"""
        
        with self.assertRaises(ValueError):
            self.device.sampling_rate = -1.
            
        with self.assertRaises(ValueError):
            self.device.sampling_rate = 0.
            
    def test_max_sampling_rate(self) -> None:
        """Maximal sampling rate property should return configured sampling rate"""
        
        self.device.sampling_rate = 1.
        self.assertEqual(1., self.device.max_sampling_rate)
            
    @patch('sounddevice.playrec')
    def test_transmit_receive(self, playrec_mock: MagicMock) -> None:
        """Test all device stages."""
        
        def side_effect(*args, **kwargs):
            self.device._AudioDevice__reception = args[0]
            
        playrec_mock.side_effect = side_effect
        
        operator = SineOperator()
        operator.device = self.device
        transmission = operator.transmit()
        
        self.device.transmit()
        self.device.trigger()
        self.device.process_input()
        
        reception = operator.receive()
        
        assert_array_almost_equal(transmission.signal.samples, reception.signal.samples)

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.device, {'antenna_positions'})
