# -*- coding: utf-8 -*-
"""Test prototype for waveform generation modeling."""

import unittest
from unittest.mock import Mock, patch, PropertyMock

import numpy as np
import numpy.random as rnd
from numpy.testing import assert_array_equal
from scipy.constants import pi
from math import floor

from hermespy.channel import ChannelStateFormat, ChannelStateInformation
from hermespy.modem import ChannelEstimation, WaveformGenerator, UniformPilotSymbolSequence, Synchronization, Symbols
from hermespy.core import Signal
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MockWaveformGenerator(WaveformGenerator):
    """Mock communication waveform for modem testing."""

    symbol_rate = 1e4

    @property
    def samples_in_frame(self) -> int:
        
        return self.oversampling_factor * self.symbols_per_frame
    
    @property
    def bits_per_frame(self) -> int:
        
        return self.symbols_per_frame * 1
    
    @property
    def symbols_per_frame(self) -> int:
        
        return 100
    
    @property
    def bit_energy(self) -> float:
    
        return 1.
    
    @property
    def symbol_energy(self) -> float:
        
        return 1.
    
    @property
    def power(self) -> float:
        
        return 1.
    
    def map(self, data_bits: np.ndarray) -> Symbols:
        
        return Symbols(data_bits[np.newaxis, np.newaxis, :])
    
    def unmap(self, symbols: Symbols) -> np.ndarray:
        
        return symbols.raw.real.flatten()
    
    def modulate(self, data_symbols: Symbols) -> Signal:
        
        return Signal(data_symbols.raw.flatten().repeat(self.oversampling_factor), self.sampling_rate)

    def demodulate(self, signal: np.ndarray) -> Symbols:
        
        symbols = Symbols(signal[np.newaxis, np.newaxis, :self.oversampling_factor * self.symbols_per_frame:self.oversampling_factor])
        return symbols
    
    @property
    def bandwidth(self) -> float:
        
        return self.sampling_rate
    
    @property
    def sampling_rate(self) -> float:
        
        return self.symbol_rate * self.oversampling_factor


class TestSynchronization(unittest.TestCase):
    """Test waveform generator synchronization base class."""

    def setUp(self) -> None:

        self.rng = np.random.default_rng(42)
        self.synchronization = Synchronization()
        self.waveform_generator = MockWaveformGenerator()
        self.waveform_generator.synchronization = self.synchronization


    def test_init(self) -> None:
        """Initialization parameters should be properly stored as object attributes"""

        self.assertIs(self.waveform_generator, self.synchronization.waveform_generator)

    def test_waveform_generator_setget(self) -> None:
        """Waveform generator property getter should return setter argument."""

        expected_waveform = Mock()
        self.synchronization.waveform_generator = expected_waveform

        self.assertIs(expected_waveform, self.synchronization.waveform_generator)

        self.synchronization.waveform_generator = None
        self.assertIs(None, self.synchronization.waveform_generator)

    def test_synchronize(self) -> None:
        """Default synchronization should properly split signals into frame-sections."""

        num_streams = 3
        num_frames = 5
        num_offset_samples = 2
        num_samples = num_frames * self.waveform_generator.samples_in_frame + num_offset_samples

        signal = np.exp(2j * self.rng.uniform(0, pi, (num_streams, 1))) @ np.exp(2j * self.rng.uniform(0, pi, (1, num_samples)))
        frames = self.synchronization.synchronize(signal)
        self.assertEqual(num_frames, len(frames))

    def test_serialization(self) -> None:
        """Test YAML serialization"""
        
        with patch('hermespy.modem.waveform_generator.Synchronization.property_blacklist', new_callable=PropertyMock) as blacklist:
        
            blacklist.return_value  = {'waveform_generator',}
            test_yaml_roundtrip_serialization(self, self.synchronization)
        
        
class TestChannelEstimation(unittest.TestCase):
    
    def setUp(self) -> None:
        
        self.waveform = MockWaveformGenerator()
        self.estimation = ChannelEstimation(self.waveform)
        
    def test_init(self) -> None:
        """Initialization should properly set class properties"""
        
        self.assertIs(self.waveform, self.estimation.waveform_generator)

    def test_serialization(self) -> None:
        """Test YAML serialization"""
        
        with patch('hermespy.modem.waveform_generator.ChannelEstimation.property_blacklist', new_callable=PropertyMock) as blacklist:
        
            blacklist.return_value  = {'waveform_generator',}
            test_yaml_roundtrip_serialization(self, self.estimation)

class TestWaveformGenerator(unittest.TestCase):
    """Test the communication waveform generator unit."""

    def setUp(self) -> None:

        self.rnd = rnd.default_rng(42)
        self.modem = Mock()

        self.waveform_generator = MockWaveformGenerator(modem=self.modem)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as attributes."""

        self.assertIs(self.modem, self.waveform_generator.modem)

    def test_oversampling_factor(self) -> None:
        """Oversampling factor property getter should return setter argument."""

        oversampling_factor = 4
        self.waveform_generator.oversampling_factor = oversampling_factor

        self.assertEqual(oversampling_factor, self.waveform_generator.oversampling_factor)

    def test_modulation_order_setget(self) -> None:
        """Modulation order property getter should return setter argument."""

        order = 256
        self.waveform_generator.modulation_order = order

        self.assertEqual(order, self.waveform_generator.modulation_order)

    def test_modulation_order_validation(self) -> None:
        """Modulation order property setter should raise ValueErrors on arguments which aren't powers of two."""

        with self.assertRaises(ValueError):
            self.waveform_generator.modulation_order = 0

        with self.assertRaises(ValueError):
            self.waveform_generator.modulation_order = -1

        with self.assertRaises(ValueError):
            self.waveform_generator.modulation_order = 20

    def test_synchronize(self) -> None:
        """Default synchronization routine should properly split signals into frame-sections."""

        num_streams = 3
        num_samples_test = [50, 100, 150, 200]

        for num_samples in num_samples_test:

            signal = np.exp(2j * self.rnd.uniform(0, pi, (num_streams, 1))) @ np.exp(2j * self.rnd.uniform(0, pi, (1, num_samples)))

            synchronized_frames = self.waveform_generator.synchronization.synchronize(signal)

            # Number of frames is the number of frames that fit into the samples
            num_frames = len(synchronized_frames)
            expected_num_frames = int(floor(num_samples / self.waveform_generator.samples_in_frame))
            self.assertEqual(expected_num_frames, num_frames)

    def test_synchronize_validation(self) -> None:
        """Synchronization should raise a ValueError if the signal shape does match the stream response shape."""

        with self.assertRaises(ValueError):
            _ = self.waveform_generator.synchronization.synchronize(np.zeros(10),
                                                                    ChannelStateInformation(ChannelStateFormat.IMPULSE_RESPONSE,
                                                                    np.zeros((10, 2))))

        with self.assertRaises(ValueError):
            _ = self.waveform_generator.synchronization.synchronize(np.zeros((10, 2)),
                                                                    ChannelStateInformation(ChannelStateFormat.IMPULSE_RESPONSE,
                                                                    np.zeros((10, 2))))

    def test_modem_setget(self) -> None:
        """Modem property getter should return setter argument."""

        modem = Mock()

        self.waveform_generator.modem = modem

        self.assertIs(self.waveform_generator, modem.waveform_generator)
        self.assertIs(self.waveform_generator.modem, modem)


class TestUniformPilotSymbolSequence(unittest.TestCase):
    """Test the uniform pilot symbol sequence."""

    def test_sequence(self) -> None:
        """The generated sequence should be an array containing only the configured symbol."""

        expected_symbol = 1.234 - 1234j
        uniform_sequence = UniformPilotSymbolSequence(expected_symbol)

        assert_array_equal(np.array([expected_symbol], dtype=complex), uniform_sequence.sequence)
