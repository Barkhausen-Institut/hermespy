# -*- coding: utf-8 -*-
"""HermesPy testing for modem base class."""

from typing import Tuple
from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from numpy import random as rnd
from numpy.testing import assert_array_equal, assert_almost_equal

from hermespy.coding import EncoderManager
from hermespy.core.channel_state_information import ChannelStateInformation
from hermespy.core.signal_model import Signal
from hermespy.modem.modem import Modem
from hermespy.modem.symbols import Symbols
from hermespy.modem.waveform_generator import WaveformGenerator
from hermespy.precoding import SymbolPrecoding
from hermespy.simulation import SimulatedDevice
from hermespy.simulation.antenna import UniformArray, IdealAntenna

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "0.2.0"
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
        
        return self.symbols_per_frame * self.bits_per_symbol
    
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
    
    @property
    def modulation_order(self) -> int:
        
        return 2
    
    @modulation_order.setter
    def modulation_order(self, *_) -> None:
        ...
    
    def map(self, data_bits: np.ndarray) -> Symbols:
        
        return Symbols(data_bits)
    
    def unmap(self, symbols: Symbols) -> np.ndarray:
        
        return symbols.raw.real.flatten()
    
    def modulate(self, data_symbols: Symbols) -> Signal:
        
        return Signal(data_symbols.raw.repeat(self.oversampling_factor, axis=1), self.sampling_rate)

    def demodulate(self, signal: np.ndarray, channel_state: ChannelStateInformation, noise_variance: float) -> Tuple[Symbols, ChannelStateInformation, np.ndarray]:
        
        symbols = Symbols(signal[:self.oversampling_factor*self.symbols_per_frame:self.oversampling_factor])
        channel_state = channel_state[:, :, :self.oversampling_factor*self.symbols_per_frame:self.oversampling_factor, :]
        noise_variances = noise_variance * np.ones(self.oversampling_factor*self.symbols_per_frame, dtype=float)

        return symbols, channel_state, noise_variances
    
    @property
    def bandwidth(self) -> float:
        
        return self.sampling_rate
    
    @property
    def sampling_rate(self) -> float:
        
        return self.symbol_rate * self.oversampling_factor
    

class TestModem(TestCase):
    """Modem Base Class Test Case"""

    def setUp(self) -> None:

        self.random_generator = rnd.default_rng(42)
        self.random_node = Mock()
        self.random_node._rng = self.random_generator

        self.device = SimulatedDevice()

        self.encoding = EncoderManager()
        self.precoding = SymbolPrecoding()
        self.waveform = MockWaveformGenerator(oversampling_factor=4)

        self.modem = Modem(encoding=self.encoding, precoding=self.precoding, waveform=self.waveform)
        self.modem.device = self.device
        self.modem.random_mother = self.random_node

    def test_initialization(self) -> None:
        """Initialization parameters should be properly stored as class attributes."""

        self.assertIs(self.encoding, self.modem.encoder_manager)
        self.assertIs(self.precoding, self.modem.precoding)
        self.assertIs(self.waveform, self.modem.waveform_generator)

    def test_num_streams(self) -> None:
        """Number of streams property should return proper number of streams."""

        self.device.antennas = UniformArray(IdealAntenna(), spacing=1., dimensions=(3,))
        self.assertEqual(3, self.modem.num_streams)

        self.device.antennas = UniformArray(IdealAntenna(), spacing=1., dimensions=(2,))
        self.assertEqual(2, self.modem.num_streams)

    def test_bits_source_setget(self) -> None:
        """Bits source property getter should return setter argument."""

        bits_source = Mock()
        self.modem.bits_source = bits_source

        self.assertIs(bits_source, self.modem.bits_source)

    def test_encoder_manager_setget(self) -> None:
        """Encoder manager property getter should return setter argument."""

        encoder_manager = Mock()
        self.modem.encoder_manager = encoder_manager

        self.assertIs(encoder_manager, self.modem.encoder_manager)
        self.assertIs(encoder_manager.modem, self.modem)

    def test_waveform_generator_setget(self) -> None:
        """Waveform generator property getter should return setter argument."""

        waveform_generator = Mock()
        self.modem.waveform_generator = waveform_generator

        self.assertIs(waveform_generator, self.modem.waveform_generator)
        self.assertIs(waveform_generator.modem, self.modem)
        
    def test_precoding_setget(self) -> None:
        """Precoding configuration property getter should return setter argument."""

        precoding = Mock()
        self.modem.precoding = precoding

        self.assertIs(precoding, self.modem.precoding)
        self.assertIs(precoding.modem, self.modem)

    def test_transmit(self) -> None:
        """Test modem data transmission."""
        
        signal, symbols, bits = self.modem.transmit()
        
        self.assertEqual(0., signal.carrier_frequency)
        self.assertEqual(self.waveform.samples_in_frame, signal.num_samples)
        self.assertEqual(self.waveform.symbols_per_frame, symbols.num_symbols)
        self.assertEqual(self.waveform.bits_per_frame, len(bits))

    def test_receive(self) -> None:
        """Test modem data reception."""
        
        _, tx_symbols, tx_bits = self.modem.transmit()
        
        device_signals = self.device.transmit()
        self.device.receive(device_signals)
        
        _, rx_symbols, rx_bits = self.modem.receive()
        
        assert_almost_equal(tx_symbols.raw, rx_symbols.raw)
        assert_almost_equal(tx_bits, rx_bits)
