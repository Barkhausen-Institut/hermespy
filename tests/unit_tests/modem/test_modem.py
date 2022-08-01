# -*- coding: utf-8 -*-
"""HermesPy testing for modem base class."""

from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from numpy import random as rnd
from numpy.testing import assert_array_equal, assert_almost_equal

from hermespy.coding import EncoderManager
from hermespy.core import UniformArray, IdealAntenna
from hermespy.modem.modem import Modem, CommunicationReception
from hermespy.precoding import SymbolPrecoding
from hermespy.simulation import SimulatedDevice

from .test_waveform_generator import MockWaveformGenerator

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestCommunicationReception(TestCase):
    """Test the communication reception data class"""

    def setUp(self) -> None:

        self.signal = Mock()

        first_frame = Mock()
        first_frame.encoded_bits = np.zeros(10, dtype=np.uint8)
        first_frame.decoded_bits = np.zeros(10, dtype=np.uint8)

        second_frame = Mock()
        second_frame.encoded_bits = np.zeros(10, dtype=np.uint8)
        second_frame.decoded_bits = np.zeros(10, dtype=np.uint8)

        self.frames = [first_frame, second_frame]

        self.reception = CommunicationReception(self.signal, self.frames)

    def test_num_frames(self) -> None:
        """Number of frames property should return the correct number of frames"""

        self.assertEqual(2, self.reception.num_frames)

    def test_encoded_bits(self) -> None:
        """Encoded bits property should return a concatenation of all encoded bits"""

        self.assertEqual(20, self.reception.encoded_bits.shape[0])

    def test_bits(self) -> None:
        """Bits property should return a concatenation of all decoded bits"""

        self.assertEqual(20, self.reception.bits.shape[0])
    

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
        """Initialization parameters should be properly stored as class attributes"""

        self.assertIs(self.encoding, self.modem.encoder_manager)
        self.assertIs(self.precoding, self.modem.precoding)
        self.assertIs(self.waveform, self.modem.waveform_generator)

    def test_num_streams(self) -> None:
        """Number of streams property should return proper number of streams"""

        self.device.antennas = UniformArray(IdealAntenna(), spacing=1., dimensions=(3,))
        self.assertEqual(3, self.modem.num_streams)

        self.device.antennas = UniformArray(IdealAntenna(), spacing=1., dimensions=(2,))
        self.assertEqual(2, self.modem.num_streams)

    def test_bits_source_setget(self) -> None:
        """Bits source property getter should return setter argument"""

        bits_source = Mock()
        self.modem.bits_source = bits_source

        self.assertIs(bits_source, self.modem.bits_source)

    def test_encoder_manager_setget(self) -> None:
        """Encoder manager property getter should return setter argument"""

        encoder_manager = Mock()
        self.modem.encoder_manager = encoder_manager

        self.assertIs(encoder_manager, self.modem.encoder_manager)
        self.assertIs(encoder_manager.modem, self.modem)

    def test_waveform_generator_setget(self) -> None:
        """Waveform generator property getter should return setter argument"""

        waveform_generator = Mock()
        self.modem.waveform_generator = waveform_generator

        self.assertIs(waveform_generator, self.modem.waveform_generator)
        self.assertIs(waveform_generator.modem, self.modem)
        
    def test_precoding_setget(self) -> None:
        """Precoding configuration property getter should return setter argument"""

        precoding = Mock()
        self.modem.precoding = precoding

        self.assertIs(precoding, self.modem.precoding)
        self.assertIs(precoding.modem, self.modem)

    def test_transmit(self) -> None:
        """Test modem data transmission"""
        
        transmission = self.modem.transmit(2 * self.waveform.frame_duration)
        
        self.assertEqual(0., transmission.signal.carrier_frequency)
        self.assertEqual(2, transmission.num_frames)
        self.assertEqual(2 * self.waveform.samples_in_frame, transmission.signal.num_samples)

    def test_empty_transmit(self) -> None:
        """Test modem data reception over an empty frame"""

        transmission = self.modem.transmit(0.)

        self.assertEqual(0, transmission.signal.num_samples)

    def test_receive(self) -> None:
        """Test modem data reception"""
        
        transmission = self.modem.transmit()
        
        device_signals = self.device.transmit()
        self.device.receive(device_signals)
        
        reception = self.modem.receive()
        
        assert_almost_equal(transmission.bits, reception.bits)
        self.assertIs(transmission, self.modem.transmission)

    def test_empty_receive(self) -> None:
        """Test modem data reception over an empty slot"""

        reception = self.modem.receive()

        self.assertEqual(0, reception.signal.num_samples)
        self.assertIs(reception, self.modem.reception)

    def test_receive_synchronization_fail(self) -> None:
        """A failed synchronization should result in an empty reception"""

        _ = self.modem.transmit()
        self.device.receive(self.device.transmit())

        self.waveform.synchronization.synchronize = lambda s: []
        
        reception = self.modem.receive()
        self.assertEqual(0, reception.num_frames)

    def test_energy(self) -> None:
        """Bit energy property should report a correct energy value"""

        self.assertEqual(1, self.modem.energy)

        self.modem.waveform_generator = None
        self.assertEqual(0, self.modem.energy)
