# -*- coding: utf-8 -*-
"""Test Precoding configuration"""

import unittest
from unittest.mock import Mock, patch, PropertyMock
from fractions import Fraction

import numpy as np
from hermespy.core import Signal

from hermespy.precoding import TransmitStreamEncoder, ReceiveStreamDecoder, TransmitStreamCoding, ReceiveStreamCoding
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TransmitStreamEncoderMock(TransmitStreamEncoder):
    """Mock class for testing"""
    
    def encode_streams(self, streams: Signal) -> Signal:
        return streams
    
    @property
    def num_transmit_input_streams(self) -> int:
        return 1
    
    @property
    def num_transmit_output_streams(self) -> int:
        return 1
    

class ReceiveStreamDecoderMock(ReceiveStreamDecoder):
    """Mock class for testing"""
    
    def decode_streams(self, streams: Signal) -> Signal:
        return streams
    
    @property
    def num_receive_input_streams(self) -> int:
        return 1
    
    @property
    def num_receive_output_streams(self) -> int:
        return 1
    
    
class TestTransmitStreamEncoder(unittest.TestCase):
    """Test transmit stream encoder base class"""
    
    def setUp(self) -> None:
        
        self.encoder = TransmitStreamEncoderMock()
        
    def test_properties(self) -> None:
        """Test properties"""
        
        self.assertEqual(1, self.encoder.num_transmit_input_streams)
        self.assertEqual(1, self.encoder.num_transmit_output_streams)
        self.assertEqual(1, self.encoder.num_input_streams)
        self.assertEqual(1, self.encoder.num_output_streams)
        

class TestReceiveStreamDecoder(unittest.TestCase):
    """Test receive stream decoder base class"""
    
    def setUp(self) -> None:
        
        self.decoder = ReceiveStreamDecoderMock()
        
    def test_properties(self) -> None:
        """Test properties"""
        
        self.assertEqual(1, self.decoder.num_receive_input_streams)
        self.assertEqual(1, self.decoder.num_receive_output_streams)
        self.assertEqual(1, self.decoder.num_input_streams)
        self.assertEqual(1, self.decoder.num_output_streams)


class TestTransmitStreamCoding(unittest.TestCase):
    """Test transmit stream coding base class"""

    def setUp(self) -> None:
        
        self.coding = TransmitStreamCoding()
        
    def test_encode_streams(self) -> None:
        """Encoding should be delegated to the registeded precoders"""
        
        encoder = Mock(spec=TransmitStreamEncoder)
        self.coding[0] = encoder
        signal = Mock()
        
        self.coding.encode(signal)
        
        encoder.encode_streams.assert_called_once_with(signal.copy())

    def test_yaml_roundtrip_serialization(self) -> None:
        """Test serialization roundtrip"""
        
        test_yaml_roundtrip_serialization(self, self.coding)


class TestReceiveStreamCoding(unittest.TestCase):
    """Test receive stream coding base class"""

    def setUp(self) -> None:
        
        self.coding = ReceiveStreamCoding()
        
    def test_decode_streams(self) -> None:
        """Decoding should be delegated to the registeded precoders"""
        
        decoder = Mock(spec=ReceiveStreamDecoder)
        self.coding[0] = decoder
        signal = Mock()
        
        self.coding.decode(signal)
        
        decoder.decode_streams.assert_called_once_with(signal.copy())

    def test_yaml_roundtrip_serialization(self) -> None:
        """Test serialization roundtrip"""
        
        test_yaml_roundtrip_serialization(self, self.coding)
