# -*- coding: utf-8 -*-
"""HermesPy testing for modem base class."""

from os import path
from tempfile import TemporaryDirectory
from typing import Type
from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from h5py import File
from numpy import random as rnd
from numpy.testing import assert_array_almost_equal

from hermespy.core import UniformArray, IdealAntenna, Signal, Device, ChannelStateFormat, ChannelStateInformation, SNRType
from hermespy.fec import EncoderManager
from hermespy.modem import Symbols, CommunicationReceptionFrame, CommunicationTransmission, CommunicationTransmissionFrame, CommunicationReception, \
                           BaseModem, TransmittingModem, ReceivingModem, DuplexModem, SimplexLink, RandomBitsSource
from hermespy.precoding import SymbolPrecoding
from hermespy.simulation import SimulatedDevice

from .test_waveform_generator import MockWaveformGenerator

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestCommunicationReception(TestCase):
    """Test the communication reception data class"""

    def setUp(self) -> None:

        self.rng = np.random.default_rng(42)
        
        self.base_signal = Signal(self.rng.uniform(size=(2,10)) + 1j * self.rng.uniform(size=(2,10)), 1)
        self.frames = [CommunicationReceptionFrame(Signal(self.rng.uniform(size=(2,10)) + 1j * self.rng.uniform(size=(2,10)), 1),
                                                   Signal(self.rng.uniform(size=(2,10)) + 1j * self.rng.uniform(size=(2,10)), 1),
                                                   Symbols(self.rng.uniform(size=(2,5)) + 1j * self.rng.uniform(size=(2,5))),
                                                   Symbols(self.rng.uniform(size=(2,5)) + 1j * self.rng.uniform(size=(2,5))),
                                                   1.2345,
                                                   Symbols(self.rng.uniform(size=(2,5)) + 1j * self.rng.uniform(size=(2,5))),
                                                   self.rng.integers(0, 2, 20), self.rng.integers(0, 2, 10),
                                                   ChannelStateInformation(ChannelStateFormat.IMPULSE_RESPONSE,
                                                                           self.rng.uniform(size=(2,1,10,2)) + 1j * self.rng.uniform(size=(2,1,10,2)))) for _ in range(2)]
        
        self.reception = CommunicationReception(self.base_signal, self.frames)

    def test_num_frames(self) -> None:
        """Number of frames property should return the correct number of frames"""

        self.assertEqual(2, self.reception.num_frames)

    def test_encoded_bits(self) -> None:
        """Encoded bits property should return a concatenation of all encoded bits"""

        self.assertEqual(40, self.reception.encoded_bits.shape[0])

    def test_bits(self) -> None:
        """Bits property should return a concatenation of all decoded bits"""

        self.assertEqual(20, self.reception.bits.shape[0])

    def test_hdf_serialization(self) -> None:
        """Serialization to and from HDF5 should yield the correct object reconstruction"""
        
        reception: CommunicationReception = None
        
        with TemporaryDirectory() as tempdir:
            
            file_location = path.join(tempdir, 'testfile.hdf5')
            
            with File(file_location, 'a') as file:
                
                group = file.create_group('testgroup')
                self.reception.to_HDF(group)
                
            with File(file_location, 'r') as file:
                
                group = file['testgroup']
                reception = self.reception.from_HDF(group)
                
        np.testing.assert_array_equal(self.base_signal.samples, reception.signal.samples)
        self.assertEqual(2, reception.num_frames)
        
        for initial_frame, serialized_frame in zip(self.frames, reception.frames):
            
            np.testing.assert_array_equal(initial_frame.signal.samples, serialized_frame.signal.samples)
            np.testing.assert_array_equal(initial_frame.decoded_signal.samples, serialized_frame.decoded_signal.samples)
            np.testing.assert_array_equal(initial_frame.symbols.raw, serialized_frame.symbols.raw)
            np.testing.assert_array_equal(initial_frame.decoded_symbols.raw, serialized_frame.decoded_symbols.raw)
            self.assertEqual(initial_frame.timestamp, serialized_frame.timestamp)
            np.testing.assert_array_equal(initial_frame.equalized_symbols.raw, serialized_frame.equalized_symbols.raw)
            np.testing.assert_array_equal(initial_frame.encoded_bits, serialized_frame.encoded_bits)
            np.testing.assert_array_equal(initial_frame.decoded_bits, serialized_frame.decoded_bits)
            np.testing.assert_array_equal(initial_frame.csi.state, serialized_frame.csi.state)
        
        
class TestCommunicationTransmission(TestCase):
    
    def setUp(self) -> None:
        
        self.rng = np.random.default_rng(42)
        
        self.base_signal = Signal(self.rng.uniform(size=(2,10)) + 1j * self.rng.uniform(size=(2,10)), 1)
        self.frames = [CommunicationTransmissionFrame(Signal(self.rng.uniform(size=(2,10)) + 1j * self.rng.uniform(size=(2,10)), 1),
                                                      self.rng.integers(0, 2, 10), self.rng.integers(0, 2, 20),
                                                      Symbols(self.rng.uniform(size=(2,5)) + 1j * self.rng.uniform(size=(2,5))),
                                                      Symbols(self.rng.uniform(size=(2,5)) + 1j * self.rng.uniform(size=(2,5))),
                                                      1.2345) for _ in range(2)]
        
        self.transmission = CommunicationTransmission(self.base_signal, self.frames)
        
    def test_hdf_serialization(self) -> None:
        """Serialization to and from HDF5 should yield the correct object reconstruction"""
        
        transmission: CommunicationTransmission = None
        
        with TemporaryDirectory() as tempdir:
            
            file_location = path.join(tempdir, 'testfile.hdf5')
            
            with File(file_location, 'a') as file:
                
                group = file.create_group('testgroup')
                self.transmission.to_HDF(group)
                
            with File(file_location, 'r') as file:
                
                group = file['testgroup']
                transmission = self.transmission.from_HDF(group)
                
        np.testing.assert_array_equal(self.base_signal.samples, transmission.signal.samples)
        self.assertEqual(2, transmission.num_frames)
        
        for initial_frame, serialized_frame in zip(self.frames, transmission.frames):
            
            np.testing.assert_array_equal(initial_frame.bits, serialized_frame.bits)
            np.testing.assert_array_equal(initial_frame.encoded_bits, serialized_frame.encoded_bits)
            np.testing.assert_array_equal(initial_frame.symbols.raw, serialized_frame.symbols.raw)
            np.testing.assert_array_equal(initial_frame.encoded_symbols.raw, serialized_frame.encoded_symbols.raw)
            np.testing.assert_array_equal(initial_frame.signal.samples, serialized_frame.signal.samples)


class BaseModemMock(BaseModem):
    """Mock class to test abstract base modem."""
    
    __transmitting_device: Device
    __receiving_device: Device
    
    def __init__(self,
                 transmitting_device: Device,
                 receiving_device: Device,
                 **kwargs) -> None:
        
        self.__transmitting_device = transmitting_device
        self.__receiving_device = receiving_device
        
        BaseModem.__init__(self, **kwargs)
        
    @property
    def transmitting_device(self) -> Device:
        
        return self.__transmitting_device
    
    @property
    def receiving_device(self) -> Device:
        
        return self.__receiving_device


class TestBaseModem(TestCase):
    """Test modem base class"""

    def _init_base_modem(self, modem_type: Type[BaseModem], **kwargs) -> None:

        self.random_generator = rnd.default_rng(42)
        self.random_node = Mock()
        self.random_node._rng = self.random_generator

        self.encoding = EncoderManager()
        self.precoding = SymbolPrecoding()
        self.waveform = MockWaveformGenerator(oversampling_factor=4)

        self.modem = modem_type(encoding=self.encoding, precoding=self.precoding, waveform=self.waveform, **kwargs)
        self.modem.random_mother = self.random_node

    def setUp(self) -> None:

        self.transmit_device = SimulatedDevice()
        self.receive_device = SimulatedDevice()
        
        self._init_base_modem(BaseModemMock, transmitting_device=self.transmit_device, receiving_device=self.receive_device)

    def test_initialization(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""

        # Test initialization assignments
        self.assertIs(self.encoding, self.modem.encoder_manager)
        self.assertIs(self.precoding, self.modem.precoding)
        self.assertIs(self.waveform, self.modem.waveform_generator)

        # Test initialization random graph
        self.assertIs(self.modem, self.modem.encoder_manager.random_mother)
        self.assertIs(self.modem, self.modem.waveform_generator.random_mother)

    def test_num_transmit_streams(self) -> None:
        """Number of transmit streams property should return proper number of streams"""

        self.transmit_device.antennas = UniformArray(IdealAntenna(), spacing=1., dimensions=(3,))
        self.assertEqual(3, self.modem.num_transmit_streams)

        self.transmit_device.antennas = UniformArray(IdealAntenna(), spacing=1., dimensions=(2,))
        self.assertEqual(2, self.modem.num_transmit_streams)
        
    def test_num_receive_streams(self) -> None:
        """Number of receive streams property should return proper number of streams"""

        self.receive_device.antennas = UniformArray(IdealAntenna(), spacing=1., dimensions=(3,))
        self.assertEqual(3, self.modem.num_receive_streams)

        self.receive_device.antennas = UniformArray(IdealAntenna(), spacing=1., dimensions=(2,))
        self.assertEqual(2, self.modem.num_receive_streams)

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

    def test_noise_power(self) -> None:
        """Noise power estiamtor should report the correct noise powers"""

        self.assertEqual(1., self.modem.noise_power(1., SNRType.EBN0))
        self.assertEqual(1., self.modem.noise_power(1., SNRType.ESN0))
        self.assertEqual(1., self.modem.noise_power(1., SNRType.PN0))
        
        with self.assertRaises(ValueError):
            _ = self.modem.noise_power(1., SNRType.EN0)


class TestTransmittingModem(TestBaseModem):
    """Test the exclusively transmitting simplex modem"""

    def setUp(self) -> None:

        self.bits_source = RandomBitsSource()
        self._init_base_modem(TransmittingModem, bits_source=self.bits_source)
        
        self.transmit_device = SimulatedDevice()
        self.transmit_device.transmitters.add(self.modem)
        
    def test_num_receive_streams(self) -> None:
        
        self.assertEqual(0, self.modem.num_receive_streams)
        
    def test_bits_source_setget(self) -> None:
        """Bits source property getter should return setter argument"""

        bits_source = Mock()
        self.modem.bits_source = bits_source

        self.assertIs(bits_source, self.modem.bits_source)

    def test_transmission(self) -> None:
        """Transmission property should return the recently transmitted information"""

        # Initially None should be returned
        self.assertIsNone(self.modem.transmission)

        expected_transmission = self.modem.transmit()
        self.assertIs(expected_transmission, self.modem.transmission)

    def test_transmit_stream_coding(self) -> None:
        """Transmit stream coding property should return correct configuration"""

        self.assertIs(self.modem, self.modem.transmit_stream_coding.modem)

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


class TestReceivingModem(TestBaseModem):
    """Test the exclusively receiving simplex modem"""

    def setUp(self) -> None:

        self._init_base_modem(ReceivingModem)
        
        self.receive_device = SimulatedDevice()
        self.receive_device.receivers.add(self.modem)
        
    def test_num_transmit_streams(self) -> None:
        
        self.assertEqual(0, self.modem.num_transmit_streams)

    def test_receive_stream_coding(self) -> None:
        """Receive stream coding property should return correct configuration"""

        self.assertIs(self.modem, self.modem.receive_stream_coding.modem)


class TestDuplexModem(TestBaseModem):
    """Test the simultaneously transmitting and receiving duplex modem"""

    def setUp(self) -> None:

        self.bits_source = RandomBitsSource()
        
        self._init_base_modem(DuplexModem, bits_source=self.bits_source)
        
        self.device = SimulatedDevice()
        self.transmit_device = self.device
        self.receive_device = self.device
        self.device.transmitters.add(self.modem)
        self.device.receivers.add(self.modem)

    def test_transmit_receive(self) -> None:
        """Test modem data transmission and subsequent reception"""
        
        operator_transmission = self.modem.transmit()
        
        device_transmission = self.device.transmit()
        self.device.receive(device_transmission)
        
        reception = self.modem.receive()
        
        assert_array_almost_equal(operator_transmission.bits, reception.bits)
        self.assertIs(operator_transmission, self.modem.transmission)

    def test_receive_synchronization_fail(self) -> None:
        """A failed synchronization should result in an empty reception"""

        _ = self.modem.transmit()
        self.device.receive(self.device.transmit())

        self.waveform.synchronization.synchronize = lambda s: []
        
        reception = self.modem.receive()
        self.assertEqual(0, reception.num_frames)
                                                                                                                                                                           
    def test_receive(self) -> None:
        """Test modem data reception"""
        
        transmission = self.modem.transmit()
        
        device_signals = self.device.transmit()
        self.device.receive(device_signals)
        
        reception = self.modem.receive()
        
        assert_array_almost_equal(transmission.bits, reception.bits)
        self.assertIs(transmission, self.modem.transmission)

    def test_receive_synchronization_fail(self) -> None:
        """A failed synchronization should result in an empty reception"""

        _ = self.modem.transmit()
        self.device.receive(self.device.transmit())

        self.waveform.synchronization.synchronize = lambda s: []
        
        reception = self.modem.receive()
        self.assertEqual(0, reception.num_frames)


class TestSimplexLink(TestCase):

    def setUp(self) -> None:

        self.transmitter = SimulatedDevice()
        self.receiver = SimulatedDevice()

        self.link = SimplexLink(self.transmitter, self.receiver)

    def test_transmitting_device(self) -> None:
        """Transmitting device property should return the correct device"""

        self.assertIs(self.transmitter, self.link.transmitting_device)

    def test_receiving_device(self) -> None:
        """Receiving device property should return the correct device"""

        self.assertIs(self.receiver, self.link.receiving_device)
