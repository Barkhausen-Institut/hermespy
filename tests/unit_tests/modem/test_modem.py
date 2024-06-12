# -*- coding: utf-8 -*-
"""HermesPy testing for modem base class."""

from fractions import Fraction
from os import path
from tempfile import TemporaryDirectory
from typing import Type
from unittest import TestCase
from unittest.mock import MagicMock, Mock

import numpy as np
from h5py import File
from numpy import random as rnd
from numpy.testing import assert_array_almost_equal

from hermespy.core import Signal, Device
from hermespy.fec import EncoderManager
from hermespy.modem import Symbols, CommunicationReceptionFrame, CommunicationTransmission, CommunicationTransmissionFrame, CommunicationReception, BaseModem, TransmittingModem, ReceivingModem, DuplexModem, SimplexLink, RandomBitsSource, SymbolPrecoder, SymbolPrecoding
from hermespy.simulation import SimulatedDevice

from .test_waveform import MockCommunicationWaveform

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestCommunicationReception(TestCase):
    """Test the communication reception data class"""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

        self.base_signal = Signal.Create(self.rng.uniform(size=(2, 10)) + 1j * self.rng.uniform(size=(2, 10)), 1)
        self.frames = [
            CommunicationReceptionFrame(
                Signal.Create(self.rng.uniform(size=(2, 10)) + 1j * self.rng.uniform(size=(2, 10)), 1),
                Signal.Create(self.rng.uniform(size=(2, 10)) + 1j * self.rng.uniform(size=(2, 10)), 1),
                Symbols(self.rng.uniform(size=(2, 1, 5)) + 1j * self.rng.uniform(size=(2, 1, 5))),
                Symbols(self.rng.uniform(size=(2, 1, 5)) + 1j * self.rng.uniform(size=(2, 1, 5))),
                1.2345,
                Symbols(self.rng.uniform(size=(2, 1, 5)) + 1j * self.rng.uniform(size=(2, 1, 5))),
                self.rng.integers(0, 2, 20),
                self.rng.integers(0, 2, 10),
            )
            for _ in range(2)
        ]

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

    def test_symbols(self) -> None:
        """Symbols property should return a concatenation of all symbols"""

        symbols = self.reception.symbols
        self.assertEqual(symbols.num_blocks, 2)
        self.assertEqual(symbols.num_streams, 2)
        self.assertEqual(symbols.num_symbols, 5)

    def test_equalized_symbols(self) -> None:
        """Equalized symbols property should return a concatenation of all equalized symbols"""

        symbols = self.reception.equalized_symbols
        self.assertEqual(symbols.num_blocks, 2)
        self.assertEqual(symbols.num_streams, 2)
        self.assertEqual(symbols.num_symbols, 5)

    def test_hdf_serialization(self) -> None:
        """Serialization to and from HDF5 should yield the correct object reconstruction"""

        reception: CommunicationReception = None

        with TemporaryDirectory() as tempdir:
            file_location = path.join(tempdir, "testfile.hdf5")

            with File(file_location, "a") as file:
                group = file.create_group("testgroup")
                self.reception.to_HDF(group)

            with File(file_location, "r") as file:
                group = file["testgroup"]
                reception = self.reception.from_HDF(group)

        np.testing.assert_array_equal(self.base_signal.getitem(), reception.signal.getitem())
        self.assertEqual(2, reception.num_frames)

        for initial_frame, serialized_frame in zip(self.frames, reception.frames):
            np.testing.assert_array_equal(initial_frame.signal.getitem(), serialized_frame.signal.getitem())
            np.testing.assert_array_equal(initial_frame.decoded_signal.getitem(), serialized_frame.decoded_signal.getitem())
            np.testing.assert_array_equal(initial_frame.symbols.raw, serialized_frame.symbols.raw)
            np.testing.assert_array_equal(initial_frame.decoded_symbols.raw, serialized_frame.decoded_symbols.raw)
            self.assertEqual(initial_frame.timestamp, serialized_frame.timestamp)
            np.testing.assert_array_equal(initial_frame.equalized_symbols.raw, serialized_frame.equalized_symbols.raw)
            np.testing.assert_array_equal(initial_frame.encoded_bits, serialized_frame.encoded_bits)
            np.testing.assert_array_equal(initial_frame.decoded_bits, serialized_frame.decoded_bits)


class TestCommunicationTransmission(TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

        self.base_signal = Signal.Create(self.rng.uniform(size=(2, 10)) + 1j * self.rng.uniform(size=(2, 10)), 1)
        self.frames = [CommunicationTransmissionFrame(Signal.Create(self.rng.uniform(size=(2, 10)) + 1j * self.rng.uniform(size=(2, 10)), 1), self.rng.integers(0, 2, 10), self.rng.integers(0, 2, 20), Symbols(self.rng.uniform(size=(2, 1, 5)) + 1j * self.rng.uniform(size=(2, 1, 5))), Symbols(self.rng.uniform(size=(2, 1, 5)) + 1j * self.rng.uniform(size=(2, 1, 5))), 1.2345) for _ in range(2)]

        self.transmission = CommunicationTransmission(self.base_signal, self.frames)

    def test_symbols(self) -> None:
        """Symbols property should return a concatenation of all symbols"""

        symbols = self.transmission.symbols
        self.assertEqual(symbols.num_blocks, 2)
        self.assertEqual(symbols.num_streams, 2)
        self.assertEqual(symbols.num_symbols, 5)

    def test_hdf_serialization(self) -> None:
        """Serialization to and from HDF5 should yield the correct object reconstruction"""

        transmission: CommunicationTransmission = None

        with TemporaryDirectory() as tempdir:
            file_location = path.join(tempdir, "testfile.hdf5")

            with File(file_location, "a") as file:
                group = file.create_group("testgroup")
                self.transmission.to_HDF(group)

            with File(file_location, "r") as file:
                group = file["testgroup"]
                transmission = self.transmission.from_HDF(group)

        np.testing.assert_array_equal(self.base_signal.getitem(), transmission.signal.getitem())
        self.assertEqual(2, transmission.num_frames)

        for initial_frame, serialized_frame in zip(self.frames, transmission.frames):
            np.testing.assert_array_equal(initial_frame.bits, serialized_frame.bits)
            np.testing.assert_array_equal(initial_frame.encoded_bits, serialized_frame.encoded_bits)
            np.testing.assert_array_equal(initial_frame.symbols.raw, serialized_frame.symbols.raw)
            np.testing.assert_array_equal(initial_frame.encoded_symbols.raw, serialized_frame.encoded_symbols.raw)
            np.testing.assert_array_equal(initial_frame.signal.getitem(), serialized_frame.signal.getitem())


class BaseModemMock(BaseModem):
    """Mock class to test abstract base modem."""

    __transmitting_device: Device
    __receiving_device: Device

    def __init__(self, transmitting_device: Device, receiving_device: Device, **kwargs) -> None:
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
        self.waveform = MockCommunicationWaveform(oversampling_factor=4)

        self.modem = modem_type(encoding=self.encoding, precoding=self.precoding, waveform=self.waveform, **kwargs)
        self.modem.random_mother = self.random_node

    def setUp(self) -> None:
        self.transmit_device = SimulatedDevice()
        self.receive_device = SimulatedDevice()

        self._init_base_modem(BaseModemMock, transmitting_device=self.transmit_device, receiving_device=self.receive_device)

    def test_arg_signature(self) -> None:
        """Test base modem serialization argument signature"""

        self.assertCountEqual(["encoding", "precoding", "waveform", "seed"], self.modem._arg_signature())

    def test_initialization(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""

        # Test initialization assignments
        self.assertIs(self.encoding, self.modem.encoder_manager)
        self.assertIs(self.precoding, self.modem.precoding)
        self.assertIs(self.waveform, self.modem.waveform)

        # Test initialization random graph
        self.assertIs(self.modem, self.modem.encoder_manager.random_mother)
        self.assertIs(self.modem, self.modem.waveform.random_mother)

    def test_encoder_manager_setget(self) -> None:
        """Encoder manager property getter should return setter argument"""

        encoder_manager = Mock()
        self.modem.encoder_manager = encoder_manager

        self.assertIs(encoder_manager, self.modem.encoder_manager)
        self.assertIs(encoder_manager.modem, self.modem)

    def test_waveform_setget(self) -> None:
        """Waveform generator property getter should return setter argument"""

        waveform = Mock()
        self.modem.waveform = waveform

        self.assertIs(waveform, self.modem.waveform)
        self.assertIs(waveform.modem, self.modem)

    def test_precoding_setget(self) -> None:
        """Precoding configuration property getter should return setter argument"""

        precoding = Mock()
        self.modem.precoding = precoding

        self.assertIs(precoding, self.modem.precoding)
        self.assertIs(precoding.modem, self.modem)

    def test_bit_requirements_validation(self) -> None:
        """Bit requirements should raise RuntimeError on invalid configurations"""

        precoder = Mock(spec=SymbolPrecoder)
        precoder.rate = Fraction(1, 3)
        self.modem.precoding[0] = precoder

        with self.assertRaises(RuntimeError):
            self.modem._bit_requirements()

    def test_num_data_bits_per_frame(self) -> None:
        """Number of data bits per frame property should return the correct number of bits"""

        self.assertEqual(100, self.modem.num_data_bits_per_frame)

    def test_samples_per_frame(self) -> None:
        """Samples per frame should correctly resolve the waveform's number of samples"""

        self.assertEqual(self.waveform.samples_per_frame, self.modem.samples_per_frame)


class TestTransmittingModem(TestBaseModem):
    """Test the exclusively transmitting simplex modem"""

    def setUp(self) -> None:
        self.bits_source = RandomBitsSource()
        self._init_base_modem(TransmittingModem, bits_source=self.bits_source)

        self.transmit_device = SimulatedDevice()
        self.transmit_device.transmitters.add(self.modem)

    def test_transmitting_device(self) -> None:
        """Transmitting device property should return the correct device"""

        self.assertIs(self.transmit_device, self.modem.transmitting_device)

    def test_receiving_device(self) -> None:
        """Receiving device proeprty should return None"""

        self.assertIsNone(self.modem.receiving_device)

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

    def test_transmit_stream_coding_setget(self) -> None:
        """Transmit stream coding property should return correct configuration"""

        transmit_stream_coding = Mock()
        self.modem.transmit_stream_coding = transmit_stream_coding

        self.assertIs(transmit_stream_coding, self.modem.transmit_stream_coding)
        self.assertIs(transmit_stream_coding.modem, self.modem)

    def test_transmit_validation(self) -> None:
        """Modem transmission should raise ValueError on invalid configurations"""

        precoding = MagicMock(spec=SymbolPrecoding)
        precoding.__len__.side_effect = lambda: 2
        precoding.rate = Fraction(1, 1)
        precoding.num_output_streams = 14
        self.modem.precoding = precoding

        with self.assertRaises(RuntimeError):
            self.modem.transmit()

        stream_coding = MagicMock()
        stream_coding.__len__.side_effect = lambda: 2
        stream_coding.num_output_streams = 14
        self.modem.transmit_stream_coding = stream_coding

        with self.assertRaises(RuntimeError):
            self.modem.transmit()

    def test_transmit(self) -> None:
        """Test modem data transmission"""

        transmission = self.modem.transmit(2 * self.waveform.frame_duration)

        self.assertEqual(0.0, transmission.signal.carrier_frequency)
        self.assertEqual(2, transmission.num_frames)
        self.assertEqual(2 * self.waveform.samples_per_frame, transmission.signal.num_samples)

    def test_empty_transmit(self) -> None:
        """Transmissions not fitting into the waveform duration should return an empty transmission"""

        transmission = self.modem.transmit(0.5 * self.waveform.frame_duration)
        self.assertEqual(0, transmission.num_frames)

    def test_device_setget(self) -> None:
        """Device property getter should return setter argument"""

        expected_device = SimulatedDevice()
        self.modem.device = expected_device

        self.assertIs(expected_device, self.modem.device)

    def test_recall_transmission(self) -> None:
        """Test modem transmission recall from HDF"""

        transmission = self.modem.transmit()

        with TemporaryDirectory() as tempdir:
            file_location = path.join(tempdir, "testfile.hdf5")

            with File(file_location, "w") as file:
                group = file.create_group("testgroup")
                transmission.to_HDF(group)

            with File(file_location, "r") as file:
                recalled_transmission = self.modem._recall_transmission(file["testgroup"])

        self.assertEqual(transmission.signal.num_samples, recalled_transmission.signal.num_samples)


class TestReceivingModem(TestBaseModem):
    """Test the exclusively receiving simplex modem"""

    def setUp(self) -> None:
        self._init_base_modem(ReceivingModem)

        self.receive_device = SimulatedDevice()
        self.receive_device.receivers.add(self.modem)

    def test_receive_stream_coding_setget(self) -> None:
        """Receive stream coding property should return correct configuration"""

        receive_stream_coding = Mock()
        self.modem.receive_stream_coding = receive_stream_coding

        self.assertIs(receive_stream_coding, self.modem.receive_stream_coding)
        self.assertIs(receive_stream_coding.modem, self.modem)

    def test_device_setget(self) -> None:
        """Device property getter should return setter argument"""

        self.modem.device = None

        expected_device = SimulatedDevice()
        self.modem.device = expected_device

        self.assertIs(expected_device, self.modem.device)

        replaced_device = SimulatedDevice()
        self.modem.device = replaced_device

        self.assertEqual(0, expected_device.receivers.num_operators)
        self.assertIs(replaced_device, self.modem.device)

    def test_transmitting_device(self) -> None:
        """Transmitting device property should return None"""

        self.assertIsNone(self.modem.transmitting_device)

    def test_receiving_device(self) -> None:
        """Receiving device property should return the correct device"""

        self.assertIs(self.receive_device, self.modem.receiving_device)

    def test_recall_reception(self) -> None:
        """Test modem reception recall from HDF"""

        transmitting_modem = TransmittingModem()
        transmitting_modem.device = SimulatedDevice()
        transmitting_modem.waveform = MockCommunicationWaveform(oversampling_factor=4)
        transmission = transmitting_modem.transmit()

        reception = self.modem.receive(transmission.signal)

        with TemporaryDirectory() as tempdir:
            file_location = path.join(tempdir, "testfile.hdf5")

            with File(file_location, "w") as file:
                group = file.create_group("testgroup")
                reception.to_HDF(group)

            with File(file_location, "r") as file:
                recalled_reception = self.modem._recall_reception(file["testgroup"])

        self.assertEqual(reception.signal.num_samples, recalled_reception.signal.num_samples)


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

    def test_transmitting_device(self) -> None:
        """Transmitting device property should return the correct device"""

        self.assertIs(self.device, self.modem.transmitting_device)

    def test_receiving_device(self) -> None:
        """Receiving device property should return the correct device"""

        self.assertIs(self.device, self.modem.receiving_device)

    def test_transmit_receive(self) -> None:
        """Test modem data transmission and subsequent reception"""

        device_transmission = self.device.transmit()
        modem_transmission = device_transmission.operator_transmissions[0]

        device_reception = self.device.receive(device_transmission)
        modem_reception = device_reception.operator_receptions[0]

        assert_array_almost_equal(modem_transmission.bits, modem_reception.bits)
        self.assertIs(modem_transmission, self.modem.transmission)
        self.assertIs(modem_reception, self.modem.reception)

    def test_receive_synchronization_fail(self) -> None:
        """A failed synchronization should result in an empty reception"""

        _ = self.modem.transmit()
        self.device.process_input(self.device.transmit())

        self.waveform.synchronization.synchronize = lambda s: []

        reception = self.modem.receive()
        self.assertEqual(0, reception.num_frames)

    def test_receive_synchronization_padding(self) -> None:
        """Received frames should be padded to the correct length"""

        transmission = self.modem.transmit()
        cutoff_samples = transmission.signal.getitem((slice(None, None), slice(None, transmission.signal.num_samples//2)))
        self.waveform.synchronization.synchronize = lambda s: [0]
        self.device.process_input(Signal.Create(cutoff_samples, transmission.signal.sampling_rate, self.device.carrier_frequency))
        reception = self.modem.receive()

        self.assertEqual(transmission.signal.num_samples, reception.frames[0].signal.num_samples)
        return

    def test_device_setget(self) -> None:
        """Device property getter should return setter argument"""

        expected_device = SimulatedDevice()
        self.modem.device = expected_device

        self.assertIs(expected_device, self.modem.device)
        self.assertIs(expected_device, TransmittingModem.device.fget(self.modem))
        self.assertIs(expected_device, ReceivingModem.device.fget(self.modem))
        self.assertTrue(self.modem in expected_device.transmitters)
        self.assertTrue(self.modem in expected_device.receivers)

        new_device = SimulatedDevice()
        self.modem.device = new_device


class TestSimplexLink(TestCase):
    def setUp(self) -> None:
        self.transmitter = SimulatedDevice()
        self.receiver = SimulatedDevice()

        self.link = SimplexLink(self.transmitter, self.receiver)

    def test_reference_validation(self) -> None:
        """Specifying the reference of a simplex link is not supported"""

        with self.assertRaises(RuntimeError):
            self.link.reference = Mock()

    def test_reference(self) -> None:
        """Reference should always be the transmitting device"""

        self.assertIs(self.transmitter, self.link.reference)

    def test_transmitting_device(self) -> None:
        """Transmitting device property should return the correct device"""

        self.assertIs(self.transmitter, self.link.transmitting_device)

    def test_receiving_device(self) -> None:
        """Receiving device property should return the correct device"""

        self.assertIs(self.receiver, self.link.receiving_device)
