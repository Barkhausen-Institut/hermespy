# -*- coding: utf-8 -*-
"""HermesPy testing for modem base class."""

from typing import Type
from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from numpy import random as rnd
from numpy.testing import assert_array_almost_equal

from hermespy.core import Signal
from hermespy.fec import EncoderManager
from hermespy.modem import Symbols, CommunicationReceptionFrame, CommunicationTransmission, CommunicationTransmissionFrame, CommunicationReception, BaseModem, TransmittingModem, ReceivingModem, DuplexModem, SimplexLink, RandomBitsSource
from hermespy.simulation import SimulatedDevice

from .test_waveform import MockCommunicationWaveform
from unit_tests.core.test_factory import test_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
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
                1,
                self.rng.integers(0, 2, 10),
                1,
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

    def test_serialization(self) -> None:
        """Test communication reception serialization"""

        test_roundtrip_serialization(self, self.reception)


class TestCommunicationTransmission(TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

        self.base_signal = Signal.Create(self.rng.uniform(size=(2, 10)) + 1j * self.rng.uniform(size=(2, 10)), 1)
        self.frames = [
            CommunicationTransmissionFrame(
                Signal.Create(self.rng.uniform(size=(2, 10)) + 1j * self.rng.uniform(size=(2, 10)), 1),
                self.rng.integers(0, 2, 10),
                1,
                self.rng.integers(0, 2, 20),
                1,
                Symbols(self.rng.uniform(size=(2, 1, 5)) + 1j * self.rng.uniform(size=(2, 1, 5))),
                Symbols(self.rng.uniform(size=(2, 1, 5)) + 1j * self.rng.uniform(size=(2, 1, 5))),
                1.2345,
            )
            for _ in range(2)]

        self.transmission = CommunicationTransmission(self.base_signal, self.frames)

    def test_symbols(self) -> None:
        """Symbols property should return a concatenation of all symbols"""

        symbols = self.transmission.symbols
        self.assertEqual(symbols.num_blocks, 2)
        self.assertEqual(symbols.num_streams, 2)
        self.assertEqual(symbols.num_symbols, 5)

    def test_serialization(self) -> None:
        """Test communication transmission serialization"""

        test_roundtrip_serialization(self, self.transmission)


class BaseModemMock(BaseModem):
    """Mock class to test abstract base modem."""

    @property
    def num_data_bits_per_frame(self) -> int:
        return 100


class TestBaseModem(TestCase):
    """Test modem base class"""

    def _init_base_modem(self, modem_type: Type[BaseModem], **kwargs) -> None:
        self.random_generator = rnd.default_rng(42)
        self.random_node = Mock()
        self.random_node._rng = self.random_generator

        self.encoding = EncoderManager()
        self.waveform = MockCommunicationWaveform(oversampling_factor=4)

        self.modem = modem_type(encoding=self.encoding, waveform=self.waveform, **kwargs)
        self.modem.random_mother = self.random_node

    def setUp(self) -> None:
        self._init_base_modem(BaseModemMock)

    def test_initialization(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""

        # Test initialization assignments
        self.assertIs(self.encoding, self.modem.encoder_manager)
        self.assertIs(self.waveform, self.modem.waveform)

        # Test initialization random graph
        self.assertIs(self.modem, self.modem.encoder_manager.random_mother)
        self.assertIs(self.modem, self.modem.waveform.random_mother)

    def test_encoder_manager_setget(self) -> None:
        """Encoder manager property getter should return setter argument"""

        encoder_manager = Mock()
        self.modem.encoder_manager = encoder_manager

        self.assertIs(encoder_manager, self.modem.encoder_manager)

    def test_waveform_setget(self) -> None:
        """Waveform generator property getter should return setter argument"""

        waveform = Mock()
        self.modem.waveform = waveform

        self.assertIs(waveform, self.modem.waveform)
        self.assertIs(waveform.modem, self.modem)

    def test_samples_per_frame(self) -> None:
        """Samples per frame should correctly resolve the waveform's number of samples"""

        self.assertEqual(self.waveform.samples_per_frame, self.modem.samples_per_frame)

    def test_serialization(self) -> None:
        """Test base modem serialzation"""

        self.modem.waveform = None  # Required because the configured mock waveform is not serializatble
        test_roundtrip_serialization(self, self.modem, {'random_mother'})


class TestTransmittingModem(TestBaseModem):
    """Test the exclusively transmitting simplex modem"""

    modem: TransmittingModem

    def setUp(self) -> None:
        self.bits_source = RandomBitsSource()
        self._init_base_modem(TransmittingModem, bits_source=self.bits_source)

        self.transmit_device = SimulatedDevice()
        self.transmit_device.transmitters.add(self.modem)

    def test_bits_source_setget(self) -> None:
        """Bits source property getter should return setter argument"""

        bits_source = Mock()
        self.modem.bits_source = bits_source

        self.assertIs(bits_source, self.modem.bits_source)

    def test_transmit_signal_coding_setget(self) -> None:
        """Transmit signal coding property should return correct configuration"""

        transmit_signal_coding = Mock()
        self.modem.transmit_signal_coding = transmit_signal_coding

        self.assertIs(transmit_signal_coding, self.modem.transmit_signal_coding)

    def test_transmit_symbol_coding_setget(self) -> None:
        """Transmit symbol coding property should return correct configuration"""

        transmit_symbol_coding = Mock()
        self.modem.transmit_symbol_coding = transmit_symbol_coding

        self.assertIs(transmit_symbol_coding, self.modem.transmit_symbol_coding)

    def test_transmit(self) -> None:
        """Test modem data transmission"""

        transmission = self.modem.transmit(self.transmit_device.state(), duration=2 * self.waveform.frame_duration)

        self.assertEqual(0.0, transmission.signal.carrier_frequency)
        self.assertEqual(2, transmission.num_frames)
        self.assertEqual(2 * self.waveform.samples_per_frame, transmission.signal.num_samples)

    def test_empty_transmit(self) -> None:
        """Transmissions not fitting into the waveform duration should return an empty transmission"""

        transmission = self.modem.transmit(self.transmit_device.state(), duration=0.5 * self.waveform.frame_duration)
        self.assertEqual(0, transmission.num_frames)

    def test_device_setget(self) -> None:
        """Device property getter should return setter argument"""

        expected_device = SimulatedDevice()
        self.modem.device = expected_device

        self.assertIs(expected_device, self.modem.device)


class TestReceivingModem(TestBaseModem):
    """Test the exclusively receiving simplex modem"""

    modem: ReceivingModem

    def setUp(self) -> None:
        self._init_base_modem(ReceivingModem)

        self.transmit_device = SimulatedDevice()
        self.receive_device = SimulatedDevice()
        self.receive_device.receivers.add(self.modem)

    def test_receive_signal_coding_setget(self) -> None:
        """Receive signal coding property should return correct configuration"""

        receive_signal_coding = Mock()
        self.modem.receive_signal_coding = receive_signal_coding

        self.assertIs(receive_signal_coding, self.modem.receive_signal_coding)

    def test_receive_symbol_coding_setget(self) -> None:
        """Receive symbol coding property should return correct configuration"""

        receive_symbol_coding = Mock()
        self.modem.receive_symbol_coding = receive_symbol_coding

        self.assertIs(receive_symbol_coding, self.modem.receive_symbol_coding)


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

        device_transmission = self.device.transmit()
        modem_transmission = device_transmission.operator_transmissions[0]

        device_reception = self.device.receive(device_transmission)
        modem_reception = device_reception.operator_receptions[0]

        assert_array_almost_equal(modem_transmission.bits, modem_reception.bits)

    def test_receive_synchronization_fail(self) -> None:
        """A failed synchronization should result in an empty reception"""

        transmission = self.modem.transmit(self.transmit_device.state())

        self.waveform.synchronization.synchronize = lambda s: []
        reception = self.modem.receive(transmission.signal, self.receive_device.state())

        self.assertEqual(0, reception.num_frames)

    def test_receive_synchronization_padding(self) -> None:
        """Received frames should be padded to the correct length"""

        transmission = self.modem.transmit(self.device.state())
        cutoff_samples = transmission.signal.getitem((slice(None, None), slice(None, transmission.signal.num_samples//2)))
        self.waveform.synchronization.synchronize = lambda s: [0]
        processed_input = self.device.process_input(Signal.Create(cutoff_samples, transmission.signal.sampling_rate, self.device.carrier_frequency))
        reception = self.modem.receive(processed_input.operator_inputs[0], self.device.state())

        self.assertEqual(transmission.signal.num_samples, reception.frames[0].signal.num_samples)


class TestSimplexLink(TestCase):
    def setUp(self) -> None:
        self.transmitter = SimulatedDevice()
        self.receiver = SimulatedDevice()

        self.link = SimplexLink()
        self.transmitter.transmitters.add(self.link)
        self.receiver.receivers.add(self.link)

    def test_serialization(self) -> None:
        """Test simplex link serialization"""

        test_roundtrip_serialization(self, self.link)
