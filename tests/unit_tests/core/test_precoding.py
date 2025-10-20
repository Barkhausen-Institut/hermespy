# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from hermespy.core import DeviceState, Signal
from hermespy.core.precoding import ReceiveStreamDecoder, ReceiveSignalCoding, TransmitSignalCoding, TransmitStreamEncoder
from hermespy.simulation import SimulatedDevice, SimulatedUniformArray, SimulatedIdealAntenna
from unit_tests.core.test_factory import test_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TransmitStreamEncoderMock(TransmitStreamEncoder):
    """Mock transmit stream encoder for unit test purposes.

    Copies a single input stream to generate the required number of output streams.
    """

    def encode_streams(self, streams: Signal, num_output_streams: int, device: DeviceState) -> Signal:
        # Repeat the number of streams
        tiled_streams = np.tile(streams[[0], :], (num_output_streams, 1))
        return tiled_streams

    def num_transmit_input_streams(self, num_output_streams: int) -> int:
        return 1


class ReceiveStreamDecoderMock(ReceiveStreamDecoder):
    """Mock receive stream decoder for unit test purposes.

    Selects a single output stream from all input streams.
    """

    def decode_streams(self, streams: Signal, num_output_streams: int, device: DeviceState) -> Signal:
        return streams[slice(0, streams.num_streams, 2), :]

    def num_receive_output_streams(self, num_input_streams: int) -> int:
        return num_input_streams // 2


class _TestSignalCoding():
    """Base class of signal coding tests."""

    coding: TransmitSignalCoding | ReceiveSignalCoding

    def test_yaml_serialization(self) -> None:
        """Test serialization to and from YAML"""

        test_roundtrip_serialization(self, self.coding)

    def test_pop_precoder(self) -> None:
        """Precoders should be properly removed from the configuration"""

        self.coding[0] = Mock()
        self.coding[1] = Mock()

        self.coding.pop_precoder(0)
        self.assertEqual(1, len(self.coding))


class TestTransmitSignalCoding(_TestSignalCoding, TestCase):
    """Test the transmit signal coding class.

    Also functions as a test for the TransmitPrecoding and Precoding base classes.
    """

    coding: TransmitSignalCoding

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.device = SimulatedDevice(antennas=SimulatedUniformArray(SimulatedIdealAntenna, 1, (2, 2, 1)))
        self.coding = TransmitSignalCoding()

    def test_num_transmit_input_streams(self) -> None:
        """Transmit coding should comput the correct number of input streams"""

        # Test default value for an empty configuration with no assigned precoders
        self.assertEqual(1, self.coding.num_transmit_input_streams(1))
        self.assertEqual(5, self.coding.num_transmit_input_streams(5))

        # Test a configuration with a single encoder
        self.coding[0] = TransmitStreamEncoderMock()
        self.assertEqual(1, self.coding.num_transmit_input_streams(5))

    def test_encode_streams_validation(self) -> None:
        """Encoding should raise a ValueError on invalid input"""

        signal = Signal.Create(self.rng.standard_normal((3, 100)) + 1j * self.rng.standard_normal((3, 100)))
        with self.assertRaises(ValueError):
            self.coding[0] = TransmitStreamEncoderMock()
            self.coding.encode_streams(signal, self.device.state())

    def test_encode_streams(self) -> None:
        """Encoding should be delegated to the registeded encoders"""

        self.coding[0] = TransmitStreamEncoderMock()
        self.coding[1] = TransmitStreamEncoderMock()

        signal = Signal.Create(self.rng.standard_normal((1, 100)) + 1j * self.rng.standard_normal((1, 100)))
        encoded_signal = self.coding.encode_streams(signal, self.device.state())

        self.assertEqual(4, encoded_signal.num_streams)


class TestReceiveSignalCoding(_TestSignalCoding, TestCase):
    """Test the receive signal coding class.

    Also functions as a test for the ReceivePrecoding and Precoding base classes.
    """

    coding: ReceiveSignalCoding

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.device = SimulatedDevice(antennas=SimulatedUniformArray(SimulatedIdealAntenna, 1, (2, 2, 1)))
        self.coding = ReceiveSignalCoding()

    def test_num_receive_output_streams(self) -> None:
        """Receive coding should compute the correct number of output streams"""

        # Test default value for an empty configuration
        self.assertEqual(1, self.coding.num_receive_output_streams(1))
        self.assertEqual(4, self.coding.num_receive_output_streams(4))

        # Test a configuration with a single decoder
        self.coding[0] = ReceiveStreamDecoderMock()
        self.assertEqual(2, self.coding.num_receive_output_streams(4))
        self.assertEqual(1, self.coding.num_receive_output_streams(2))

    def test_decode_streams(self) -> None:
        """Decoding should be delegated to the registeded decoders"""

        self.coding[0] = ReceiveStreamDecoderMock()
        self.coding[1] = ReceiveStreamDecoderMock()

        signal = Signal.Create(self.rng.standard_normal((4, 100)) + 1j * self.rng.standard_normal((4, 100)))
        decoded_signal = self.coding.decode_streams(signal, self.device.state())

        self.assertEqual(1, decoded_signal.num_streams)
