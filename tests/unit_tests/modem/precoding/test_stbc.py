# -*- coding: utf-8 -*-

from itertools import product
from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal

from hermespy.modem import StatedSymbols, Alamouti, Ganesan
from hermespy.simulation import SimulatedDevice, SimulatedIdealAntenna, SimulatedUniformArray
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Egor Achkasov"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestAlamouti(TestCase):
    """Test alamouti space time block coding precoder"""

    """Test alamouti space time block coding precoder"""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.alamouti = Alamouti()

    def test_properties(self) -> None:
        """Properties should return the expected values"""

        self.assertEqual(2, self.alamouti.num_transmit_input_symbols)
        self.assertEqual(2, self.alamouti.num_transmit_output_symbols)
        self.assertEqual(2, self.alamouti.num_receive_input_symbols)
        self.assertEqual(2, self.alamouti.num_receive_output_symbols)

    def test_encode_validation(self) -> None:
        """Encoding routine should raise errors on invalid calls"""

        # Test assertion for single-stream input
        raw_symbols = self.rng.standard_normal((2, 2, 100)) + 1j * self.rng.standard_normal((2, 2, 100))
        states = self.rng.standard_normal((2, 1, 2, 100)) + 1j * self.rng.standard_normal((2, 1, 2, 100))
        with self.assertRaises(ValueError):
            self.alamouti.encode_symbols(StatedSymbols(raw_symbols, states), 2)

        # Test assertion for two-stream output
        raw_symbols = self.rng.standard_normal((1, 2, 100)) + 1j * self.rng.standard_normal((1, 2, 100))
        states = self.rng.standard_normal((1, 1, 2, 100)) + 1j * self.rng.standard_normal((1, 1, 2, 100))
        with self.assertRaises(ValueError):
            self.alamouti.encode_symbols(StatedSymbols(raw_symbols, states), 3)

        # Test assertion for number of blocks not divisible by 2
        raw_symbols = self.rng.standard_normal((1, 3, 100)) + 1j * self.rng.standard_normal((1, 3, 100))
        states = self.rng.standard_normal((1, 1, 3, 100)) + 1j * self.rng.standard_normal((1, 1, 3, 100))
        with self.assertRaises(ValueError):
            self.alamouti.encode_symbols(StatedSymbols(raw_symbols, states), 2)

    def test_encode(self) -> None:
        """Test Alamouti MIMO encoding"""

        raw_symbols = self.rng.standard_normal((1, 2, 100)) + 1j * self.rng.standard_normal((1, 2, 100))
        states = self.rng.standard_normal((1, 1, 2, 100)) + 1j * self.rng.standard_normal((1, 1, 2, 100))
        stated_symbols = StatedSymbols(raw_symbols, states)

        encoded_symbols = self.alamouti.encode_symbols(stated_symbols, 2)

        self.assertEqual(2, encoded_symbols.num_streams)

    def test_decode_validation(self) -> None:
        """Decoding routine should raise errors on invalid calls"""

        raw_symbols = self.rng.standard_normal((2, 1, 100)) + 1j * self.rng.standard_normal((2, 1, 100))
        states = self.rng.standard_normal((2, 2, 1, 100)) + 1j * self.rng.standard_normal((2, 2, 1, 100))

        with self.assertRaises(ValueError):
            self.alamouti.decode_symbols(StatedSymbols(raw_symbols, states), 1)

    def test_decode(self) -> None:
        """Test Alamouti MIMO decoding"""

        num_blocks = 8
        num_symbols = 2

        raw_symbols = self.rng.standard_normal((1, num_blocks, num_symbols)) + 1j * self.rng.standard_normal((1, num_blocks, num_symbols))
        states = self.rng.standard_normal((1, 1, num_blocks, num_symbols)) + 1j * self.rng.standard_normal((1, 1, num_blocks, num_symbols))
        stated_symbols = StatedSymbols(raw_symbols, states)

        encoded_symbols = self.alamouti.encode_symbols(stated_symbols, 2)

        ideal_channel_state = np.zeros((2, 2, num_blocks, num_symbols), dtype=np.complex128)
        ideal_channel_state[0, 0, :, :] = 1.0
        ideal_channel_state[1, 1, :, :] = 1j

        ideal_received_symbols = encoded_symbols.raw.copy()
        ideal_decoded_symbols = self.alamouti.decode_symbols(StatedSymbols(ideal_received_symbols, ideal_channel_state), 1)

        assert_array_almost_equal(raw_symbols, ideal_decoded_symbols.raw[[0], ::])

        channel_state = self.rng.standard_normal((2, 2, int(0.5 * num_blocks), num_symbols)) + 1j * self.rng.standard_normal((2, 2, int(0.5 * num_blocks), num_symbols))
        channel_state = np.repeat(channel_state, 2, axis=2)  # Make the channel coherent over two symbol blocks

        received_encoded_symbols = np.zeros((2, num_blocks, num_symbols), dtype=np.complex128)
        for b, s in product(range(num_blocks), range(num_symbols)):
            received_encoded_symbols[:, b, s] = channel_state[:, :, b, s] @ encoded_symbols.raw[:, b, s]

        decoded_symbols = self.alamouti.decode_symbols(StatedSymbols(received_encoded_symbols, channel_state), 1)
        assert_array_almost_equal(raw_symbols, decoded_symbols.raw[[0], ::])

    def test_num_transmit_input_streams(self) -> None:
        """Number of transmit input streams should report the correct value"""

        self.assertEqual(1, self.alamouti._num_transmit_input_streams(2))
        self.assertEqual(-1, self.alamouti._num_transmit_input_streams(1))

    def test_receive_output_streams(self) -> None:
        """Number of receive output streams should report the correct value"""

        self.assertEqual(5, self.alamouti.num_receive_output_streams(5))

    def test_yaml_roundtrip_serialization(self) -> None:
        """Test YAML roundtrip serialization"""

        self.alamouti.precoding = None
        test_yaml_roundtrip_serialization(self, self.alamouti)


class TestGanesan(TestCase):
    """Test ganesan space time block coding precoder"""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.ganesan = Ganesan()

    def test_properties(self) -> None:
        """Properties should return the expected values"""

        self.assertEqual(3, self.ganesan.num_transmit_input_symbols)
        self.assertEqual(4, self.ganesan.num_transmit_output_symbols)
        self.assertEqual(4, self.ganesan.num_receive_input_symbols)
        self.assertEqual(3, self.ganesan.num_receive_output_symbols)

    def test_encode_validation(self) -> None:
        """Encoding routine should raise errors on invalid calls"""

        # check exception on number of stream != 1
        raw_symbols = self.rng.standard_normal((2, 4, 100)) + 1j * self.rng.standard_normal((2, 4, 100))
        states = self.rng.standard_normal((2, 1, 4, 100)) + 1j * self.rng.standard_normal((2, 1, 4, 100))
        with self.assertRaises(ValueError):
            self.ganesan.encode_symbols(StatedSymbols(raw_symbols, states), 4)

        # check exception on number of antenas != 4
        raw_symbols = self.rng.standard_normal((1, 2, 100)) + 1j * self.rng.standard_normal((1, 2, 100))
        states = self.rng.standard_normal((1, 1, 2, 100)) + 1j * self.rng.standard_normal((1, 1, 2, 100))
        with self.assertRaises(RuntimeError):
            self.ganesan.encode_symbols(StatedSymbols(raw_symbols, states), 5)

        # check exception on number of blocks % 3 != 0
        raw_symbols = self.rng.standard_normal((1, 1, 100)) + 1j * self.rng.standard_normal((1, 1, 100))
        states = self.rng.standard_normal((1, 4, 1, 100)) + 1j * self.rng.standard_normal((1, 4, 1, 100))
        with self.assertRaises(ValueError):
            self.ganesan.encode_symbols(StatedSymbols(raw_symbols, states), 4)

    def test_encode(self) -> None:
        """Test Ganesan MIMO encoding"""

        raw_symbols = self.rng.standard_normal((1, 3, 100)) + 1j * self.rng.standard_normal((1, 3, 100))
        states = self.rng.standard_normal((1, 1, 3, 100)) + 1j * self.rng.standard_normal((1, 1, 3, 100))
        stated_symbols = StatedSymbols(raw_symbols, states)

        encoded_symbols = self.ganesan.encode_symbols(stated_symbols ,4)
        self.assertEqual(4, encoded_symbols.num_streams)

    def test_decode_validation(self) -> None:
        """Decoding routine should raise errors on invalid calls"""

        # check ValueError on num blocks % 4 != 0
        raw_symbols = self.rng.standard_normal((4, 9, 100)) + 1j * self.rng.standard_normal((4, 9, 100))
        states = self.rng.standard_normal((4, 2, 9, 100)) + 1j * self.rng.standard_normal((4, 2, 9, 100))
        with self.assertRaises(ValueError):
            self.ganesan.decode_symbols(StatedSymbols(raw_symbols, states), 1)

        # check ValueError on num stream != 4
        raw_symbols = self.rng.standard_normal((2, 8, 100)) + 1j * self.rng.standard_normal((2, 8, 100))
        states = self.rng.standard_normal((2, 2, 8, 100)) + 1j * self.rng.standard_normal((2, 2, 8, 100))
        with self.assertRaises(ValueError):
            self.ganesan.decode_symbols(StatedSymbols(raw_symbols, states), 1)

    def test_decode(self) -> None:
        """Test Ganesan MIMO decoding"""

        num_blocks = 6  # Ganesan MIMO has symbol rate of 3/4, so this must be divisable by 3
        num_symbols = 10
        num_tx = 4  # Ganesan requires 4 Tx
        num_rx = 2

        # Generate symbols and encode them
        raw_symbols = self.rng.standard_normal((1, num_blocks, num_symbols)) + 1j * self.rng.standard_normal((1, num_blocks, num_symbols))
        states = np.empty((1, num_rx, num_blocks, num_symbols))
        raw_stated_symbols = StatedSymbols(raw_symbols, states)
        encoded_symbols = self.ganesan.encode_symbols(raw_stated_symbols, 4)

        # Test the decoder in case of ideal communication channels

        # Init ideal channel states as
        # an absense of information transfer between Tx i and Rx j where i != j
        # and changless transfer between Tx i and Rx j where i == j
        ideal_channel_state = np.zeros((num_rx, num_tx, num_blocks // 3 * 4, num_symbols), dtype=np.complex128)
        ideal_channel_state[0, 0, :, :] = 1.0
        ideal_channel_state[0, 1, :, :] = 1j
        ideal_channel_state[0, 2, :, :] = 1.0
        ideal_channel_state[0, 3, :, :] = 1j
        ideal_channel_state[1, 0, :, :] = 1j
        ideal_channel_state[1, 1, :, :] = 1
        ideal_channel_state[1, 2, :, :] = 1j
        ideal_channel_state[1, 3, :, :] = 1

        # Apply channel gains and pass the result to the decoder
        ideal_encoded_symbols = np.empty((num_rx, num_blocks // 3 * 4, num_symbols), dtype=np.complex128)
        for b, s in product(range(num_blocks // 3 * 4), range(num_symbols)):
            ideal_encoded_symbols[:, b, s] = ideal_channel_state[:, :, b, s] @ encoded_symbols.raw[:, b, s]
        ideal_decoded_symbols = self.ganesan.decode_symbols(StatedSymbols(ideal_encoded_symbols, ideal_channel_state), 2)

        # Compare the decoded symbols from the first Rx with the originally generated symbols
        assert_array_almost_equal(raw_stated_symbols.raw, ideal_decoded_symbols.raw[[0], ::])

        # Apply channel gains and pass the result to the decoder
        random_channel_state = self.rng.standard_normal((num_rx, num_tx, num_blocks // 3 * 4, num_symbols)) + 1j * self.rng.standard_normal((num_rx, num_tx, num_blocks // 3 * 4, num_symbols))
        propagated_symbols = np.empty((num_rx, num_blocks // 3 * 4, num_symbols), dtype=np.complex128)
        for b, s in product(range(num_blocks // 3 * 4), range(num_symbols)):
            propagated_symbols[:, b, s] = random_channel_state[:, :, b, s] @ encoded_symbols.raw[:, b, s]

        stated_propagated_symbols = StatedSymbols(propagated_symbols, random_channel_state)
        decoded_symbols = self.ganesan.decode_symbols(stated_propagated_symbols, 1)

        # Compare the decoded symbols from the first Rx with the originally generated symbols
        assert_array_almost_equal(raw_stated_symbols.raw, decoded_symbols.raw[[0], ::])

    def test_num_transmit_input_streams(self) -> None:
        """Number of transmit input streams should report the correct value"""

        self.assertEqual(1, self.ganesan._num_transmit_input_streams(4))
        self.assertEqual(-1, self.ganesan._num_transmit_input_streams(3))

    def test_receive_output_streams(self) -> None:
        """Number of receive output streams should report the correct value"""

        self.assertEqual(5, self.ganesan.num_receive_output_streams(5))

    def test_yaml_roundtrip_serialization(self) -> None:
        """Test YAML roundtrip serialization"""

        self.ganesan.precoding = None
        test_yaml_roundtrip_serialization(self, self.ganesan)
