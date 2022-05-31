# -*- coding: utf-8 -*-
"""Coding testing."""

from typing import Type
from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_equal
from ruamel.yaml import SafeConstructor, SafeRepresenter, Node

from hermespy.core.factory import Serializable
from hermespy.coding import Encoder, EncoderManager


class StubEncoder(Encoder):
    """Encoder mock for testing only."""

    __block_size: int

    def __init__(self, manager: Mock, block_size: int) -> None:

        Encoder.__init__(self, manager)
        self.__block_size = block_size

    def encode(self, bits: np.ndarray) -> np.ndarray:
        return bits.repeat(2)

    def decode(self, encoded_bits: np.ndarray) -> np.ndarray:
        return encoded_bits[::2]

    @property
    def bit_block_size(self) -> int:
        return self.__block_size

    @property
    def code_block_size(self) -> int:
        return 2 * self.__block_size

    @classmethod
    def from_yaml(cls: Type[Serializable], constructor: SafeConstructor, node: Node) -> Serializable:
        pass

    @classmethod
    def to_yaml(cls: Type[Serializable], representer: SafeRepresenter, node: Serializable) -> Node:
        pass


class TestEncoder(TestCase):
    """Test the abstract Encoder base class."""

    def setUp(self) -> None:

        self.bits_in_frame = 100
        self.manager = Mock()
        self.encoder = StubEncoder(self.manager, self.bits_in_frame)

    def test_init(self) -> None:
        """Test that the init properly stores all parameters."""

        self.assertIs(self.encoder.manager, self.manager, "Manager init failed")

    def test_manager(self) -> None:
        """Encoder manager getter must return setter value."""

        manager = Mock()
        self.encoder.manager = manager
        self.assertIs(manager, self.encoder.manager, "Manager get / set failed")

    def test_rate(self) -> None:
        """Rate property check."""

        expected_rate = 0.5
        self.assertAlmostEqual(expected_rate, self.encoder.rate,
                               msg="Rate produced unexpected value")


class TestEncoderManager(TestCase):
    """Test the `EncoderManager`, responsible for configuring arbitrary channel encodings."""

    def setUp(self) -> None:

        self.modem = Mock()
        self.encoder_alpha = StubEncoder(Mock(), 64)
        self.encoder_beta = StubEncoder(Mock(), 16)
        self.encoder_manager = EncoderManager(self.modem)
        self.rng = np.random.default_rng(42)

    def test_init(self) -> None:
        """Test the object initialization behaviour."""

        self.assertIs(self.modem, self.encoder_manager.modem, "Modem not properly initialized")
        self.assertIs(0, len(self.encoder_manager.encoders), "Encoder list not properly initialized")
        self.assertEqual(True, self.encoder_manager.allow_truncating, "Truncating flag not properly initialized")
        self.assertEqual(True, self.encoder_manager.allow_padding, "Padding flag not properly initialized")

    def test_to_yaml(self) -> None:
        """Serialization to YAML."""

        safe_representer = SafeRepresenter()
        node = EncoderManager.to_yaml(safe_representer, self.encoder_manager)
        self.assertEqual(node.value, 'null', "YAML serialization produced unexpected result")

    def test_from_yaml(self) -> None:
        """Recall from YAML dump."""
        pass

    def test_modem(self) -> None:
        """Modem getter must return setter value."""

        modem = Mock()
        self.encoder_manager.modem = modem
        self.assertIs(modem, self.encoder_manager.modem, "Modem getter does not return setter value")

    def test_modem_getter_assert(self) -> None:
        """Modem getter must throw `RuntimeError` if manager is floating."""

        self.encoder_manager.modem = None
        with self.assertRaises(RuntimeError):
            self.assertEqual(self.encoder_manager.modem, None, "This assert is never called")

    def test_add_encoder_registration(self) -> None:
        """Added encoders must refer back to the manager they have been added to."""

        self.encoder_manager.add_encoder(self.encoder_alpha)
        self.encoder_manager.add_encoder(self.encoder_beta)

        self.assertIs(self.encoder_alpha.manager, self.encoder_manager, "Added encoder does not refer back to manager")
        self.assertIs(self.encoder_beta.manager, self.encoder_manager, "Added encoder does not refer back to manager")

    def test_encoder_sorting(self) -> None:
        """Test that encoders are automatically ordered in ascending order,
         depending on their expected number of input bits.
         """

        self.encoder_manager.add_encoder(self.encoder_alpha)    # Encoder alpha expects 32 input bits
        self.encoder_manager.add_encoder(self.encoder_beta)     # Encoder beta expects 16 input bits

        self.assertIs(self.encoder_manager.encoders[1], self.encoder_alpha, "Encoders sorted in unexpected order")
        self.assertIs(self.encoder_manager.encoders[0], self.encoder_beta, "Encoders sorted in unexpected order")

    def test_bit_block_size(self) -> None:
        """Test the bit block size calculation."""

        self.assertEqual(self.encoder_manager.bit_block_size, 1,
                         "Bit block size calculation produced unexpected result")

        self.encoder_manager.add_encoder(self.encoder_alpha)
        self.encoder_manager.add_encoder(self.encoder_beta)

        self.assertEqual(self.encoder_manager.bit_block_size, 32,
                         "Bit block size calculation produced unexpected result")

    def test_code_block_size(self) -> None:
        """Test the code block size calculation."""

        self.assertEqual(self.encoder_manager.code_block_size, 1,
                         "Code block size calculation produced unexpected result")

        self.encoder_manager.add_encoder(self.encoder_alpha)
        self.encoder_manager.add_encoder(self.encoder_beta)

        self.assertEqual(self.encoder_manager.code_block_size, 128)

    def test_rate(self) -> None:
        """Test the coding rate calculation."""

        self.assertEqual(self.encoder_manager.rate, 1.0,
                         "Coding rate calculation produced unexpected result")

        self.encoder_manager.add_encoder(self.encoder_alpha)
        self.assertEqual(self.encoder_manager.rate, self.encoder_alpha.rate,
                         "Coding rate calculation produced unexpected result")

        self.encoder_manager.add_encoder(self.encoder_beta)
        self.assertEqual(self.encoder_manager.rate, self.encoder_alpha.rate * self.encoder_beta.rate,
                         "Coding rate calculation produced unexpected result")

    def test_empty_encoding(self) -> None:
        """Test proper encoding without an encoder."""

        data = np.arange(10)
        expected_code = data.copy()
        code = self.encoder_manager.encode(data)
        assert_array_equal(expected_code, code)

    def test_single_encoding(self) -> None:
        """Test proper encoding with a single encoder."""

        self.encoder_manager.add_encoder(self.encoder_alpha)

        data = ((np.arange(self.encoder_manager.bit_block_size) % 2) == 1).astype(int)
        expected_code = data.repeat(2)
        code = self.encoder_manager.encode(data)
        assert_array_equal(code, expected_code)

    def test_chained_encodings(self) -> None:
        """Test proper encoding with multiple chained encoders."""

        self.encoder_manager.add_encoder(self.encoder_alpha)
        self.encoder_manager.add_encoder(self.encoder_beta)

        data = ((np.arange(self.encoder_manager.bit_block_size) % 2) == 1).astype(int)
        expected_code = data.repeat(4)
        code = self.encoder_manager.encode(data)
        assert_array_equal(code, expected_code)

    def test_empty_decoding(self) -> None:
        """Test proper decoding without an encoder."""

        code = np.arange(10)
        expected_data = code.copy()
        data = self.encoder_manager.decode(code)
        assert_array_equal(expected_data, data)

    def test_single_decoding(self) -> None:
        """Test proper decoding with a single decoder"""

        self.encoder_manager.add_encoder(self.encoder_alpha)

        expected_data = ((np.arange(self.encoder_manager.bit_block_size) % 2) == 1).astype(int)
        code = expected_data.repeat(2)
        data = self.encoder_manager.decode(code)
        assert_array_equal(data, expected_data)

    def test_chained_decoding(self) -> None:
        """Test proper decoding with multiple chained encoders."""

        # Configured (mocked) encoder manager
        self.encoder_manager.add_encoder(self.encoder_alpha)
        self.encoder_manager.add_encoder(self.encoder_beta)

        expected_data = ((np.arange(self.encoder_manager.bit_block_size) % 2) == 1).astype(int)
        code = expected_data.repeat(4)
        data = self.encoder_manager.decode(code)
        assert_array_equal(data, expected_data)

    def test_padding(self) -> None:
        """Test the padding of zeros on insufficient code lengths."""

        self.encoder_manager.add_encoder(self.encoder_alpha)

        data = np.ones(self.encoder_manager.bit_block_size, dtype=bool)
        code = self.encoder_manager.encode(data, self.encoder_manager.code_block_size + 2)
        self.assertEqual(len(code), self.encoder_manager.code_block_size + 2)

    def test_truncating(self) -> None:
        """Test the truncating of bits on overflowing data lengths"""

        self.encoder_manager.add_encoder(self.encoder_alpha)

        expected_data = (np.arange(self.encoder_manager.bit_block_size) % 2) == 1
        code = expected_data.repeat(2)
        data = self.encoder_manager.decode(code, expected_data.shape[0]-2)
        assert_array_equal(data, expected_data[:-2])

    def test_encode_decode(self) -> None:
        """Encoding a bit set and subsequently decoding it should yield the original set"""
        
        self.encoder_manager.add_encoder(self.encoder_alpha)
        self.encoder_manager.add_encoder(self.encoder_beta)
        
        expected_data = self.rng.integers(0, 2, 20, dtype=bool)
        
        code = self.encoder_manager.encode(expected_data)
        data = self.encoder_manager.decode(code, len(expected_data))
        
        assert_array_equal(expected_data, data)
