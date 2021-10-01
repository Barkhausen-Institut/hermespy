import unittest
from unittest.mock import Mock, call
from ruamel.yaml import SafeRepresenter
import numpy as np
from numpy.testing import assert_array_equal

from .test_encoder import StubEncoder
from modem.coding.encoder_manager import EncoderManager


class TestEncoderManager(unittest.TestCase):
    """Test the `EncoderManager`, responsible for configuring arbitrary channel encodings."""

    def setUp(self) -> None:

        self.modem = Mock()
        self.encoder_alpha = StubEncoder(Mock(), 64)
        self.encoder_beta = StubEncoder(Mock(), 16)
        self.encoder_manager = EncoderManager(self.modem)

    def test_init(self) -> None:
        """Test the object initialization behaviour."""

        self.assertIs(self.modem, self.encoder_manager.modem, "Modem not properly initialized")
        self.assertIs(0, len(self.encoder_manager.encoders), "Encoder list not properly initialized")

    def test_to_yaml(self) -> None:
        """Serialization to YAML."""

        safe_representer = SafeRepresenter()
        node = EncoderManager.to_yaml(safe_representer, self.encoder_manager)
        self.assertEquals(node.value, 'null', "YAML serialization produced unexpected result")

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
            self.assertEquals(self.encoder_manager.modem, None, "This assert is never called")

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

        self.assertEquals(self.encoder_manager.bit_block_size, 1,
                          "Bit block size calculation produced unexpected result")

        self.encoder_manager.add_encoder(self.encoder_alpha)
        self.encoder_manager.add_encoder(self.encoder_beta)

        self.assertEquals(self.encoder_manager.bit_block_size, 32,
                          "Bit block size calculation produced unexpected result")

    def test_code_block_size(self) -> None:
        """Test the code block size calculation."""

        self.assertEquals(self.encoder_manager.code_block_size, 1,
                          "Code block size calculation produced unexpected result")

        self.encoder_manager.add_encoder(self.encoder_alpha)
        self.encoder_manager.add_encoder(self.encoder_beta)

        self.assertEquals(self.encoder_manager.code_block_size, 128)

    def test_rate(self) -> None:
        """Test the coding rate calculation."""

        self.assertEquals(self.encoder_manager.rate, 1.0,
                          "Coding rate calculation produced unexpected result")

        self.encoder_manager.add_encoder(self.encoder_alpha)
        self.assertEquals(self.encoder_manager.rate, self.encoder_alpha.rate,
                          "Coding rate calculation produced unexpected result")

        self.encoder_manager.add_encoder(self.encoder_beta)
        self.assertEquals(self.encoder_manager.rate, self.encoder_alpha.rate * self.encoder_beta.rate,
                          "Coding rate calculation produced unexpected result")

    def test_encoding(self) -> None:
        """Test the encoding behaviour."""

        # Default (empty) encoder manager
        data = np.arange(10)
        expected_code = data.copy()
        code = self.encoder_manager.encode(data)
        assert_array_equal(expected_code, code, "Default encoding behaviour unexpected")

        # Configured (mocked) encoder manager
        self.encoder_manager.add_encoder(self.encoder_alpha)
        self.encoder_manager.add_encoder(self.encoder_beta)

        data = ((np.arange(self.encoder_manager.bit_block_size) % 2) == 1).astype(int)
        expected_code = data.repeat(4)
        code = self.encoder_manager.encode(data)
        assert_array_equal(code, expected_code, "Encode mocking produced unexpected result")

    def test_decoding(self) -> None:
        """Test the decoding behaviour."""

        # Default (empty) encoder manager
        code = np.arange(10)
        expected_data = code.copy()
        data = self.encoder_manager.decode(code)
        assert_array_equal(expected_data, data, "Default decoding behaviour unexpected")

        # Configured (mocked) encoder manager
        self.encoder_manager.add_encoder(self.encoder_alpha)
        self.encoder_manager.add_encoder(self.encoder_beta)

        expected_data = ((np.arange(self.encoder_manager.bit_block_size) % 2) == 1).astype(int)
        code = expected_data.repeat(4)
        data = self.encoder_manager.decode(code)
        assert_array_equal(data, expected_data, "Decode mocking produced unexpected result")
