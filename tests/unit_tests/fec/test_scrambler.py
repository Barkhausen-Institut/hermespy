import unittest
import numpy as np
from numpy.testing import assert_array_equal

from hermespy.fec.scrambler import PseudoRandomGenerator
from hermespy.fec import Scrambler3GPP, Scrambler80211a
from unit_tests.core.test_factory import test_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestPseudoRandomGenerator(unittest.TestCase):
    """Test the pseudo random numbers rng"""

    def setUp(self) -> None:
        init_sequence = np.zeros(4)
        offset = 0
        self.expected_output = np.array([1, 0, 0, 0, 0, 0, 0, 1], dtype=int)
        self.generator = PseudoRandomGenerator(init_sequence, offset)

    def test_init_validation(self) -> None:
        """Initializtion should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            PseudoRandomGenerator(np.zeros(3))

    def test_generate(self) -> None:
        """Test the first generated bit"""

        for bit in self.expected_output:
            self.assertEqual(self.generator.generate(), bit, "Unexpected rng result")

    def test_generate_sequence(self) -> None:
        """Test the sequence generation"""

        assert_array_equal(self.expected_output, self.generator.generate_sequence(self.expected_output.shape[0]), "Unexpected sequence generated")

    def test_reset(self) -> None:
        """Test the rng reset behaviour"""

        _ = self.generator.generate_sequence(1000)
        self.generator.reset()

        assert_array_equal(self.expected_output, self.generator.generate_sequence(self.expected_output.shape[0]), "Unexpected sequence generated after reset")


class TestScrambler3GPP(unittest.TestCase):
    """Test the bit scrambling 3GPP standard implementation"""

    def setUp(self) -> None:
        """Set up testing"""

        self.scrambler = Scrambler3GPP()

    def test_properties(self) -> None:
        """Properties should return the correct values"""

        self.assertEqual(1, self.scrambler.bit_block_size)
        self.assertEqual(1, self.scrambler.code_block_size)

    def test_coding(self) -> None:
        """Test encoding and subsequent decoding behaviour"""

        data = np.random.randint(0, 2, 31)
        code = self.scrambler.encode(data)

        descrambler = Scrambler3GPP()
        decoded_data = descrambler.decode(code)

        assert_array_equal(data, decoded_data)

    def test_serialization(self) -> None:
        """Test scrambler serialization"""

        test_roundtrip_serialization(self, self.scrambler)


class TestScrambler80211a(unittest.TestCase):
    """Test the scrambling implemented according to"""

    def setUp(self) -> None:
        """Set up testing"""

        self.scrambler = Scrambler80211a()

    def test_properties(self) -> None:
        """Properties should return the correct values"""

        self.assertEqual(1, self.scrambler.bit_block_size)
        self.assertEqual(1, self.scrambler.code_block_size)

    def test_seed_validation(self) -> None:
        """Seed property setter should raise ValueErrors on invalid arguments"""

        with self.assertRaises(ValueError):
            self.scrambler.seed = np.array([0])

        with self.assertRaises(ValueError):
            self.scrambler.seed = np.array([0, 2, 0, 1, 1, 0, 1], dtype=int)

    def test_seed_setget(self) -> None:
        """Seed getter should return setter value"""

        seed = np.array([0, 1, 0, 1, 1, 0, 1], dtype=int)
        self.scrambler.seed = seed
        assert_array_equal(seed, self.scrambler.seed, "Seed getter does not return setter value")

    def test_sequence(self) -> None:
        """Make sure the scrambler produces an expected sequence internally"""

        self.scrambler.seed = np.ones(7, dtype=np.int8)
        expected_sequence = np.array([0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0])

        sequence = np.empty(expected_sequence.shape, dtype=int)
        for n in range(sequence.shape[0]):
            sequence[n] = self.scrambler._Scrambler80211a__forward_code_bit()

        np.testing.assert_array_equal(expected_sequence, sequence)

    def test_encode(self) -> None:
        """Test the scrambling against the example tables of IEEE 802.11a.

        See also Annex G.5.2 of IEEE Std 802.11a-1999
        """

        self.scrambler.seed = np.array([1, 0, 1, 1, 1, 0, 1])
        data = np.zeros(72, dtype=np.int8)
        data[[18, 25, 41, 42, 43, 45, 61, 62, 67]] = 1
        expected_scramble = np.zeros(72, dtype=np.int8)
        expected_scramble[[1, 2, 4, 5, 11, 12, 15, 16, 20, 23, 24, 28, 29, 30, 31, 33, 34, 36, 42, 47, 48, 49, 50, 51, 53, 56, 58, 61, 63, 65, 66, 71]] = 1

        assert_array_equal(expected_scramble, self.scrambler.encode(data))

    def test_decode(self) -> None:
        """Test the de-scrambling against the example tables of IEEE 802.11a.

        See also Annex G.5.2 of IEEE Std 802.11a-1999
        """

        self.scrambler.seed = np.array([1, 0, 1, 1, 1, 0, 1])
        scramble = np.zeros(72, dtype=np.int8)
        scramble[[1, 2, 4, 5, 11, 12, 15, 16, 20, 23, 24, 28, 29, 30, 31, 33, 34, 36, 42, 47, 48, 49, 50, 51, 53, 56, 58, 61, 63, 65, 66, 71]] = 1
        expected_data = np.zeros(72, dtype=np.int8)
        expected_data[[18, 25, 41, 42, 43, 45, 61, 62, 67]] = 1

        assert_array_equal(expected_data, self.scrambler.decode(scramble))

    def test_serialization(self) -> None:
        """Test scrambler serialization"""

        test_roundtrip_serialization(self, self.scrambler)
