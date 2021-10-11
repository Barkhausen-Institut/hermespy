import unittest
from fractions import Fraction
import os
from numpy.testing import assert_array_equal
from scipy.io import loadmat

from coding.ldpc import LDPC
from coding.ldpc_binding.ldpc import LDPCBinding


class TestLDPC(unittest.TestCase):
    """Test LDPC encoding behaviour."""

    def setUp(self) -> None:

        self.block_size = 256
        self.rate = Fraction(2, 3)
        self.iterations = 20

        self.encoder = LDPC(self.block_size, self.rate, self.iterations)

        self.encoderTestResultsDir = os.path.join(os.path.dirname(__file__), 'res', 'ldpc')

    def test_init(self) -> None:
        """Test the init parameter adoptions."""

        self.assertEqual(self.iterations, self.encoder.iterations, "Iterations not properly initialized")
        self.assertAlmostEqual(float(self.rate), self.encoder.rate, msg="Rate not properly initialized")

    def test_iterations_get_set(self) -> None:
        """Iterations property getter must return setter argument."""

        iterations = 25
        self.encoder.iterations = iterations
        self.assertEqual(iterations, self.encoder.iterations)

    def test_iterations_validation(self) -> None:
        """Iterations setter must raise Exception on invalid arguments."""

        with self.assertRaises(ValueError):
            self.encoder.iterations = -1

        with self.assertRaises(ValueError):
            self.encoder.iterations = 0

    def test_encoding(self) -> None:
        """Test encoding behaviour against a pre-calculated set of data-code pairs."""

        ldpc_results_mat = loadmat(os.path.join(
                self.encoderTestResultsDir, 'test_data_encoder_256_2_3.mat'
            ), squeeze_me=True)

        iterations = ldpc_results_mat['LDPC']['iterations'].item()
        bit_blocks = ldpc_results_mat['bit_words'].astype(int)
        expected_codes = ldpc_results_mat['code_words'].astype(int)
        num_blocks = expected_codes.shape[0]

        self.encoder.set_rate(256, Fraction(2, 3))
        self.encoder.iterations = iterations

        for n in range(num_blocks):

            bit_block = bit_blocks[n, :]
            expected_code = expected_codes[n, :]
            code = self.encoder.encode(bit_block)

            assert_array_equal(code, expected_code, "LDPC encoding produced unexpected result in block {}".format(n))

    def test_decoding(self) -> None:
        """Test decoding behaviour against a pre-calculated set of data-code pairs."""

        ldpc_results_mat = loadmat(os.path.join(
                self.encoderTestResultsDir, 'test_data_encoder_256_2_3.mat'
            ), squeeze_me=True)

        iterations = ldpc_results_mat['LDPC']['iterations'].item()
        expected_bit_blocks = ldpc_results_mat['bit_words'].astype(int)
        codes = ldpc_results_mat['code_words'].astype(int)
        num_blocks = codes.shape[0]

        self.encoder.set_rate(256, Fraction(2, 3))
        self.encoder.iterations = iterations

        for n in range(num_blocks):

            code = codes[n, :]
            expected_bit_block = expected_bit_blocks[n, :]
            bit_block = self.encoder.decode(code)

            assert_array_equal(bit_block, expected_bit_block,
                               "LDPC decoding produced unexpected result in block {}".format(n))

    def test_readCustomLdpcEncoder_fileExists(self) -> None:
        """Test the LDPC code lookup from custom directories."""

        custom_codes = os.path.join(os.path.dirname(os.path.abspath(__file__)), "res", "ldpc", "custom_ldpc_codes")
        block_size = 123
        rate = Fraction(9, 10)
        file_location = os.path.join(custom_codes, "BS{}_CR{}_{}.mat".format(
            block_size,
            rate.numerator,
            rate.denominator
        ))

        mat = loadmat(file_location, squeeze_me=True)
        Z = mat['LDPC']['Z'].item()
        H = mat['LDPC']['H'].item()
        G = mat['LDPC']['G'].item()
        G = G[:, 2*Z:]

        self.encoder.custom_codes.add(custom_codes)
        self.encoder.set_rate(block_size, rate)

        self.assertEqual(G.shape[0], self.encoder.bit_block_size, "Custom LDPC import produced unexpected result")
        self.assertEqual(G.shape[1], self.encoder.code_block_size, "Custom LDPC import produced unexpected result")
        self.assertEqual(H.shape[1] - 2 * Z, self.encoder.code_block_size, "Custom LDPC import produced unexpected result")
        self.assertEqual(H.shape[1] - G.shape[1], 2 * Z, "Custom LDPC import produced unexpected result")

    def test_rate(self) -> None:
        """Test the rate property getter."""

        self.assertAlmostEqual(float(self.rate), self.encoder.rate, msg="Rate computation produced unexpected result")

    def test_set_rate(self) -> None:
        """Test the rate set routine."""

        block_size = 512
        rate = Fraction(1, 2)

        self.encoder.set_rate(block_size, rate)
        self.assertAlmostEqual(float(rate), self.encoder.rate, msg="Rate set unexpected result")

    def test_set_rate_assert(self) -> None:
        """Rate set routine should raise ValueError on unsupported parameters."""

        with self.assertRaises(ValueError):
            self.encoder.set_rate(1, Fraction(1, 2))

        with self.assertRaises(ValueError):
            self.encoder.set_rate(512, Fraction(2, 1))

        with self.assertRaises(ValueError):
            self.encoder.set_rate(1, Fraction(1, 1))


class TestLDPCBinding(unittest.TestCase):
    """Test Cpp bindings of the LDPC encoder."""

    def setUp(self) -> None:

        self.block_size = 256
        self.rate = Fraction(2, 3)
        self.iterations = 20

        self.encoder = LDPCBinding(self.block_size, self.rate, self.iterations)

        self.encoderTestResultsDir = os.path.join(os.path.dirname(__file__), 'res', 'ldpc')

    def test_encoding(self) -> None:
        """Test encoding behaviour against a pre-calculated set of data-code pairs."""

        ldpc_results_mat = loadmat(os.path.join(
                self.encoderTestResultsDir, 'test_data_encoder_256_2_3.mat'
            ), squeeze_me=True)

        iterations = ldpc_results_mat['LDPC']['iterations'].item()
        bit_blocks = ldpc_results_mat['bit_words'].astype(int)
        expected_codes = ldpc_results_mat['code_words'].astype(int)
        num_blocks = expected_codes.shape[0]

        self.encoder.set_rate(256, Fraction(2, 3))
        self.encoder.iterations = iterations

        for n in range(num_blocks):

            bit_block = bit_blocks[n, :]
            expected_code = expected_codes[n, :]
            code = self.encoder.encode(bit_block)

            assert_array_equal(code, expected_code,
                               "LDPC Cpp binding encoding produced unexpected result in block {}".format(n))

    def test_decoding(self) -> None:
        """Test decoding behaviour against a pre-calculated set of data-code pairs."""

        ldpc_results_mat = loadmat(os.path.join(
                self.encoderTestResultsDir, 'test_data_encoder_256_2_3.mat'
            ), squeeze_me=True)

        iterations = ldpc_results_mat['LDPC']['iterations'].item()
        expected_bit_blocks = ldpc_results_mat['bit_words'].astype(int)
        codes = ldpc_results_mat['code_words'].astype(int)
        num_blocks = codes.shape[0]

        self.encoder.set_rate(256, Fraction(2, 3))
        self.encoder.iterations = iterations

        for n in range(num_blocks):

            code = codes[n, :]
            expected_bit_block = expected_bit_blocks[n, :]
            bit_block = self.encoder.decode(code)

            assert_array_equal(bit_block, expected_bit_block,
                               "LDPC Cpp binding decoding produced unexpected result in block {}".format(n))
