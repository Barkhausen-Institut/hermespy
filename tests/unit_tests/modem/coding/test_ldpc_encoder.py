import unittest
from fractions import Fraction
from tests.unit_tests.modem.utils import flatten_blocks
import os
import numpy as np
from numpy.testing import assert_array_equal
from scipy.io import loadmat

from modem.coding import LDPC


class TestLdpcEncoder(unittest.TestCase):
    """Test LDPC encoding behaviour."""

    def setUp(self) -> None:

        self.block_size = 256
        self.rate = Fraction(2, 3)
        self.iterations = 20

        self.encoder = LDPC(self.block_size, self.rate, self.iterations)

        self.encoderTestResultsDir = os.path.join(os.path.dirname(__file__), 'res', 'ldpc')

        """self.params = ParametersLdpcEncoder()
        self.params.code_rate = 2 / 3
        self.params.block_size =
        self.params.code_rate_fraction = Fraction(2, 3)
        self.params.no_iterations = 20
        self.params.custom_ldpc_codes = ""
        self.params.use_binding = False

        self.nActual = 528
        self.kActual = 352
        self.no_code_blocks = 2
        self.source_bits = self.no_code_blocks * self.kActual
        self.bits_in_frame = self.no_code_blocks * self.nActual"""

    def test_ldpcBindingEncodingYieldsSameResultsAsPythonCode(self) -> None:
        params = ParametersLdpcEncoder()
        params.code_ratio = 2 / 3
        params.block_size = 256
        params.code_rate_fraction = Fraction(2, 3)
        params.custom_ldpc_codes = ""
        params.use_binding = False

        ldpc_results_mat = loadmat(os.path.join(
                self.encoderTestResultsDir, 'test_data_encoder_256_2_3.mat'
            ), squeeze_me=True)
        params.no_iterations = ldpc_results_mat['LDPC']['iterations'].item()

        bits_in_frame = len(ldpc_results_mat['code_words'][0])
        data_word = ldpc_results_mat['bit_words']

        encoder = LdpcEncoder(params, bits_in_frame)
        encoded_word = encoder.encode([data_word[0]])
        encoder.params.use_binding = True
        encoded_word_binding = encoder.encode([data_word[0]])

        np.testing.assert_array_almost_equal(encoded_word[0], encoded_word_binding[0])

    def test_ldpcBindingDecodingYieldsSameResultsAsPythonCode(self) -> None:
        params = ParametersLdpcEncoder()
        params.code_ratio = 2 / 3
        params.block_size = 256
        params.code_rate_fraction = Fraction(2, 3)
        params.custom_ldpc_codes = ""
        params.use_binding = False

        ldpc_results_mat = loadmat(os.path.join(
                self.encoderTestResultsDir, 'test_data_decoder_256_2_3.mat'
            ), squeeze_me=True)
        params.no_iterations = ldpc_results_mat['LDPC']['iterations'].item()
        bits_in_frame = len(ldpc_results_mat['llrs'][0])
        llrs = -ldpc_results_mat['llrs'][0]

        encoder = LdpcEncoder(params, bits_in_frame)
        decoded_word = encoder.decode([llrs])
        encoder.params.use_binding = True
        decoded_word_binding = encoder.decode([llrs])
        np.testing.assert_array_almost_equal(decoded_word[0], decoded_word_binding[0])

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
        self.params.custom_ldpc_codes = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "res", "ldpc", "custom_ldpc_codes"
        )
        self.params.code_rate = 9/10
        self.params.code_rate_fraction = Fraction(9, 10)
        self.params.block_size = 123

        encoder = LdpcEncoder(self.params, 1000)
        self.assertEqual(encoder.Z, 6)

    def test_properEncoding_multipleBlocks(self) -> None:

        ldpc_results_mat = loadmat(os.path.join(
                self.encoderTestResultsDir, 'test_data_encoder_256_2_3.mat'
            ), squeeze_me=True)
        params.no_iterations = ldpc_results_mat['LDPC']['iterations'].item()

        bits_in_frame = len(ldpc_results_mat['code_words'][0])

        encoder = LdpcEncoder(params, bits_in_frame)

        expected_encoded_words = ldpc_results_mat['code_words']
        data_words = ldpc_results_mat['bit_words']
        encoded_words = encoder.encode(data_words)
        np.testing.assert_array_almost_equal(encoded_words, expected_encoded_words)

    def test_properDecoding_oneBlock(self) -> None:
        params = ParametersLdpcEncoder()
        params.code_ratio = 2 / 3
        params.block_size = 256
        params.code_rate_fraction = Fraction(2, 3)
        params.custom_ldpc_codes = ""
        params.use_binding = False

        ldpc_results_mat = loadmat(os.path.join(
                self.encoderTestResultsDir, 'test_data_decoder_256_2_3.mat'
            ), squeeze_me=True)

        params.no_iterations = ldpc_results_mat['LDPC']['iterations'].item()

        bits_in_frame = len(ldpc_results_mat['llrs'][0])

        encoder = LdpcEncoder(params, bits_in_frame)

        expected_decoded_word = ldpc_results_mat['est_bit_words'][0]
        llrs = -ldpc_results_mat['llrs']
        decoded_word = encoder.decode([llrs[0]])

        np.testing.assert_array_almost_equal(decoded_word[0], expected_decoded_word)

    def test_properDecoding_multipleBlocks(self) -> None:
        params = ParametersLdpcEncoder()
        params.code_ratio = 2 / 3
        params.block_size = 256
        params.code_rate_fraction = Fraction(2, 3)
        params.custom_ldpc_codes = ""
        params.use_binding = False

        ldpc_results_mat = loadmat(os.path.join(
                self.encoderTestResultsDir, 'test_data_decoder_256_2_3.mat'
            ), squeeze_me=True)
        params.no_iterations = ldpc_results_mat['LDPC']['iterations'].item()

        bits_in_frame = len(ldpc_results_mat['llrs'][0])

        encoder = LdpcEncoder(params, bits_in_frame)

        expected_decoded_words = ldpc_results_mat['est_bit_words']
        llrs = -ldpc_results_mat['llrs']
        decoded_words = encoder.decode(llrs)

        np.testing.assert_array_almost_equal(decoded_words, expected_decoded_words)

    def test_properDecoding_multipleBlocks_pythonBinding(self) -> None:
        params = ParametersLdpcEncoder()
        params.code_ratio = 2 / 3
        params.block_size = 256
        params.code_rate_fraction = Fraction(2, 3)
        params.custom_ldpc_codes = ""
        params.use_binding = True

        ldpc_results_mat = loadmat(os.path.join(
                self.encoderTestResultsDir, 'test_data_decoder_256_2_3.mat'
            ), squeeze_me=True)
        params.no_iterations = ldpc_results_mat['LDPC']['iterations'].item()

        bits_in_frame = len(ldpc_results_mat['llrs'][0])

        encoder = LdpcEncoder(params, bits_in_frame)

        expected_decoded_words = ldpc_results_mat['est_bit_words']
        llrs = -ldpc_results_mat['llrs']
        decoded_words = encoder.decode(llrs)

        np.testing.assert_array_almost_equal(decoded_words, expected_decoded_words)

    def test_noDataBitsLongerThanCodeBlock(self) -> None:
        data_word_too_long = np.random.randint(2, size=self.encoder.data_bits_k + 3)

        self.assertRaises(
            ValueError,
            lambda: self.encoder.encode([data_word_too_long])
        )

    def test_paramCalculation(self) -> None:
        self.assertEqual(self.encoder.encoded_bits_n, self.nActual)
        self.assertEqual(self.encoder.data_bits_k, self.kActual)
        self.assertEqual(self.encoder.source_bits, self.no_code_blocks * self.kActual)

    def test_decoding_checkIfFillUpBitsAreDiscarded(self) -> None:
        no_bits_filled_up = 10
        self.bits_in_frame += no_bits_filled_up

        encoder = LdpcEncoder(self.params, self.bits_in_frame)
        encoded_bits = []

        for block_idx in range(self.no_code_blocks):
            encoded_bits.append(np.random.randint(2, size=self.nActual))

        encoded_bits.append(np.random.randint(2, size=no_bits_filled_up))
        decoded_bits = encoder.decode([flatten_blocks(encoded_bits)])

        self.assertEqual(len(decoded_bits[0]), self.no_code_blocks * self.kActual)

    def test_decoding_checkIfFillUpBitsAreDiscarded_pythonBinding(self) -> None:
        no_bits_filled_up = 10
        self.bits_in_frame += no_bits_filled_up
        self.params.use_binding = True

        encoder = LdpcEncoder(self.params, self.bits_in_frame)
        encoded_bits = []

        for block_idx in range(self.no_code_blocks):
            encoded_bits.append(np.random.randint(2, size=self.nActual))

        encoded_bits.append(np.random.randint(2, size=no_bits_filled_up))
        decoded_bits = encoder.decode([flatten_blocks(encoded_bits)])

        self.assertEqual(len(decoded_bits[0]), self.no_code_blocks * self.kActual)

    def test_ifCodeBlocksDoNotFillUpFrame(self) -> None:
        no_bits_frame_too_long = 10
        self.bits_in_frame += no_bits_frame_too_long
        no_data_bits = self.no_code_blocks * self.kActual

        encoder = LdpcEncoder(self.params, self.bits_in_frame)

        data_bits_frame = np.random.randint(2, size=no_data_bits)
        encoded_frame = encoder.encode([data_bits_frame])

        self.assertEqual(len(encoded_frame[-1]), no_bits_frame_too_long)

    def test_ifCodeBlocksDoNotFillUpFrame_pythonBinding(self) -> None:
        no_bits_frame_too_long = 10
        self.bits_in_frame += no_bits_frame_too_long
        no_data_bits = self.no_code_blocks * self.kActual
        self.params.use_binding = True

        encoder = LdpcEncoder(self.params, self.bits_in_frame)

        data_bits_frame = np.random.randint(2, size=no_data_bits)
        encoded_frame = encoder.encode([data_bits_frame])

        self.assertEqual(len(encoded_frame[-1]), no_bits_frame_too_long)

    def test_noCodeBlocksCalculation_bitsInFrame_noMultipleOfN(self) -> None:
        self.bits_in_frame = 2 * self.bits_in_frame + 1
        encoder = LdpcEncoder(self.params, self.bits_in_frame)

        self.assertEqual(encoder.code_blocks, int(self.bits_in_frame / self.nActual))
