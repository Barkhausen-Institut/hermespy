import unittest
from fractions import Fraction
from tests.unit_tests.modem.utils import flatten_blocks
import os

import numpy as np
from scipy.io import loadmat

from modem.coding.ldpc_encoder import LdpcEncoder
from parameters_parser.parameters_ldpc_encoder import ParametersLdpcEncoder
import modem.coding.ldpc_binding.ldpc_binding as ldpc_binding


class TestLdpcEncoder(unittest.TestCase):
    def setUp(self) -> None:
        self.params = ParametersLdpcEncoder()
        self.params.code_rate = 2 / 3
        self.params.block_size = 512
        self.params.code_rate_fraction = Fraction(2, 3)
        self.params.no_iterations = 20
        self.params.custom_ldpc_codes = ""

        self.nActual = 528
        self.kActual = 352
        self.no_code_blocks = 2
        self.source_bits = self.no_code_blocks * self.kActual
        self.bits_in_frame = self.no_code_blocks * self.nActual

        self.encoder = LdpcEncoder(self.params, self.bits_in_frame)
        self.encoderTestResultsDir = os.path.join(
            os.path.dirname(__file__), 'res', 'ldpc')

    def test_ldpcBindingEncodingYieldsSameResultsAsPythonCode(self) -> None:
        params = ParametersLdpcEncoder()
        params.code_ratio = 2 / 3
        params.block_size = 256
        params.code_rate_fraction = Fraction(2, 3)
        params.custom_ldpc_codes = ""

        ldpc_results_mat = loadmat(os.path.join(
                self.encoderTestResultsDir, 'test_data_encoder_256_2_3.mat'
            ), squeeze_me=True)
        params.no_iterations = ldpc_results_mat['LDPC']['iterations'].item()

        bits_in_frame = len(ldpc_results_mat['code_words'][0])

        encoder = LdpcEncoder(params, bits_in_frame)
        data_word = ldpc_results_mat['bit_words']
        encoded_word = encoder.encode([data_word[0]])
        encoded_word_binding = encoder.encode_binding([data_word[0]])

        np.testing.assert_array_almost_equal(encoded_word[0], encoded_word_binding[0])


    def test_properEncoding_oneBlock(self) -> None:
        params = ParametersLdpcEncoder()
        params.code_ratio = 2 / 3
        params.block_size = 256
        params.code_rate_fraction = Fraction(2, 3)
        params.custom_ldpc_codes = ""

        ldpc_results_mat = loadmat(os.path.join(
                self.encoderTestResultsDir, 'test_data_encoder_256_2_3.mat'
            ), squeeze_me=True)
        params.no_iterations = ldpc_results_mat['LDPC']['iterations'].item()

        bits_in_frame = len(ldpc_results_mat['code_words'][0])

        encoder = LdpcEncoder(params, bits_in_frame)

        expected_encoded_word = ldpc_results_mat['code_words'][0]
        data_word = ldpc_results_mat['bit_words']
        encoded_word = encoder.encode([data_word[0]])

        np.testing.assert_array_almost_equal(encoded_word[0], expected_encoded_word)

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
        params = ParametersLdpcEncoder()
        params.code_ratio = 2 / 3
        params.block_size = 256
        params.code_rate_fraction = Fraction(2, 3)
        params.custom_ldpc_codes = ""

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

    def test_ifCodeBlocksDoNotFillUpFrame(self) -> None:
        no_bits_frame_too_long = 10
        self.bits_in_frame += no_bits_frame_too_long
        no_data_bits = self.no_code_blocks * self.kActual

        encoder = LdpcEncoder(self.params, self.bits_in_frame)

        data_bits_frame = np.random.randint(2, size=no_data_bits)
        encoded_frame = encoder.encode([data_bits_frame])

        self.assertEqual(len(encoded_frame[-1]), no_bits_frame_too_long)

    def test_noCodeBlocksCalculation_bitsInFrame_noMultipleOfN(self) -> None:
        self.bits_in_frame = 2 * self.bits_in_frame + 1
        encoder = LdpcEncoder(self.params, self.bits_in_frame)

        self.assertEqual(encoder.code_blocks, int(self.bits_in_frame / self.nActual))
