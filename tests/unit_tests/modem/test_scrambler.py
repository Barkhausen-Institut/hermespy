import unittest
import numpy as np

from .test_encoder import TestAbstractEncoder
from modem.coding.scrambler import Scrambler3GPP, Scrambler80211a
from parameters_parser.parameters_scrambler import ParametersScrambler
from .utils import assert_frame_equality


class TestScrambler3GPP(TestAbstractEncoder, unittest.TestCase):

    @property
    def encoder(self) -> Scrambler3GPP:
        return self.scrambler

    def setUp(self) -> None:
        self.scrambler = Scrambler3GPP(ParametersScrambler(), 31)


class TestScrambler80211a(TestAbstractEncoder, unittest.TestCase):

    @property
    def encoder(self) -> Scrambler80211a:
        return self.scrambler

    def setUp(self) -> None:

        self.scrambler = Scrambler80211a(ParametersScrambler(), 31)

        self.__sequence_seed = np.ones(7, dtype=int)
        self.__sequence_check = np.array([0, 0, 0, 0, 1, 1, 1, 0,
                                          1, 1, 1, 1, 0, 0, 1, 0,
                                          1, 1, 0, 0, 1, 0, 0, 1,
                                          0, 0, 0, 0, 0, 0, 1, 0,
                                          0, 0, 1, 0, 0, 1, 1, 0])

        self.__seed = np.array([1, 0, 1, 1, 1, 0, 1])
        self.__data = np.zeros(72, dtype=int)
        self.__data[[18, 25, 41, 42, 43, 45, 61, 62, 67]] = 1
        self.__scramble = np.zeros(72, dtype=int)
        self.__scramble[[1, 2, 4, 5, 11, 12, 15, 16, 20, 23, 24, 28, 29, 30, 31, 33, 34,
                         36, 42, 47, 48, 49, 50, 51, 53, 56, 58, 61, 63, 65, 66, 71]] = 1

        self.scrambler.seed = self.__seed

    def test_sequence(self) -> None:

        scrambler = Scrambler80211a(ParametersScrambler(), 31)
        scrambler.seed = self.__sequence_seed

        sequence = np.empty(self.__sequence_check.shape, dtype=int)
        for n in range(sequence.shape[0]):

            sequence[n] = scrambler._Scrambler80211a__forward_code_bit()

        np.testing.assert_array_equal(self.__sequence_check, sequence)

    def test_scrambling(self) -> None:
        """Test the scrambling against the example tables of IEEE 802.11a.

        See also Annex G.5.2 of IEEE Std 802.11a-1999
        """

        assert_frame_equality([self.__scramble], self.scrambler.encode([self.__data]))
