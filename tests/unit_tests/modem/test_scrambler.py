import unittest

from .test_encoder import TestAbstractEncoder
from modem.coding.scrambler import Scrambler3GPP
from modem.coding.encoder import ParametersEncoder


class TestScrambler3GPP(TestAbstractEncoder, unittest.TestCase):

    @property
    def encoder(self) -> Scrambler3GPP:
        return self.scrambler

    def setUp(self) -> None:

        self.scrambler = Scrambler3GPP(ParametersEncoder(), 31)

    def tearDown(self) -> None:
        pass
