import unittest
import numpy as np

from modem.coding.scrambler import Scrambler3GPP


class TestScrambler3GPP(unittest.TestCase):

    def setUp(self) -> None:

        self.scrambler = TestScrambler3GPP()

    def tearDown(self) -> None:
        pass

