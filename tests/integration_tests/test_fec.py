# -*- coding: utf-8 -*-

from itertools import chain, product
from os import path
from unittest import TestCase
from unittest.mock import patch
from tempfile import TemporaryDirectory

import ray
import numpy as np
from numpy.testing import assert_array_equal

from hermespy.simulation import SimulatedDevice
from hermespy.fec import EncoderManager, LDPCCoding, TurboCoding, RepetitionEncoder, Scrambler3GPP, BlockInterleaver, Scrambler80211a, CyclicRedundancyCheck, PolarSCCoding, ReedSolomonCoding, RSCCoding, BCHCoding
from hermespy.modem import RootRaisedCosineWaveform, SimplexLink
from hermespy.tools import db2lin

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestFEC(TestCase):
    """Test the forward error correction (FECT) integration with communication dsp layers."""

    def setUp(self) -> None:
        self.tx_device = SimulatedDevice(bandwidth=100e6, oversampling_factor=1, seed=42)
        self.rx_device = SimulatedDevice(bandwidth=100e6, oversampling_factor=1, seed=42)
        self.link = SimplexLink()
        self.tx_device.transmitters.add(self.link)
        self.rx_device.receivers.add(self.link)

        self.link.waveform = RootRaisedCosineWaveform(num_data_symbols=1024, modulation_order=64, num_preamble_symbols=0)

        self.alpha_candidates = [
            Scrambler3GPP(),
#            Scrambler80211a(),  The scrambler80211a decsrambling is incorrect
            BlockInterleaver(block_size=self.link.waveform.bits_per_frame(),interleave_blocks=8),
        ]

        self.beta_candidates = [
            CyclicRedundancyCheck(bit_block_size=60, check_block_size=4),
            RepetitionEncoder(bit_block_size=64, repetitions=3),
            LDPCCoding(100, path.join(path.dirname(path.realpath(__file__)), "..", "..", "submodules", "affect", "conf", "dec", "LDPC", "CCSDS_64_128.alist"), "", False, 10),
            TurboCoding(40, 13, 15, 100),
            PolarSCCoding(256, 512, 0.5),
#            PolarSCLCoding(256, 512, 0.1, 256),
#            ReedSolomonCoding(107, 10),  The reed solomon coding is not working and will crash
            RSCCoding(20, 46, False, 13, 15),
            BCHCoding(99, 127, 4),
        ]

    def __test_frame(self) -> None:
        """Validate the transmission of a single frame"""

        transmission = self.tx_device.transmit()
        reception = self.rx_device.receive(transmission)
        transmitted_bits = transmission.operator_transmissions[0].bits
        received_bits = reception.operator_receptions[0].bits

        assert_array_equal(transmitted_bits, received_bits)

    def test_individual_codings(self) -> None:
        """Test integration of individual coding schemes"""

        for coding in chain(self.alpha_candidates, self.beta_candidates):
            self.link.encoder_manager = EncoderManager()
            self.link.encoder_manager.add_encoder(coding)

            with self.subTest(msg="Test " + coding.__class__.__name__):
                self.__test_frame()

    def test_coding_combinations(self) -> None:
        """Test integration of combinations of coding schemes"""

        for alpha_coding, beta_coding in product(self.alpha_candidates, self.beta_candidates):
            self.link.encoder_manager = EncoderManager()
            self.link.encoder_manager.add_encoder(beta_coding)
            self.link.encoder_manager.add_encoder(alpha_coding)

            with self.subTest(msg="Testing combination " + beta_coding.__class__.__name__ + " -> " + alpha_coding.__class__.__name__):
                self.__test_frame()
