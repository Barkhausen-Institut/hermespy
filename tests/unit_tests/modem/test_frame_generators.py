# -*- coding: utf-8 -*-
"""HermesPy FrameGenerators testing."""

from hermespy.modem.bits_source import RandomBitsSource
from hermespy.modem.frame_generator import FrameGeneratorStub
from hermespy.modem.frame_generator.scapy import FrameGeneratorScapy

import unittest
import numpy as np

from numpy.testing import assert_array_equal

from scapy.layers.dot11 import Dot11


class TestFrameGeneratorStub(unittest.TestCase):
    """Test the placeholder stub frame generator"""

    def setUp(self) -> None:
        self.fg = FrameGeneratorStub()
        self.bs = RandomBitsSource(42)

    def test_pack_unpack(self):
        for num_frame_bits in [0, 1, 8, 2**10, 11]:
            frame = self.fg.pack_frame(self.bs, num_frame_bits)
            payload = self.fg.unpack_frame(frame)
            # TODO how can this be asserted?


class TestFrameGeneratorScapy(unittest.TestCase):
    """Test the Scapy wrapper frame generator"""

    def setUp(self) -> None:
        self.packet_base = Dot11(proto=1, ID=1337, addr1='01:23:45:67:89:ab', addr2='ff:ee:dd:cc:bb:aa')
        self.packet_base_num_bits = len(self.packet_base)*8
        self.fg = FrameGeneratorScapy(self.packet_base)
        self.bs = RandomBitsSource(42)

    def test_pack_unpack(self):
        """Test pack_bits and unpack_bits with the 802.11 Scapy implementation"""

        # Try packing frames with valid number of bits
        for num_bits in np.array([8, 16, 2**10]) + self.packet_base_num_bits:
            packet = self.fg.pack_frame(self.bs, num_bits)
            payload = self.fg.unpack_frame(packet)
            # Strip the Dot11 head off the packet and
            # assert that what is left is the same payload
            payload_expected = packet[-num_bits+self.packet_base_num_bits:]
            assert_array_equal(payload_expected, payload)

        # Test a zero-sized payload
        num_bits = self.packet_base_num_bits
        packet = self.fg.pack_frame(self.bs, num_bits)
        payload = self.fg.unpack_frame(packet)
        self.assertEqual(packet.size, self.packet_base_num_bits)
        assert_array_equal(payload, [])

        # Test invalid num_bits
        for num_bits in [-1, self.packet_base_num_bits-1]:
            with self.assertRaises(ValueError):
                self.fg.pack_frame(self.bs, num_bits)

        # Test unpacking of an invalid frame
        with self.assertRaises(ValueError):
            # Assuming this is a packet from the zero-sized payload test
            self.fg.unpack_frame(packet[1:])
