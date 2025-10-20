# -*- coding: utf-8 -*-

from unittest import TestCase

from hermespy.simulation.rf.chain import RFChain, RFBlockReference
from .test_block import MockRFBlock
from unit_tests.core.test_factory import test_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestRFBlockReference(TestCase):
    """Test the RF block reference class."""

    def setUp(self) -> None:
        self.block = MockRFBlock(num_input_ports=2, num_output_ports=3)
        self.block_ref = RFBlockReference(self.block)

    def test_serialization(self) -> None:
        """Test serialization of RF block reference"""

        test_roundtrip_serialization(self, self.block_ref, additional_tags={MockRFBlock.serialization_tag(): MockRFBlock})


class TestRFChain(TestCase):
    """Test the RF chain model configuration class."""

    def setUp(self) -> None:
        self.chain = RFChain(seed=42)

        # Initialize mock blocks        
        self.input_blocks = [MockRFBlock(num_output_ports=2), MockRFBlock(num_output_ports=1)]
        self.output_blocks = [MockRFBlock(num_input_ports=2), MockRFBlock(num_input_ports=1)]
        self.block = MockRFBlock(num_input_ports=3, num_output_ports=3)

        # Add mock blocks to the chain
        self.input_blocks_refs = [self.chain.add_block(b) for b in self.input_blocks]
        self.output_blocks_refs = [self.chain.add_block(b) for b in self.output_blocks]
        self.block_ref = self.chain.add_block(self.block)

        # Connect blocks
        self.chain.connect(self.input_blocks_refs[0].o, self.block_ref.i[:2])
        self.chain.connect(self.input_blocks_refs[1].o, self.block_ref.i[2])
        self.chain.connect(self.block_ref.o[:2], self.output_blocks_refs[0].i)
        self.chain.connect(self.block_ref.o[2], self.output_blocks_refs[1].i)
    
    def test_serialization(self) -> None:
        """Test serialization of RF chain model configuration"""

        test_roundtrip_serialization(self, self.chain, additional_tags={MockRFBlock.serialization_tag(): MockRFBlock})
